# algo_exact.py (Corrected Version)

import numpy as np
try:
    import gurobipy as gp
except ImportError:
    gp = None
except ImportError:
    pass

try:
    from gurobipy import GRB
except ImportError:
    # GRB constants defined as None
    GRB = None
import time

def algo(data):
    """
    通过求解一个大型联合优化问题来计算聚合灵活性（精确Minkowski和）。
    【修正版】：确保物理模型与近似算法使用的模型完全一致。
    """
    # 1. 解包数据和参数
    tcl_objs = data['tcl_objs']
    prices = data['prices']
    demands = data['demands']  # (T, H)
    periods = data['periods']
    num_households = data['households']
    P0_agg = np.sum(demands, axis=1) # (T,)

    start_time = time.time()
    
    # 2. 为每个TCL提取详细参数
    u_min_all, u_max_all = [], []
    x_min_phys_all, x_max_phys_all = [], []
    a_all, delta_all, x0_all = [], [], []

    for i, tcl in enumerate(tcl_objs):
        params = tcl.tcl_params
        a = params['a']
        delta = params['delta']
        P_m = params['P_m']
        x0 = params.get('x0', 0.0)
        
        # 状态物理边界
        x_plus = (params['C_th'] * params['delta_val']) / params['eta'] if params['eta'] > 0 else 0
        x_min_phys_all.append(-x_plus)
        x_max_phys_all.append(x_plus)
        
        # 控制变量边界
        P0_forecast = demands[:, i]
        u_min_all.append(-P0_forecast)
        u_max_all.append(P_m - P0_forecast)
        
        a_all.append(a)
        delta_all.append(delta)
        x0_all.append(x0)

    u_min_all = np.array(u_min_all).T
    u_max_all = np.array(u_max_all).T
    
    pre_algo_time = time.time() - start_time
    
    # 【诊断输出】打印第一个TCL的边界值，用于与G-Poly算法对比
    print(f"\n[Exact] TCL 0 物理边界:")
    print(f"  x_min_phys={x_min_phys_all[0]:.4f}, x_max_phys={x_max_phys_all[0]:.4f}")
    print(f"  a={a_all[0]:.4f}, delta={delta_all[0]:.4f}")
    print(f"  u_min_phys范围=[{u_min_all[:, 0].min():.4f}, {u_min_all[:, 0].max():.4f}]")
    print(f"  u_max_phys范围=[{u_max_all[:, 0].min():.4f}, {u_max_all[:, 0].max():.4f}]")
    
    # 3. 求解优化问题
    results = {'status': 'error: optimization failed'}
    results['algo_time'] = pre_algo_time  # 【修复】记录预处理时间

    # --- Cost-Minimization ---
    opt_start_time_cost = time.time()
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as model:
                # u[t,i] 是 u_i(t)
                u = model.addMVar((periods, num_households), lb=u_min_all, ub=u_max_all, name="u")
                # x[i,t] 是 x_i(t)
                x = model.addMVar((num_households, periods + 1), lb=-GRB.INFINITY, name="x")
                u_agg = model.addMVar(periods, lb=-GRB.INFINITY, name="u_agg")

                for i in range(num_households):
                    model.addConstr(x[i, 0] == x0_all[i])
                    
                    # 【关键修正】: 状态约束在 t=0 时刻也必须满足
                    model.addConstr(x[i, 0] >= x_min_phys_all[i])
                    model.addConstr(x[i, 0] <= x_max_phys_all[i])
                    
                    # 循环 periods 次, 从 t=0 到 T-1
                    for t in range(periods):
                        # 【核心修正】: x_{t+1} = a*x_t + delta*u_t
                        # Gurobi变量索引从0开始, x[i, t+1]对应x_i(t+1), x[i,t]对应x_i(t), u[t,i]对应u_i(t)
                        model.addConstr(x[i, t + 1] == a_all[i] * x[i, t] + delta_all[i] * u[t, i])
                        
                        # 状态约束施加在 t+1 时刻的状态上 (对所有未来时刻)
                        model.addConstr(x[i, t + 1] >= x_min_phys_all[i])
                        model.addConstr(x[i, t + 1] <= x_max_phys_all[i])
                
                model.addConstrs((u_agg[t] == gp.quicksum(u[t, i] for i in range(num_households)) for t in range(periods)))
                
                total_power = P0_agg + u_agg
                objective = prices @ total_power
                model.setObjective(objective, GRB.MINIMIZE)
                model.optimize()

                if model.status == GRB.OPTIMAL:
                    results['cost_value'] = model.ObjVal
                    results['status'] = 'success'
                else:
                    results['cost_value'] = np.nan
    except gp.GurobiError as e:
        print(f"Gurobi Error during cost optimization: {e}")
        results['cost_value'] = np.nan
    results['cost_time'] = time.time() - opt_start_time_cost

    # --- Peak-Minimization ---
    opt_start_time_peak = time.time()
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as model:
                u = model.addMVar((periods, num_households), lb=u_min_all, ub=u_max_all, name="u")
                x = model.addMVar((num_households, periods + 1), lb=-GRB.INFINITY, name="x")
                u_agg = model.addMVar(periods, lb=-GRB.INFINITY, name="u_agg")
                peak_var = model.addVar(name="peak", lb=0.0) # 峰值必须非负

                for i in range(num_households):
                    model.addConstr(x[i, 0] == x0_all[i])
                    
                    # 【关键修正】: 状态约束在 t=0 时刻也必须满足
                    model.addConstr(x[i, 0] >= x_min_phys_all[i])
                    model.addConstr(x[i, 0] <= x_max_phys_all[i])
                    
                    for t in range(periods):
                        model.addConstr(x[i, t + 1] == a_all[i] * x[i, t] + delta_all[i] * u[t, i])
                        model.addConstr(x[i, t + 1] >= x_min_phys_all[i])
                        model.addConstr(x[i, t + 1] <= x_max_phys_all[i])
                
                model.addConstrs((u_agg[t] == gp.quicksum(u[t, i] for i in range(num_households)) for t in range(periods)))
                
                total_power = P0_agg + u_agg
                
                # 【核心修正】: 正确定义L-infinity范数 (peak_var >= |total_power|)
                model.addConstr(total_power <= peak_var, name="peak_upper")
                model.addConstr(total_power >= -peak_var, name="peak_lower")
                
                model.setObjective(peak_var, GRB.MINIMIZE)
                model.optimize()
                
                if model.status == GRB.OPTIMAL:
                    results['peak_value'] = model.ObjVal
                    if results.get('status') != 'success': results['status'] = 'success'
                else:
                    results['peak_value'] = np.nan
    except gp.GurobiError as e:
        print(f"Gurobi Error during peak optimization: {e}")
        results['peak_value'] = np.nan
    results['peak_time'] = time.time() - opt_start_time_peak

    results['algo_time'] = pre_algo_time
    return results