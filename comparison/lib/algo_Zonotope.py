# -*- coding: utf-8 -*-
"""
模型: Zonotope (使用盒约束近似)
"""

import numpy as np
import time
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
from .tools import generateZonotope, getMatrixC, getHyperplaneOffset, optimalZonotopeMaxNorm, getVectord

def algo(data):
    """
    Algorithm: Zonotope Aggregation for TCL
    使用真正的Zonotope方法聚合TCL的灵活性集合
    """
    # 1. 提取数据
    dt = data.get('dt', 1)
    prices = data['prices']
    demands = data['demands']
    periods = data['periods']
    tcl_polytopes = data['tcl_polytopes']
    start_time = time.time()
    # 2. 直接用多面体
    A_list = [item['A'] for item in tcl_polytopes]
    b_list = [item['b'] for item in tcl_polytopes]
    # 3. 使用原始Zonotope算法的聚合方法
    Z, G = generateZonotope(periods, [0]*periods)
    C = getMatrixC(periods)
    Zonotope_list = []
    for i, (A_tcl, b_tcl) in enumerate(zip(A_list, b_list)):
        d_new = getHyperplaneOffset(A_tcl, C, b_tcl, periods)
        Z_opt = optimalZonotopeMaxNorm(A_tcl, b_tcl, G, C, d_new)
        Zonotope_list.append(Z_opt)
    # 4. 计算Zonotope的Minkowski和
    Zonotope_minkowski_list = []
    for l in range(len(Zonotope_list[0])):
        s = np.array(Zonotope_list[0][l])
        for h in range(1, len(Zonotope_list)):
            s = s + np.array(Zonotope_list[h][l])
        Zonotope_minkowski_list.append(list(s))
    # 5. 生成聚合约束
    b_approx = getVectord(C, Zonotope_minkowski_list, periods)
    A_approx = C
    P0_agg = np.sum(demands, axis=1)
    pre_algo_time = time.time() - start_time
    # 6. 求解优化问题
    results = {'status': 'error: optimization failed'}
    # Cost-Minimization
    opt_start_time_cost = time.time()
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as model:
                x = model.addMVar(periods, lb=-GRB.INFINITY, name="x")
                model.addConstr(A_approx @ x <= b_approx)
                total_power = P0_agg + x
                objective = prices @ total_power
                model.setObjective(objective, GRB.MINIMIZE)
                model.optimize()
                if model.status == GRB.OPTIMAL:
                    results['cost_value'] = model.ObjVal * dt
                    results['status'] = 'success'
                else:
                    results['cost_value'] = np.nan
    except gp.GurobiError as e:
        results['cost_value'] = np.nan
    cost_time = time.time() - opt_start_time_cost
    # Peak-Minimization  
    opt_start_time_peak = time.time()
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as model:
                x = model.addMVar(periods, lb=-GRB.INFINITY, name="x")
                peak_var = model.addVar(lb=0, name="peak")
                model.addConstr(A_approx @ x <= b_approx)
                total_power = P0_agg + x
                for t in range(periods):
                    model.addConstr(total_power[t] <= peak_var)
                    model.addConstr(-total_power[t] <= peak_var)
                model.setObjective(peak_var, GRB.MINIMIZE)
                model.optimize()
                if model.status == GRB.OPTIMAL:
                    results['peak_value'] = model.ObjVal
                else:
                    results['peak_value'] = np.nan
    except gp.GurobiError as e:
        results['peak_value'] = np.nan
    peak_time = time.time() - opt_start_time_peak
    # 返回结果
    algo_res = {
        'sample': data.get('sample', 0),
        'algo time': pre_algo_time,
        'cost_time': cost_time,
        'peak_time': peak_time,
        'cost_value': results['cost_value'],
        'peak_value': results['peak_value'],
        'cost_im_en': np.nan,
        'peak_im_en': np.nan
    }
    return algo_res