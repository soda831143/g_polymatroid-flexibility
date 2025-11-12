# -*- coding: utf-8 -*-
"""
模型: Homothet Projection (内部近似)
"""

from . import tools
import time
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
import sys
import os

# 动态添加项目根目录到sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from flexitroid.utils.tcl_utils import get_true_tcl_polytope_H_representation
except ImportError:
    raise ImportError("无法从 'flexitroid.utils.tcl_utils' 导入 'get_true_tcl_polytope_H_representation'。")

def run(data, config, identifier=None):
    """
    使用Homothet Projection方法进行内部近似。
    该方法基于每个设备的真实多面体H表示来工作。
    """
    # 1. 解包TCL数据并生成每个设备的H表示
    tcls_params = data['tcls']
    theta_a_forecast = data['theta_a_forecast']
    T = data['T']
    prices = data['prices']
    P0_agg = data['P0']
    num_households = len(tcls_params)
    
    start_time = time.time()
    
    list_A_i, list_b_i = [], []
    for params in tcls_params:
        R_th, C_th, P_m, eta = params['R_th'], params['C_th'], params['P_m'], params['eta']
        theta_r, delta_val = params['theta_r'], params['delta_val']
        delta = params.get('delta', 1.0)
        x0 = params.get('x0', 0.0)
        a = np.exp(-delta / (R_th * C_th))
        b_coef = R_th * eta
        
        P0_unconstrained = (theta_a_forecast - theta_r) / b_coef
        P0_forecast = np.maximum(0, P0_unconstrained)
        u_min = -P0_forecast
        u_max = P_m - P0_forecast
        x_plus = (C_th * delta_val) / eta
        x_min_phys, x_max_phys = -x_plus, x_plus
        
        A_i, b_i = get_true_tcl_polytope_H_representation(
            T, a, x0, u_min, u_max, x_min_phys, x_max_phys, delta
        )
        list_A_i.append(A_i)
        list_b_i.append(b_i)

    # 2. 运行Homothet Projection算法
    A_i_list = list_A_i
    b_i_list = list_b_i
    H = num_households

    # a. 计算每个多面体的Chebyshev中心和半径
    u_i_cheby = np.full((T, H), np.NaN)
    for h in range(H):
        u_i_cheby[:, h], _ = tools.chebyshev_center(A_i_list[h], b_i_list[h])
        
    # b. 定义投影矩阵 P
    P = np.eye(T)
    
    # c. 求解 alpha
    model = gp.Model('alpha_lp')
    model.Params.OutputFlag = 0
    alpha = model.addMVar(shape=H, lb=0.0, ub=1.0, name='alpha')
    w = model.addMVar(shape=(T, H), lb=-GRB.INFINITY, name='w')
    v = model.addMVar(shape=T, lb=-GRB.INFINITY, name='v')
    
    model.setObjective(gp.quicksum(alpha), GRB.MAXIMIZE)
    
    for h in range(H):
        model.addConstr(A_i_list[h] @ w[:, h] <= alpha[h] * b_i_list[h])
    
    model.addConstr(v == w.sum(axis=1))
    model.addConstr(v == P @ (np.sum(u_i_cheby, axis=1) - (w - alpha.reshape(1, H) * u_i_cheby).sum(axis=1)))
    
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        raise RuntimeError('Homothet projection LP has not been solved to optimality.')
    
    alpha_sol = alpha.X
    
    # d. 计算内部近似的H表示
    A_inner = np.vstack(A_i_list)
    b_inner_parts = []
    for h in range(H):
        b_part = alpha_sol[h] * b_i_list[h] + A_i_list[h] @ (1 - alpha_sol[h]) @ u_i_cheby[:, h]
        b_inner_parts.append(b_part)
    b_inner = np.hstack(b_inner_parts)
    
    algo_time = time.time() - start_time
    
    # 3. 求解基于近似的优化问题
    opt_start_time = time.time()
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            u_dev = model.addMVar(T, lb=-GRB.INFINITY, name="u_dev")
            model.addConstr(A_inner @ u_dev <= b_inner, name="approx_flex")
            
            total_power = P0_agg + u_dev
            objective = prices @ total_power
            model.setObjective(objective, GRB.MINIMIZE)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                opt_cost = model.ObjVal
                opt_u_dev = u_dev.X
                opt_peak = np.max(P0_agg + opt_u_dev)
            else:
                opt_cost = -1
                opt_u_dev = np.zeros(T)
                opt_peak = np.max(P0_agg)
                
    opt_time = time.time() - opt_start_time

    # 4. 格式化结果
    results = {
        'identifier': identifier,
        'cost': opt_cost,
        'peak': opt_peak,
        'power_dev': opt_u_dev.tolist(),
        'algo_time': algo_time,
        'opt_time': opt_time
    }
    
    return results