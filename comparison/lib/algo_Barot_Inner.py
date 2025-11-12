# -*- coding: utf-8 -*-
"""
模型: Barot Inner Approximation
"""

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
from . import tools

def algo(data):
    """
    使用Barot等人的内部近似方法。
    该方法基于每个设备的真实多面体H表示来工作。
    【核心修正】：不再自行计算H表示，而是直接使用主框架传入的多面体。
    """
    # 1. 解包TCL数据并生成每个设备的H表示
    tcl_polytopes = data['tcl_polytopes']
    T = data['periods']
    prices = data['prices']
    P0_agg = data['P0']
    num_households = len(tcl_polytopes)
    start_time = time.time()
    list_A_i = [item['A'] for item in tcl_polytopes]
    list_b_i = [item['b'] for item in tcl_polytopes]
    H = num_households
    # a. 计算每个多面体的Chebyshev中心和半径
    u_i_cheby = np.full((T, H), np.nan)
    r_i = np.full(H, np.nan)
    for h in range(H):
        u_i_cheby[:, h], r_i[h] = tools.chebyshev_center(list_A_i[h], list_b_i[h])
    # b. 计算聚合的Chebyshev中心
    u_cheby_agg = np.sum(u_i_cheby, axis=1)
    # c. 计算lambda_i
    lambdas = np.full(H, np.nan)
    for h in range(H):
        model = gp.Model('lambda_lp')
        model.Params.OutputFlag = 0
        l = model.addVar(lb=0.0, name='l')
        u = model.addMVar(shape=T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u")
        model.setObjective(l, GRB.MAXIMIZE)
        model.addConstr(list_A_i[h] @ u <= list_b_i[h])
        for t in range(T):
            model.addConstr(u[t] - u_i_cheby[t, h] == l * (u_cheby_agg[t] - u_i_cheby[t, h]))
        model.optimize()
        if model.status == GRB.OPTIMAL:
            lambdas[h] = l.x
        else:
            lambdas[h] = 0
    # d. 找到最小的lambda
    lambda_min = np.min(lambdas)
    # e. 计算内部近似多面体的H表示 (A_inner, b_inner)
    A_inner = np.vstack(list_A_i)
    b_inner_parts = []
    for h in range(H):
        b_part = list_b_i[h] - list_A_i[h] @ (u_i_cheby[:, h] - lambdas[h] * u_i_cheby[:, h] - (1 - lambdas[h]) * u_cheby_agg)
        b_inner_parts.append(b_part)
    b_inner = np.hstack(b_inner_parts)
    algo_time = time.time() - start_time
    # 3. 求解基于近似的优化问题
    opt_start_time_cost = time.time()
    cost_value = np.nan
    try:
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
                    cost_value = model.ObjVal
    except Exception as e:
        cost_value = np.nan
    cost_time = time.time() - opt_start_time_cost
    opt_start_time_peak = time.time()
    peak_value = np.nan
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as model:
                u_dev = model.addMVar(T, lb=-GRB.INFINITY, name="u_dev")
                peak_var = model.addVar(lb=0, name="peak")
                model.addConstr(A_inner @ u_dev <= b_inner, name="approx_flex")
                total_power = P0_agg + u_dev
                model.addConstr(total_power <= peak_var)
                model.addConstr(total_power >= -peak_var)
                model.setObjective(peak_var, GRB.MINIMIZE)
                model.optimize()
                if model.status == GRB.OPTIMAL:
                    peak_value = model.ObjVal
    except Exception as e:
        peak_value = np.nan
    peak_time = time.time() - opt_start_time_peak
    # 4. 格式化结果
    results = {
        'cost_value': cost_value,
        'peak_value': peak_value,
        'algo_time': algo_time,
        'cost_time': cost_time,
        'peak_time': peak_time
    }
    return results