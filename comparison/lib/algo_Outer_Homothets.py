# -*- coding: utf-8 -*-
"""
模型: Outer Homothets (外部单应)
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
from . import tools

def algo(data):
    """
    使用Outer Homothets方法聚合TCL的灵活性。
    【核心修正】：不再自行计算H表示，而是直接使用主框架传入的多面体。
    """
    # 1. 直接从data中解包预先计算好的多面体
    tcl_polytopes = data['tcl_polytopes']
    prices = data['prices']
    demands = data['demands']
    periods = data['periods']
    
    start_time = time.time()
    
    A_list = [p['A'] for p in tcl_polytopes]
    b_list = [p['b'] for p in tcl_polytopes]
    
    # 2. 应用Outer Homothets聚合
    A_common = A_list[0]
    b_mean = np.mean(b_list, axis=0)
    
    beta_list, t_list = [], []
    for b in b_list:
        beta, t = tools.fitHomothet(A_common, b, b_mean, False, periods) # False for outer
        beta_list.append(beta)
        t_list.append(t)
        
    beta_sum = np.sum(beta_list)
    t_sum = np.sum(t_list, axis=0)
    
    A_approx = A_common
    b_approx = b_mean * beta_sum + A_common @ t_sum
    
    P0_agg = np.sum(demands, axis=1)
    pre_algo_time = time.time() - start_time
    
    # 3. 求解优化问题
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
                    results['cost_value'] = model.ObjVal
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
                peak_var = model.addVar(lb=0.0, name="peak")
                model.addConstr(A_approx @ x <= b_approx)
                
                total_power = P0_agg + x
                model.addConstr(total_power <= peak_var)
                model.addConstr(total_power >= -peak_var)
                
                model.setObjective(peak_var, GRB.MINIMIZE)
                model.optimize()
                
                if model.status == GRB.OPTIMAL:
                    results['peak_value'] = model.ObjVal
                else:
                    results['peak_value'] = np.nan
    except gp.GurobiError as e:
        results['peak_value'] = np.nan
    peak_time = time.time() - opt_start_time_peak
    
    # 4. 格式化结果
    results['algo_time'] = pre_algo_time
    results['cost_time'] = cost_time
    results['peak_time'] = peak_time
    
    return results