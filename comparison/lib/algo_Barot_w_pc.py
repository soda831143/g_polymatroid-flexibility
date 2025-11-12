# -*- coding: utf-8 -*-
"""
模型: Barot (带完美预知)
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
import sys
import os

from . import tools

# 动态添加项目根目录到sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from flexitroid.devices.tcl import TCL
except ImportError:
    raise ImportError("无法从 'flexitroid.devices.tcl' 导入 TCL 类。")


def run(data, config, identifier=None):
    """
    使用Barot方法（带完美预知变体）。
    此实现基于从每个TCL的内部g-多胞体近似中提取的盒约束。
    """
    # 1. 解包数据并为每个TCL构建g-多胞体近似
    tcls_params = data['tcls']
    theta_a_forecast = data['theta_a_forecast']
    T = data['T']
    prices = data['prices']
    P0_agg = data['P0']
    
    start_time = time.time()
    
    tcl_fleet = []
    for params in tcls_params:
        try:
            full_params = {**params, 'T': T}
            tcl_fleet.append(TCL(full_params, build_g_poly=True, theta_a_forecast=theta_a_forecast))
        except Exception as e:
            print(f"警告: 创建TCL对象失败: {e}")
            continue
            
    if not tcl_fleet:
        return {'identifier': identifier, 'cost': -1, 'peak': -1, 'power_dev': [0]*T, 'algo_time': 0, 'opt_time': 0}

    # 2. 从每个TCL的内部近似中提取盒约束
    u_i_min_list = [tcl._internal_g_poly.u_min for tcl in tcl_fleet]
    u_i_max_list = [tcl._internal_g_poly.u_max for tcl in tcl_fleet]
    e_i_list = [tcl._internal_g_poly.x_min[-1] for tcl in tcl_fleet]
    e_f_i_list = [tcl._internal_g_poly.x_max[-1] for tcl in tcl_fleet]

    # 3. 运行Barot(带预处理)算法
    
    # a. 为每个设备构建约束 A u <= b_i
    # A 矩阵对于所有盒约束都是相同的
    A = np.vstack([np.eye(T), -np.eye(T), np.ones((1, T)), -np.ones((1, T))])
    
    b_list = []
    for u_max, u_min, e_f, e_i in zip(u_i_max_list, u_i_min_list, e_f_i_list, e_i_list):
        b_vec = np.concatenate([u_max, -u_min, [e_f], [-e_i]])
        b_list.append(b_vec)
    
    # b. 计算预处理矩阵 L_i 并应用
    b_preconditioned_list = []
    for b_i in b_list:
        L_i = tools.preconditioning_matrix(A, b_i)
        A_preconditioned = L_i @ A
        b_preconditioned = L_i @ b_i
        b_preconditioned_list.append(b_preconditioned)
    
    # c. 聚合预处理后的约束
    A_approx = A_preconditioned # 预处理后所有A_i都相同
    b_approx = np.sum(b_preconditioned_list, axis=0)

    algo_time = time.time() - start_time
    
    # 4. 求解基于聚合约束的优化问题
    opt_start_time = time.time()
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            u_dev = model.addMVar(T, lb=-GRB.INFINITY, name="u_dev")
            
            model.addConstr(A_approx @ u_dev <= b_approx, name="approx_flex")

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

    # 5. 格式化结果
    results = {
        'identifier': identifier,
        'cost': opt_cost,
        'peak': opt_peak,
        'power_dev': opt_u_dev.tolist(),
        'algo_time': algo_time,
        'opt_time': opt_time
    }
    
    return results
    
        
        
    
    
    
    
    
    