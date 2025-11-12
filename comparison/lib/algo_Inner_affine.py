# -*- coding: utf-8 -*-
"""
模型: Inner Affine (内部仿射) - 根据 Al Taha et al. (2024) 论文修正
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

def solve_individual_affine_lp(H, h_i, h_0):
    """
    根据 Al Taha et al. 论文的公式 (28)，为单个DER求解最优的仿射变换 (gamma_i, Gamma_i)。
    """
    m, n = H.shape
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            model.setParam('OutputFlag', 0)
            # 定义优化变量
            gamma_i = model.addMVar(n, lb=-GRB.INFINITY, name="gamma_i")
            Gamma_i = model.addMVar((n, n), lb=-GRB.INFINITY, name="Gamma_i")
            Lambda_i = model.addMVar((m, m), lb=0.0, name="Lambda_i") # Lambda_i >= 0

            # 目标函数: maximize Tr(Gamma_i)
            model.setObjective(gp.quicksum(Gamma_i[i, i] for i in range(n)), GRB.MAXIMIZE)

            # 添加约束
            # 约束1: Lambda_i * H = H * Gamma_i
            model.addConstr(Lambda_i @ H == H @ Gamma_i)
            
            # 约束2: Lambda_i * h_0 <= h_i - H * gamma_i
            model.addConstr(Lambda_i @ h_0 <= h_i - H @ gamma_i)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                return gamma_i.X, Gamma_i.X
            else:
                return None, None

def algo(data):
    """
    使用通用仿射变换 (General Affine Transformation) 的内近似方法。
    此实现基于 Al Taha et al. (2024) 论文，对每个独立的DER灵活性多面体
    进行最优仿射变换，然后将变换参数相加得到最终的聚合模型。
    """
    # 1. 解包数据
    tcl_polytopes = data['tcl_polytopes']
    prices = data['prices']
    P0_agg = data['P0']
    T = data['periods']
    
    start_time = time.time()
    
    if not tcl_polytopes:
        return {'cost_value': np.nan, 'peak_value': np.nan, 'status': 'error: no tcl objects'}

    # 2. 准备H表示和基础多面体
    # 所有DER共享同一个H矩阵
    H_matrix = tcl_polytopes[0]['A']
    # 每个DER有自己的h向量 (即b向量)
    h_list = [p['b'] for p in tcl_polytopes]
    # 基础多面体的h_0向量为所有h向量的平均值
    h_0 = np.mean(h_list, axis=0)

    # 3. 为每个DER独立求解最优仿射变换
    gamma_list = []
    Gamma_list = []
    print(f"  运行 Inner Affine 算法: 为 {len(h_list)} 个设备独立求解LP...")
    for i, h_i in enumerate(h_list):
        gamma_i, Gamma_i = solve_individual_affine_lp(H_matrix, h_i, h_0)
        if gamma_i is not None:
            gamma_list.append(gamma_i)
            Gamma_list.append(Gamma_i)
        else:
            print(f"    - 警告: 设备 {i+1} 的仿射变换求解失败，已跳过。")

    if not gamma_list:
        return {'cost_value': np.nan, 'peak_value': np.nan, 'status': 'error: all affine fits failed'}

    # 4. 聚合变换参数
    p_bar = np.sum(gamma_list, axis=0)
    P_matrix = np.sum(Gamma_list, axis=0)
    
    algo_time = time.time() - start_time
    
    # 5. 构建并求解最终的优化问题
    # 近似可行域为 A_approx = {u | H @ P_inv @ (u - p_bar) <= h_0}
    # 这等价于 H @ v <= h_0，其中 u = p_bar + P @ v
    results = {'status': 'success', 'algo_time': algo_time}

    try:
        P_inv = np.linalg.inv(P_matrix)
    except np.linalg.LinAlgError:
        return {'cost_value': np.nan, 'peak_value': np.nan, 'status': 'error: Aggregate P matrix is singular'}

    # --- Cost-Minimization ---
    opt_start_time_cost = time.time()
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as model:
                model.setParam('OutputFlag', 0)
                # 决策变量是在基础多面体空间中的 v
                v = model.addMVar(T, lb=-GRB.INFINITY, name="v")
                # 约束 v 在基础多面体内
                model.addConstr(H_matrix @ v <= h_0)
                
                # 将 v 映射回原始功率空间 u
                u_dev = p_bar + P_matrix @ v
                
                total_power = P0_agg + u_dev
                objective = prices @ total_power
                model.setObjective(objective, GRB.MINIMIZE)
                model.optimize()
                
                if model.status == GRB.OPTIMAL:
                    results['cost_value'] = model.ObjVal
                else:
                    results['cost_value'] = np.nan
    except gp.GurobiError:
        results['cost_value'] = np.nan
    results['cost_time'] = time.time() - opt_start_time_cost

    # --- Peak-Minimization ---
    opt_start_time_peak = time.time()
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as model:
                model.setParam('OutputFlag', 0)
                v = model.addMVar(T, lb=-GRB.INFINITY, name="v")
                peak_var = model.addVar(name="peak", lb=0.0)
                model.addConstr(H_matrix @ v <= h_0)
                
                u_dev = p_bar + P_matrix @ v
                total_power = P0_agg + u_dev
                
                model.addConstr(total_power <= peak_var)
                model.addConstr(total_power >= -peak_var)

                model.setObjective(peak_var, GRB.MINIMIZE)
                model.optimize()
                
                if model.status == GRB.OPTIMAL:
                    results['peak_value'] = model.ObjVal
                else:
                    results['peak_value'] = np.nan
    except gp.GurobiError:
        results['peak_value'] = np.nan
    results['peak_time'] = time.time() - opt_start_time_peak
    
    return results