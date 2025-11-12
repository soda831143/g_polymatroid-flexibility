# -*- coding: utf-8 -*-
"""
模型: Union of Homothets (单应并集) - Stage 1
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
from sklearn.cluster import KMeans
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
    使用单应并集(Union of Homothets)方法 - Stage 1。
    此实现基于从每个TCL的内部g-多胞体近似中提取的盒约束。
    """
    # 1. 解包数据并为每个TCL构建g-多胞体近似
    tcls_params = data['tcls']
    theta_a_forecast = data['theta_a_forecast']
    T = data['T']
    prices = data['prices']
    P0_agg = data['P0']
    
    # 从配置中获取簇的数量
    n_clusters = config.get('n_clusters', 3)
    
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

    # 3. 运行单应并集算法
    
    # a. 为每个设备构建约束向量 b_i
    A = np.vstack([np.eye(T), -np.eye(T), np.ones((1, T)), -np.ones((1, T))])
    
    b_list = []
    for u_max, u_min, e_f, e_i in zip(u_i_max_list, u_i_min_list, e_f_i_list, e_i_list):
        b_vec = np.concatenate([u_max, -u_min, [e_f], [-e_i]])
        b_list.append(b_vec)
    
    # b. 使用K-Means对 b 向量进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(b_list)
    
    # c. 为每个簇计算一个单应(homothet)
    A_approx_list, b_approx_list = [], []
    for j in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == j)[0]
        if len(cluster_indices) == 0:
            continue
        
        b_cluster = [b_list[i] for i in cluster_indices]
        b_mean = np.mean(b_cluster, axis=0)
        
        beta_list, t_list = [], []
        for b_k in b_cluster:
            beta, t = fit_homothet(A, b_k, b_mean, True, T)
            beta_list.append(beta)
            t_list.append(t)
            
        beta_sum = np.sum(beta_list, axis=0)
        t_sum = np.sum(t_list, axis=0)
        
        b_approx = b_mean * beta_sum + A @ t_sum
        A_approx = A
        
        A_approx_list.append(A_approx)
        b_approx_list.append(b_approx)
        
    algo_time = time.time() - start_time
    
    # 4. 求解基于单应并集的优化问题
    opt_start_time = time.time()
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            u_j = model.addMVar((T, n_clusters), lb=-GRB.INFINITY, name="u_j")
            u_dev = model.addMVar(T, lb=-GRB.INFINITY, name="u_dev")
            
            model.addConstr(u_dev == u_j.sum(axis=1))
            
            for j in range(len(A_approx_list)):
                model.addConstr(A_approx_list[j] @ u_j[:, j] <= b_approx_list[j])

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


def fit_homothet(A, b, b_mean, inner, T):
    """计算最优的偏移和缩放因子"""
    aux_len = np.shape(b_mean)[0]
    model = gp.Model("fit_homothet")
    model.Params.OutputFlag = 0
    s = model.addMVar(1)
    G = model.addMVar((aux_len, aux_len))
    r = model.addMVar(T, lb=-GRB.INFINITY)
    aux = model.addMVar(aux_len, lb=-GRB.INFINITY)
    model.setObjective(s, GRB.MINIMIZE)

    for i in range(aux_len):
        model.addConstrs((G[i, k] * A[:, j] == A[i, j] for k in range(aux_len) for j in range(T)), name=f"G_constr_{i}_{j}")

    if inner:
        model.addConstrs(aux[i] == gp.quicksum(G[i,k]*b_mean[k] for k in range(aux_len)) for i in range(aux_len))
        model.addConstr(aux <= b.reshape(aux_len, 1) @ s + A @ r)
    else: # outer
        model.addConstrs(aux[i] == gp.quicksum(G[i, k] * b[k] for k in range(aux_len)) for i in range(aux_len))
        model.addConstr(aux <= b_mean.reshape(aux_len, 1) @ s + A @ r)
        
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        print("警告: fit_homothet 求解失败, 返回默认值。")
        return 1.0, np.zeros(T)

    beta = 1 / s.X[0] if inner else s.X[0]
    t = -r.X / s.X[0] if inner else r.X
    return beta, t