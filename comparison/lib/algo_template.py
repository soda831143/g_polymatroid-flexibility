# -*- coding: utf-8 -*-
"""
算法模板
"""

import time
import numpy as np
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None
import sys
import os

# 建议将此段代码添加到所有新算法中，以确保能正确导入项目模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # 示例导入：根据需要导入TCL类或其他工具
    from flexitroid.devices.tcl import TCL
except ImportError:
    # 建议保留此错误处理，以便在环境设置不正确时提供清晰的提示
    raise ImportError("无法从 'flexitroid' 导入模块。请确保项目已正确安装或路径已设置。")


def run(data, config, identifier=None):
    """
    这是一个比较算法的模板。

    Args:
        data (dict): 包含测试数据的字典。新数据结构包含:
            - 'tcls' (list): TCL参数字典的列表。
            - 'theta_a_forecast' (np.ndarray): 温度预测向量。
            - 'T' (int): 时间步长。
            - 'prices' (np.ndarray): 电价向量。
            - 'P0' (np.ndarray): 聚合的基线功率向量。
        config (dict): 包含算法特定配置的字典 (例如，簇的数量)。
        identifier (str, optional): 算法的唯一标识符。

    Returns:
        dict: 包含结果的字典，必须包含 'cost', 'peak', 'power_dev', 'algo_time', 'opt_time'。
    """
    # -------------------------------------------------------------------------
    # 1. 解包数据和参数
    # -------------------------------------------------------------------------
    tcls_params = data['tcls']
    theta_a_forecast = data['theta_a_forecast']
    T = data['T']
    prices = data['prices']
    P0_agg = data['P0']
    
    # 算法计算开始计时
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # 2. (可选) 为每个TCL实例化对象并提取约束
    #    这是处理基于盒约束近似的算法的推荐模式。
    # -------------------------------------------------------------------------
    tcl_fleet = []
    for params in tcls_params:
        try:
            full_params = {**params, 'T': T}
            tcl_fleet.append(TCL(full_params, build_g_poly=True, theta_a_forecast=theta_a_forecast))
        except Exception as e:
            print(f"警告: 模板中创建TCL对象失败: {e}")
            continue
            
    if not tcl_fleet:
        # 在没有可用设备的情况下返回默认失败结果
        return {'identifier': identifier, 'cost': -1, 'peak': -1, 'power_dev': [0]*T, 'algo_time': 0, 'opt_time': 0}

    # 从每个TCL的内部g-多胞体近似中提取盒约束
    u_i_min = np.array([tcl._internal_g_poly.u_min for tcl in tcl_fleet])
    u_i_max = np.array([tcl._internal_g_poly.u_max for tcl in tcl_fleet])
    e_i = np.array([tcl._internal_g_poly.x_min[-1] for tcl in tcl_fleet])
    e_f_i = np.array([tcl._internal_g_poly.x_max[-1] for tcl in tcl_fleet])
    
    # -------------------------------------------------------------------------
    # 3. 在此实现你的聚合算法
    #    示例：简单的Minkowski和（聚合盒约束）
    # -------------------------------------------------------------------------
    agg_u_min = np.sum(u_i_min, axis=0)
    agg_u_max = np.sum(u_i_max, axis=0)
    agg_e_min = np.sum(e_i)
    agg_e_max = np.sum(e_f_i)

    # 算法计算结束计时
    algo_time = time.time() - start_time
    
    # -------------------------------------------------------------------------
    # 4. 求解基于聚合灵活性的优化问题
    # -------------------------------------------------------------------------
    opt_start_time = time.time()
    
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as model:
                # 定义聚合功率偏差变量
                u_dev = model.addMVar(T, lb=agg_u_min, ub=agg_u_max, name="u_dev")
                
                # 添加聚合能量约束
                model.addConstr(gp.quicksum(u_dev) >= agg_e_min, name="e_min_agg")
                model.addConstr(gp.quicksum(u_dev) <= agg_e_max, name="e_max_agg")

                # 定义目标函数
                total_power = P0_agg + u_dev
                objective = prices @ total_power
                model.setObjective(objective, GRB.MINIMIZE)
                model.optimize()

                if model.status == GRB.OPTIMAL:
                    opt_cost = model.ObjVal
                    opt_u_dev = u_dev.X
                    opt_peak = np.max(P0_agg + opt_u_dev)
                else:
                    print(f"警告: 模板优化求解失败，状态码: {model.status}")
                    opt_cost = -1
                    opt_u_dev = np.zeros(T)
                    opt_peak = np.max(P0_agg)
    except Exception as e:
        print(f"优化过程中发生错误: {e}")
        opt_cost = -1
        opt_u_dev = np.zeros(T)
        opt_peak = np.max(P0_agg)

    # 优化过程结束计时
    opt_time = time.time() - opt_start_time

    # -------------------------------------------------------------------------
    # 5. 格式化并返回结果
    # -------------------------------------------------------------------------
    results = {
        'identifier': identifier,
        'cost': opt_cost,
        'peak': opt_peak,
        'power_dev': opt_u_dev.tolist(),
        'algo_time': algo_time,
        'opt_time': opt_time
    }
    
    return results