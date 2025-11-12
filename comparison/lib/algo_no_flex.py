# -*- coding: utf-8 -*-
"""
模型(0)：无灵活性
"""

import numpy as np
import time

def algo(data):
    """
    无灵活性基准算法：所有设备按基线需求运行，无任何灵活性调度。
    Args:
        data (dict): 输入数据，包含'demands' (TxH), 'prices' (T,), 'periods', 'objectives', 'sample'等。
    Returns:
        dict: 结果字典，字段与主框架一致。
    """
    res = {}
    res['sample'] = data.get('sample', None)
    dt = data.get('dt', 1.0)
    T_eval = data['periods']
    prices = data['prices']
    demands = data['demands']  # shape: (T, H)
    objectives = data.get('objectives', ['cost', 'peak'])

    demand_aggr = np.sum(demands, axis=1)  # shape: (T,)
    x_sol = np.zeros_like(demand_aggr)     # 无灵活性，偏差为0

    start_time = time.time()
    for obj in objectives:
        if obj == 'cost':
            res['cost_time'] = 0.0
            res['cost_value'] = float(np.dot(prices[:T_eval], demand_aggr[:T_eval] + x_sol[:T_eval]) * dt)
        elif obj == 'peak':
            res['peak_time'] = 0.0
            res['peak_value'] = float(np.linalg.norm(demand_aggr[:T_eval] + x_sol[:T_eval], ord=np.inf))
        else:
            raise ValueError(f'Objective "{obj}" not implemented.')
    res['algo_time'] = time.time() - start_time
    res['status'] = 'success'
    return res
