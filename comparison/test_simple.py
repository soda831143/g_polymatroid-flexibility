# -*- coding: utf-8 -*-
"""
简单测试脚本 - 调试算法接口问题
"""

import numpy as np
import sys
import os

# 添加路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# 导入算法
from comparison.lib import algo_exact
from comparison.lib import algo_no_flex

def create_simple_test_data():
    """创建简单的测试数据"""
    periods = 6  # 简化为6个时间段
    num_households = 3  # 简化为3个家庭
    
    # 温度预测
    theta_a_forecast = np.array([25.0, 26.0, 27.0, 26.5, 25.5, 24.5])
    
    # 电价
    prices = np.array([0.1, 0.1, 0.25, 0.25, 0.1, 0.1])
    
    # 生成TCL参数
    tcls = []
    demands = []
    
    for h in range(num_households):
        # 固定参数
        C_th = 3.0
        R_th = 2.5
        P_m = 2.0
        eta = 2.8
        theta_r = 24.0
        delta_val = 1.0
        delta = 0.1
        
        # 计算导出参数
        a = np.exp(-delta / (R_th * C_th))
        b_coef = R_th * eta
        
        # 基线功率
        P0_unconstrained = (theta_a_forecast - theta_r) / b_coef
        P0_forecast = np.maximum(0.1, P0_unconstrained)
        
        # 控制变量边界
        u_min = -P0_forecast
        u_max = P_m - P0_forecast
        
        # 状态约束
        x_plus = (C_th * delta_val) / eta
        x_min_phys, x_max_phys = -x_plus, x_plus
        
        # 初始状态
        x0 = 0.0
        
        tcl_params = {
            'C_th': C_th,
            'R_th': R_th, 
            'P_m': P_m,
            'eta': eta,
            'theta_r': theta_r,
            'delta_val': delta_val,
            'delta': delta,
            'x0': x0,
            'theta_a_forecast': theta_a_forecast,
            'a': a,
            'b': b_coef,
            'u_min': u_min,
            'u_max': u_max,
            'x_min_phys': x_min_phys,
            'x_max_phys': x_max_phys,
            'P0_forecast': P0_forecast,
            'T': periods
        }
        
        tcls.append(tcl_params)
        demands.append(P0_forecast)
    
    demands = np.array(demands).T  # Shape: (periods, households)
    
    return {
        'tcls': tcls,
        'theta_a_forecast': theta_a_forecast,
        'prices': prices,
        'demands': demands,
        'periods': periods,
        'households': num_households,
        'dt': 0.25,
        'objectives': ['cost', 'peak'],
        'T': periods,
        'P0': np.sum(demands, axis=1)
    }

def test_algorithm(algo_name, algo_module, data):
    """测试单个算法"""
    print(f"测试算法: {algo_name}")
    
    try:
        if hasattr(algo_module, 'run'):
            result = algo_module.run(data)
        elif hasattr(algo_module, 'algo'):
            result = algo_module.algo(data)
        else:
            print(f"  错误: 算法模块没有run或algo函数")
            return
            
        print(f"  成功! 结果类型: {type(result)}")
        if isinstance(result, dict):
            for key in ['cost', 'peak', 'cost_value', 'peak_value']:
                if key in result:
                    print(f"    {key}: {result[key]}")
        
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("创建测试数据...")
    data = create_simple_test_data()
    
    print(f"数据结构:")
    print(f"  periods: {data['periods']}")
    print(f"  households: {data['households']}")
    print(f"  demands shape: {data['demands'].shape}")
    print(f"  tcls count: {len(data['tcls'])}")
    print(f"  tcl[0] keys: {list(data['tcls'][0].keys())}")
    print()
    
    # 测试算法
    test_algorithm("No Flexibility", algo_no_flex, data)
    print()
    test_algorithm("Exact Minkowski", algo_exact, data) 