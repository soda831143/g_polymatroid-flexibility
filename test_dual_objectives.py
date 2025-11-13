#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双目标对比测试：成本vs峰值优化
测试三种算法在两个不同目标上的性能
"""

import sys
import os
import numpy as np
import pandas as pd
import time

# 路径设置
project_root = os.path.abspath(os.path.dirname(__file__))
lib_path = os.path.join(project_root, 'comparison', 'lib')
if project_root not in sys.path: sys.path.insert(0, project_root)
if lib_path not in sys.path: sys.path.insert(0, lib_path)

from flexitroid.devices.tcl import TCL
from comparison.lib import (
    algo_no_flex,
    algo_g_polymatroid_transform_det,
    algo_g_polymatroid_jcc_sro,
    algo_g_polymatroid_jcc_resro
)

# 固定参数分布
FIXED_PARAM_DIST = {
    'R_th_range': (2.0, 3.0), 
    'C_th_range': (5.0, 15.0), 
    'P_m_range': (10.0, 20.0),
    'eta_range': (2.5, 3.5), 
    'theta_r_range': (22.0, 23.0), 
    'delta_val_range': (1.0, 2.0)
}

def generate_tcl_data(num_households=10, periods=24):
    """生成TCL群体数据"""
    np.random.seed(42)
    T = periods
    
    # 生成预测温度
    base_temp = 27.0
    temp_amplitude = 7.0
    time_hours = np.arange(T)
    theta_a_forecast = base_temp + temp_amplitude * np.cos(2 * np.pi * (time_hours - 15) / 24)
    theta_a_forecast = np.clip(theta_a_forecast, 20.0, 35.0)
    
    # 生成电价
    prices = np.ones(T) * 0.10
    prices[0:8] = 0.05
    prices[8:13] = 0.15
    prices[13:21] = 0.60
    prices[16:19] = 0.80
    prices[21:24] = 0.12
    
    # 生成TCL群体
    tcl_objs_list = []
    P0_individual_list = []
    
    attempts = 0
    max_attempts = num_households * 100
    
    while len(tcl_objs_list) < num_households and attempts < max_attempts:
        attempts += 1
        
        C_th = np.random.uniform(*FIXED_PARAM_DIST['C_th_range'])
        R_th = np.random.uniform(*FIXED_PARAM_DIST['R_th_range'])
        P_m = np.random.uniform(*FIXED_PARAM_DIST['P_m_range'])
        eta = np.random.uniform(*FIXED_PARAM_DIST['eta_range'])
        theta_r = np.random.uniform(*FIXED_PARAM_DIST['theta_r_range'])
        delta_val = np.random.uniform(*FIXED_PARAM_DIST['delta_val_range'])
        
        a = 1 - 1/(R_th * C_th)
        b_coef = R_th * eta
        delta = 1.0
        x0 = 0.0
        
        tcl_params = {
            'T': T,
            'C_th': C_th, 'R_th': R_th, 'P_m': P_m, 'eta': eta, 'theta_r': theta_r,
            'delta_val': delta_val, 'delta': delta, 'x0': x0, 'a': a, 'b': b_coef,
            'theta_a_forecast': theta_a_forecast, 'P_min': 0
        }
        
        try:
            tcl = TCL(tcl_params, build_g_poly=True, theta_a_forecast=theta_a_forecast)
            total_set = frozenset(range(T))
            if tcl.p(total_set) > tcl.b(total_set):
                continue
            
            P0_i = np.maximum(0, (theta_a_forecast - theta_r) / b_coef)
            tcl_objs_list.append(tcl)
            P0_individual_list.append(P0_i)
        except Exception:
            continue
    
    if len(tcl_objs_list) < num_households:
        raise RuntimeError(f"Only generated {len(tcl_objs_list)}/{num_households} TCLs")
    
    P0_individual = np.array(P0_individual_list)
    P0_agg = np.sum(P0_individual, axis=0)
    
    return {
        'tcl_objs': tcl_objs_list,
        'theta_a_forecast': theta_a_forecast,
        'P0': P0_agg,
        'demands': P0_individual.T,
        'prices': prices,
        'periods': T,
        'households': num_households,
        'dt': 1,
        'objectives': ['cost', 'peak']
    }

# 主程序
print("="*80)
print("双目标对比测试：成本 vs 峰值优化")
print("="*80)

num_households = 15
periods = 24

print(f"\n生成TCL数据: {num_households} 个家庭, {periods} 个时间段")
data = generate_tcl_data(num_households, periods)
print(f"基线功率P0范围: [{data['P0'].min():.1f}, {data['P0'].max():.1f}] kW")

# 基准算法 (无灵活性)
print("\n" + "="*80)
print("计算基准：无灵活性")
print("="*80)
result_noflex = algo_no_flex.algo(data)
cost_noflex = result_noflex['cost_value']
peak_noflex = result_noflex['peak_value']
print(f"基准成本: {cost_noflex:.2f}, 基准峰值: {peak_noflex:.2f} kW")

# 测试矩阵
algorithms = [
    ('G-Poly-Transform-Det', algo_g_polymatroid_transform_det),
    ('JCC-SRO', algo_g_polymatroid_jcc_sro),
    ('JCC-Re-SRO', algo_g_polymatroid_jcc_resro)
]

objectives = ['cost', 'peak']

results = []

for algo_name, algo_module in algorithms:
    for objective in objectives:
        print(f"\n{'='*80}")
        print(f"运行 {algo_name} - 目标: {objective}")
        print(f"{'='*80}")
        
        try:
            if algo_name == 'JCC-SRO' or algo_name == 'JCC-Re-SRO':
                # JCC算法需要不确定性数据
                run_data = data.copy()
                if algo_name == 'JCC-Re-SRO':
                    # 加载Re-SRO数据
                    try:
                        omega_sro_shape = np.load('omega_sro_shape_summer.npy')
                        omega_sro_calib = np.load('omega_sro_calib_summer.npy')
                        omega_resro_calib = np.load('omega_resro_calib_summer.npy')
                        run_data['uncertainty_data'] = {
                            'D_shape': omega_sro_shape,
                            'D_calib': omega_sro_calib,
                            'D_resro_calib': omega_resro_calib,
                            'epsilon': 0.05,
                            'delta': 0.05,
                            'use_full_cov': True
                        }
                    except:
                        print(f"  警告: 无法加载Re-SRO数据,跳过")
                        continue
                else:
                    # 加载SRO数据
                    try:
                        omega_shape_set = np.load('omega_shape_set_summer.npy')
                        omega_calibration_set = np.load('omega_calibration_set_summer.npy')
                        run_data['uncertainty_data'] = {
                            'D_shape': omega_shape_set,
                            'D_calib': omega_calibration_set,
                            'epsilon': 0.05,
                            'delta': 0.05,
                            'use_full_cov': True
                        }
                    except:
                        print(f"  警告: 无法加载SRO数据,跳过")
                        continue
                
                result = algo_module.solve(run_data, objective=objective)
            else:
                result = algo_module.solve(data, objective=objective)
            
            cost = result.get('total_cost', np.nan)
            peak = result.get('peak_power', np.nan)
            comp_time = result.get('computation_time', np.nan)
            
            # 计算UPR (Unmet Performance Ratio)
            if not np.isnan(cost) and not np.isnan(peak):
                cost_gap = cost - cost_noflex
                peak_gap = peak - peak_noflex
                print(f"\n结果:")
                print(f"  成本: {cost:.2f} (相对基准: {cost_gap:+.2f}, {100*cost_gap/cost_noflex:+.1f}%)")
                print(f"  峰值: {peak:.2f} kW (相对基准: {peak_gap:+.2f}, {100*peak_gap/peak_noflex:+.1f}%)")
                print(f"  时间: {comp_time:.2f}s")
                
                results.append({
                    'Algorithm': algo_name,
                    'Objective': objective,
                    'Cost': cost,
                    'Peak': peak,
                    'Cost_Gap': cost_gap,
                    'Peak_Gap': peak_gap,
                    'Cost_Gap_Pct': 100*cost_gap/cost_noflex,
                    'Peak_Gap_Pct': 100*peak_gap/peak_noflex,
                    'Time': comp_time
                })
            else:
                print(f"  ERROR: 算法未返回有效结果")
        
        except Exception as e:
            print(f"  ERROR: {str(e)}")

# 显示结果汇总
if results:
    print("\n" + "="*80)
    print("性能对比汇总")
    print("="*80)
    df = pd.DataFrame(results)
    print("\n成本目标对比:")
    cost_results = df[df['Objective']=='cost'][['Algorithm', 'Cost', 'Cost_Gap_Pct', 'Time']]
    print(cost_results.to_string(index=False))
    
    print("\n峰值目标对比:")
    peak_results = df[df['Objective']=='peak'][['Algorithm', 'Peak', 'Peak_Gap_Pct', 'Time']]
    print(peak_results.to_string(index=False))
    
    # 保存结果
    df.to_csv('dual_objective_comparison.csv', index=False)
    print(f"\n详细结果已保存到 'dual_objective_comparison.csv'")
