# -*- coding: utf-8 -*-
"""
简化版本对比框架：G-Polymatroid算法 vs 经典算法

专注于对比以下算法：
1. G-Polymatroid方法：
   - G-Poly Approximate (algo_g_polymatroid_approximate)

2. 经典方法：
   - Exact Minkowski (algo_exact)
   - No Flexibility (algo_no_flex)
   - Barot Outer (algo_Barot_wo_pc)
   - Inner Homothets (algo_Inner_Homothets)
   - Zonotope (algo_Zonotope)

排除鲁棒算法：algo_g_polymatroid_initial_robust, algo_g_polymatroid_final_robust
"""

import numpy as np
import pandas as pd
import time
import sys
import os
from functools import partial

# 添加路径以访问算法库
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# 导入算法模块
from comparison.lib import algo_exact, algo_no_flex, algo_Barot_wo_pc, algo_Inner_Homothets, algo_Zonotope
from comparison.lib import algo_g_polymatroid_approximate
from comparison.lib import algo_Barot_Inner  # <-- 新增导入
from comparison.lib import algo_Inner_affine # <-- 新增导入
# 导入G-Polymatroid的依赖
from flexitroid.devices.tcl import TCL
# 【核心修正】: 从权威位置导入H表示生成函数
from flexitroid.utils.tcl_utils import get_true_tcl_polytope_H_representation

def check_tcl_feasibility(params, theta_a_forecast):
    """
    检查给定的TCL参数集是否可行，方式与simulation_utils.py一致。
    """
    try:
        tcl_instance = TCL(params, build_g_poly=True, theta_a_forecast=theta_a_forecast)
        T = params['T']
        total_A = frozenset(range(T))
        # 检查最关键的约束：整个时间跨度内的总能量
        if tcl_instance.p(total_A) > tcl_instance.b(total_A):
            return False
        g_poly = getattr(tcl_instance, '_internal_g_poly', None)
        # 只有有x_min/x_max属性时才进一步检查
        if g_poly is not None and hasattr(g_poly, 'x_min') and hasattr(g_poly, 'x_max'):
            y_lower = g_poly.x_min
            y_upper = g_poly.x_max
            if np.any(np.isinf(y_lower)) or np.any(np.isinf(y_upper)) or \
               np.any(np.isnan(y_lower)) or np.any(np.isnan(y_upper)):
                return False
        # 如果没有x_min/x_max属性，认为可行（或可根据实际需求return False）
        return True
    except Exception:
        return False

def generate_tcl_objs_from_params(tcl_param_list, theta_a_forecast):
    """
    用统一参数生成TCL对象（build_g_poly=False），所有算法都用这批对象。
    """
    tcl_objs = []
    for params in tcl_param_list:
        tcl_objs.append(TCL({**params, 'theta_a_forecast': theta_a_forecast}, build_g_poly=False, theta_a_forecast=theta_a_forecast))
    return tcl_objs

# from tools import get_true_tcl_polytope_H_representation # <-- 确保导入

def generate_realistic_tcl_data(num_households=10, periods=24, sample_seed=42):
    """
    统一采样TCL参数，并为所有算法生成一致的、格式正确的数据。
    - 使用更真实的TCL参数以创造有意义的约束。
    - 为每个TCL生成H表示(A, b)，供经典算法使用。
    - 【核心修正】：根据物理模型精确计算delta，保证模型一致性。
    """
    np.random.seed(sample_seed)
    T_HORIZON = periods

    # --- 步骤 1: 定义环境和真实的TCL参数范围 ---
    base_temp = 27.0
    temp_amplitude = 7.0
    time_hours = np.arange(T_HORIZON)
    theta_a_forecast = base_temp + temp_amplitude * np.cos(2 * np.pi * (time_hours - 15) / 24)
    theta_a_forecast = np.clip(theta_a_forecast, 22.0, 35.0)

    prices = np.ones(periods) * 0.15
    peak_hours = list(range(13, 20))
    for hour in peak_hours:
        prices[hour] = 0.30
    
    param_dist = {
        'R_th_range': (2.0, 5.0),
        'C_th_range': (10.0, 20.0),
        'P_m_range': (3.0, 5.0),
        'eta_range': (2.5, 4.0),
        'theta_r_range': (22.0, 24.0),
        'delta_val_range': (0.5, 1.5)
    }
    
    tcl_param_list = []
    tcl_polytope_list = []
    tcl_objs_list = []
    demands = []

    # --- 步骤 2: 生成TCL参数、对象、H表示和基线功率 ---
    attempts = 0
    while len(tcl_param_list) < num_households and attempts < num_households * 100: # 增加尝试次数
        attempts += 1
        
        # 采样物理参数
        C_th = np.random.uniform(*param_dist['C_th_range'])
        R_th = np.random.uniform(*param_dist['R_th_range'])
        P_m = np.random.uniform(*param_dist['P_m_range'])
        eta = np.random.uniform(*param_dist['eta_range'])
        theta_r = np.random.uniform(*param_dist['theta_r_range'])
        delta_val = np.random.uniform(*param_dist['delta_val_range'])
        
        
        b_coef = R_th * eta
        
        # ==================== 【核心修正点】 ====================
        delta = 1
        a = np.exp(-delta / (R_th * C_th))  # delta=1, 单位小时
        # =======================================================
        
        x0 = 0.0

        tcl_params = {
            'C_th': C_th, 'R_th': R_th, 'P_m': P_m, 'eta': eta, 'theta_r': theta_r,
            'delta_val': delta_val, 'delta': delta, 'x0': x0, 'a': a, 'b': b_coef,
            'T': periods, 'theta_a_forecast': theta_a_forecast, 'P_min': 0
        }

        if not check_tcl_feasibility(tcl_params, theta_a_forecast):
            continue

        # 【核心修正】: 使用正确的函数签名和参数调用
        u_min = -np.maximum(0, (theta_a_forecast - theta_r) / b_coef)
        u_max = P_m + u_min # u_max = P_m - P0
        x_plus = (C_th * delta_val) / eta
        
        A_i, b_i = get_true_tcl_polytope_H_representation(
            T=periods, a=a, x0=x0, u_min=u_min, u_max=u_max,
            x_min_phys=-x_plus, x_max_phys=x_plus, delta=delta
        )
        tcl_polytope_list.append({'A': A_i, 'b': b_i})
        
        tcl_obj = TCL(tcl_params, build_g_poly=False, theta_a_forecast=theta_a_forecast)
        tcl_objs_list.append(tcl_obj)
        
        P0_unconstrained = (theta_a_forecast - theta_r) / b_coef
        P0_forecast = np.maximum(0, P0_unconstrained)
        demands.append(P0_forecast)
        
        tcl_param_list.append(tcl_params)

    if len(tcl_param_list) < num_households:
        raise RuntimeError(f"在 {attempts} 次尝试后，未能采样足够数量的可行TCL参数！")

    demands_np = np.array(demands).T

    # --- 步骤 3: 打包成统一的data字典 ---
    return {
        'tcl_params': tcl_param_list,
        'tcl_polytopes': tcl_polytope_list,
        'tcl_objs': tcl_objs_list,
        'prices': prices,
        'demands': demands_np,
        'periods': periods,
        'households': num_households,
        'dt': 1,
        'objectives': ['cost', 'peak'],
        'P0': np.sum(demands_np, axis=1),
        'theta_a_forecast': theta_a_forecast
    }

def run_algorithm(algo_name, algo_module, data):
    """运行单个算法并返回结果"""
    print(f"  运行算法: {algo_name}...")
    
    try:
        start_time = time.time()
        
        # 支持algo_module本身就是函数（如G-Poly Approximate已注册为algo函数）
        if callable(algo_module):
            result = algo_module(data)
        elif hasattr(algo_module, 'run'):
            result = algo_module.run(data)
        elif hasattr(algo_module, 'algo'):
            result = algo_module.algo(data)
        else:
            raise AttributeError(f"Algorithm module {algo_name} has no callable, 'run' or 'algo' function")
        
        total_time = time.time() - start_time
        
        # 标准化结果格式
        if isinstance(result, dict):
            cost_value = result.get('cost_value', result.get('cost', np.nan))
            peak_value = result.get('peak_value', result.get('peak', np.nan))
            cost_time = result.get('cost_time', result.get('opt_time', 0.0))
            peak_time = result.get('peak_time', result.get('opt_time', 0.0))
            algo_time = result.get('algo_time', total_time)
        else:
            cost_value = peak_value = np.nan
            cost_time = peak_time = algo_time = 0.0
        
        print(f"    完成: cost={cost_value:.3f}, peak={peak_value:.3f}")
        
        return {
            'cost_value': cost_value,
            'peak_value': peak_value,
            'cost_time': cost_time,
            'peak_time': peak_time,
            'algo_time': algo_time,
            'total_time': total_time,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"    错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            'cost_value': np.nan,
            'peak_value': np.nan,
            'cost_time': 0.0,
            'peak_time': 0.0,
            'algo_time': 0.0,
            'total_time': 0.0,
            'status': f'error: {e}'
        }

def calculate_upr(approx_value, exact_value, no_flex_value):
    """计算UPR（Under-Performance Ratio）"""
    if np.isnan(approx_value) or np.isnan(exact_value) or np.isnan(no_flex_value):
        return np.nan
    
    denominator = no_flex_value - exact_value
    if abs(denominator) < 1e-6:
        return 0.0
    
    return ((approx_value - exact_value) / denominator) * 100.0

def run_simplified_comparison(num_samples=5, num_households=10, periods=24):
    """
    运行简化的算法对比
    
    Args:
        num_samples: 测试样本数量
        num_households: 每个样本的家庭数量
        periods: 时间周期数
    """
    
    print("=" * 60)
    print("简化版 G-POLYMATROID 性能对比")
    print("=" * 60)
    print(f"测试配置: {num_samples}个样本, {num_households}个家庭, {periods}个时间段")
    print()
    
    # 定义算法 - 增加多个内近似方法
    algorithms = {
        # 基准方法
        'Exact Minkowski': algo_exact,
        'No Flexibility': algo_no_flex,
        
        # 外近似方法
        'Barot Outer': algo_Barot_wo_pc,
        
        # 内近似方法
        'G-Poly Approximate': algo_g_polymatroid_approximate.algo,
        'Barot Inner': algo_Barot_Inner,
        'Inner Homothets': algo_Inner_Homothets,
        'Zonotope': algo_Zonotope,
        'Inner Affine': algo_Inner_affine
    }
    
    g_poly_methods = ['G-Poly Approximate']
    classical_methods = ['Exact Minkowski', 'No Flexibility', 'Barot Outer', 'Barot Inner', 'Inner Homothets', 'Zonotope', 'Inner Affine']
    
    all_results = []
    
    # 运行所有样本
    for sample_idx in range(num_samples):
        print(f"样本 {sample_idx + 1}/{num_samples}")
        
        # 生成测试数据
        data = generate_realistic_tcl_data(num_households, periods)
        data['sample'] = sample_idx
        
        sample_results = {'sample': sample_idx}
        
        # 运行所有算法
        for algo_name, algo_module in algorithms.items():
            result = run_algorithm(algo_name, algo_module, data)
            
            # 存储结果
            for key, value in result.items():
                sample_results[f"{algo_name}_{key}"] = value
        
        all_results.append(sample_results)
        print()
    
    # 转换为DataFrame进行分析
    df = pd.DataFrame(all_results)
    
    # 计算UPR
    print("计算UPR指标...")
    
    for sample_idx in range(num_samples):
        sample_data = df[df['sample'] == sample_idx].iloc[0]
        
        exact_cost = sample_data['Exact Minkowski_cost_value']
        exact_peak = sample_data['Exact Minkowski_peak_value']
        no_flex_cost = sample_data['No Flexibility_cost_value']
        no_flex_peak = sample_data['No Flexibility_peak_value']
        
        for algo_name in algorithms.keys():
            if algo_name in ['Exact Minkowski', 'No Flexibility']:
                continue
                
            # 计算成本UPR
            approx_cost = sample_data[f'{algo_name}_cost_value']
            cost_upr = calculate_upr(approx_cost, exact_cost, no_flex_cost)
            df.loc[df['sample'] == sample_idx, f'{algo_name}_cost_upr'] = cost_upr
            
            # 计算峰值UPR
            approx_peak = sample_data[f'{algo_name}_peak_value']
            peak_upr = calculate_upr(approx_peak, exact_peak, no_flex_peak)
            df.loc[df['sample'] == sample_idx, f'{algo_name}_peak_upr'] = peak_upr
    
    # 计算统计结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    
    # 算法数量统计
    print(f"算法数量:")
    print(f"  - G-Polymatroid方法: {len(g_poly_methods)}")
    print(f"  - 经典方法: {len(classical_methods)}")
    print()
    
    # UPR统计
    print("平均UPR对比 (%):")
    print(f"{'算法':<25} {'成本UPR':<12} {'峰值UPR':<12}")
    print("-" * 50)
    
    upr_summary = []
    
    for algo_name in algorithms.keys():
        if algo_name in ['Exact Minkowski', 'No Flexibility']:
            continue
            
        cost_upr_col = f'{algo_name}_cost_upr'
        peak_upr_col = f'{algo_name}_peak_upr'
        
        if cost_upr_col in df.columns and peak_upr_col in df.columns:
            avg_cost_upr = df[cost_upr_col].mean()
            avg_peak_upr = df[peak_upr_col].mean()
            
            print(f"{algo_name:<25} {avg_cost_upr:<12.2f} {avg_peak_upr:<12.2f}")
            
            upr_summary.append({
                'Algorithm': algo_name,
                'Cost_UPR': avg_cost_upr,
                'Peak_UPR': avg_peak_upr,
                'Type': 'G-Polymatroid' if algo_name in g_poly_methods else 'Classical'
            })
    
    print()
    
    # 找出最佳性能
    upr_df = pd.DataFrame(upr_summary)
    if not upr_df.empty and not upr_df['Cost_UPR'].isna().all():
        try:
            best_cost_algo = upr_df.loc[upr_df['Cost_UPR'].idxmin(), 'Algorithm']
            best_peak_algo = upr_df.loc[upr_df['Peak_UPR'].idxmin(), 'Algorithm']
            
            print("最佳性能:")
            print(f"  - 成本优化最佳: {best_cost_algo}")
            print(f"  - 削峰最佳: {best_peak_algo}")
        except (KeyError, ValueError):
            print("最佳性能: 无法确定（所有算法都失败）")
        print()
    else:
        print("最佳性能: 无法确定（没有有效结果）")
        print()
    
    # G-Polymatroid内部对比
    if not upr_df.empty and 'Type' in upr_df.columns:
        g_poly_data = upr_df[upr_df['Type'] == 'G-Polymatroid']
        if not g_poly_data.empty:
            print("G-Polymatroid方法详细结果:")
            for _, row in g_poly_data.iterrows():
                print(f"  {row['Algorithm']}: 成本={row['Cost_UPR']:.2f}%, 峰值={row['Peak_UPR']:.2f}%")
        else:
            print("G-Polymatroid方法详细结果: 暂无")
        print()
    else:
        print("G-Polymatroid方法详细结果: 暂无")
        print()
    
    # 计算时间统计
    print("平均计算时间 (秒):")
    for algo_name in algorithms.keys():
        time_col = f'{algo_name}_total_time'
        if time_col in df.columns:
            avg_time = df[time_col].mean()
            print(f"  {algo_name}: {avg_time:.3f}")
    
    print()
    print("=" * 60)
    
    # 保存详细结果
    output_file = "comparison_results/simplified_comparison_results.csv"
    os.makedirs("comparison_results", exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"详细结果已保存到: {output_file}")
    
    # 保存汇总报告
    report_file = "comparison_results/simplified_comparison_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("简化版 G-POLYMATROID 性能对比报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"算法数量:\n")
        f.write(f"  - G-Polymatroid方法: {len(g_poly_methods)}\n")
        f.write(f"  - 经典方法: {len(classical_methods)}\n\n")
        
        f.write("平均UPR对比 (%):\n")
        f.write(f"{'算法':<25} {'成本UPR':<12} {'峰值UPR':<12}\n")
        f.write("-" * 50 + "\n")
        
        for _, row in upr_df.iterrows():
            f.write(f"{row['Algorithm']:<25} {row['Cost_UPR']:<12.2f} {row['Peak_UPR']:<12.2f}\n")
        
        f.write(f"\n最佳性能:\n")
        if not upr_df.empty and not upr_df['Cost_UPR'].isna().all():
            try:
                best_cost_algo = upr_df.loc[upr_df['Cost_UPR'].idxmin(), 'Algorithm']
                best_peak_algo = upr_df.loc[upr_df['Peak_UPR'].idxmin(), 'Algorithm']
                f.write(f"  - 成本优化最佳: {best_cost_algo}\n")
                f.write(f"  - 削峰最佳: {best_peak_algo}\n")
            except (KeyError, ValueError):
                f.write(f"  - 无法确定（所有算法都失败）\n")
        else:
            f.write(f"  - 无法确定（没有有效结果）\n")
        
        f.write(f"\nG-Polymatroid方法详细结果:\n")
        if not upr_df.empty and 'Type' in upr_df.columns:
            g_poly_data = upr_df[upr_df['Type'] == 'G-Polymatroid']
            if not g_poly_data.empty:
                for _, row in g_poly_data.iterrows():
                    f.write(f"  {row['Algorithm']}: 成本={row['Cost_UPR']:.2f}%, 峰值={row['Peak_UPR']:.2f}%\n")
            else:
                f.write(f"  暂无\n")
        else:
            f.write(f"  暂无\n")
        
        f.write(f"\n平均计算时间 (秒):\n")
        for algo_name in algorithms.keys():
            time_col = f'{algo_name}_total_time'
            if time_col in df.columns:
                avg_time = df[time_col].mean()
                f.write(f"  {algo_name}: {avg_time:.3f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"汇总报告已保存到: {report_file}")
    
    return df, upr_df

if __name__ == "__main__":
    # 运行简化对比
    results_df, upr_df = run_simplified_comparison(
        num_samples=1,      # 测试5个样本
        num_households=10,  # 每个样本10个家庭
        periods=24          # 24个时间段 
    ) 