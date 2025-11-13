# -*- coding: utf-8 -*-
"""
高级版对比框架：确定性算法 vs 鲁棒G-Polymatroid算法
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# --- 路径设置 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if project_root not in sys.path: sys.path.insert(0, project_root)
if lib_path not in sys.path: sys.path.insert(0, lib_path)

# --- 算法模块导入 ---
from comparison.lib import (
    algo_exact, algo_no_flex, algo_Barot_wo_pc, algo_Inner_Homothets, 
    algo_Zonotope, algo_Barot_Inner, algo_Inner_affine
)
from comparison.lib import (
    algo_g_polymatroid_transform_det,  # 确定性坐标变换 (新框架)
    algo_g_polymatroid_jcc_sro,        # JCC-SRO算法 (坐标变换)
    algo_g_polymatroid_jcc_resro       # JCC-Re-SRO算法 (坐标变换)
)

# --- 依赖项导入 ---
from flexitroid.devices.tcl import TCL
from flexitroid.utils.simulation_utils import (
    generate_ground_truth_data_summer, 
    generate_temperature_uncertainty_data  # 只使用温度误差
)

# --- 固定TCL参数分布 (确定性,无参数不确定性) ---
FIXED_PARAM_DIST = {
    'R_th_range': (2.0, 3.0), 
    'C_th_range': (5.0, 15.0), 
    'P_m_range': (10.0, 20.0),
    'eta_range': (2.5, 3.5), 
    'theta_r_range': (22.0, 23.0), 
    'delta_val_range': (1.0, 2.0)
}

# --- 数据生成函数 (清理版) ---
def generate_realistic_tcl_data(num_households=10, periods=24, sample_seed=42):
    """
    生成TCL群体数据 (仅考虑温度预测不确定性)
    
    清理说明:
    - 移除参数不确定性相关代码
    - TCL参数从固定分布采样,但对所有算法保持一致
    - 只有温度预测存在不确定性(通过JCC算法处理)
    - 基线功率P0基于预测温度theta_a_forecast计算
    
    Args:
        num_households: TCL数量
        periods: 时间步数
        sample_seed: 随机种子(用于参数采样,保证可重复性)
    
    Returns:
        包含TCL对象、基线功率、电价等信息的字典
    """
    np.random.seed(sample_seed)
    T = periods
    
    # 1. 生成预测温度序列 (名义/确定性预测)
    base_temp = 27.0
    temp_amplitude = 7.0
    time_hours = np.arange(T)
    theta_a_forecast = base_temp + temp_amplitude * np.cos(2 * np.pi * (time_hours - 15) / 24)
    theta_a_forecast = np.clip(theta_a_forecast, 20.0, 35.0)
    
    # 2. 生成分时电价
    prices = np.ones(T) * 0.10      # 基础价
    prices[0:8] = 0.05              # 0-7点 低谷
    prices[8:13] = 0.15             # 8-12点 平段
    prices[13:21] = 0.60            # 13-20点 峰段
    prices[16:19] = 0.80            # 16-18点 超峰
    prices[21:24] = 0.12            # 21-23点 平段
    
    # 3. 生成TCL群体 (参数固定,无不确定性)
    tcl_objs_list = []
    tcl_polytope_list = []
    P0_individual_list = []  # 每个TCL的基线功率
    
    attempts = 0
    max_attempts = num_households * 100
    
    while len(tcl_objs_list) < num_households and attempts < max_attempts:
        attempts += 1
        
        # 从固定分布采样参数
        C_th = np.random.uniform(*FIXED_PARAM_DIST['C_th_range'])
        R_th = np.random.uniform(*FIXED_PARAM_DIST['R_th_range'])
        P_m = np.random.uniform(*FIXED_PARAM_DIST['P_m_range'])
        eta = np.random.uniform(*FIXED_PARAM_DIST['eta_range'])
        theta_r = np.random.uniform(*FIXED_PARAM_DIST['theta_r_range'])
        delta_val = np.random.uniform(*FIXED_PARAM_DIST['delta_val_range'])
        
        # 计算派生参数
        a = 1 - 1/(R_th * C_th)
        b_coef = R_th * eta
        delta = 1.0
        x0 = 0.0
        
        # 构建TCL参数字典
        tcl_params = {
            'T': T,
            'C_th': C_th, 
            'R_th': R_th, 
            'P_m': P_m, 
            'eta': eta, 
            'theta_r': theta_r,
            'delta_val': delta_val, 
            'delta': delta, 
            'x0': x0, 
            'a': a, 
            'b': b_coef,
            'theta_a_forecast': theta_a_forecast,
            'P_min': 0  # 仅制冷
        }
        
        # 创建TCL对象并验证可行性
        try:
            tcl = TCL(tcl_params, build_g_poly=True, theta_a_forecast=theta_a_forecast)
            
            # 验证g-polymatroid可行性: p(T) <= b(T)
            total_set = frozenset(range(T))
            if tcl.p(total_set) > tcl.b(total_set):
                continue  # 不可行,重新采样
            
            # 计算基线功率 (基于预测温度)
            P0_i = np.maximum(0, (theta_a_forecast - theta_r) / b_coef)
            
            # 构建polytope表示 (传统算法需要)
            if tcl.A_phys is not None and tcl.b_phys_nom is not None:
                try:
                    A_u, b_u = tcl.get_u_only_polytope()
                    tcl_polytope_list.append({'A': A_u, 'b': b_u})
                except Exception as e:
                    print(f"警告: TCL {len(tcl_objs_list)} polytope构建失败: {e}")
                    tcl_polytope_list.append({'A': np.array([]), 'b': np.array([])})
            else:
                tcl_polytope_list.append({'A': np.array([]), 'b': np.array([])})
            
            # 保存成功的TCL
            tcl_objs_list.append(tcl)
            P0_individual_list.append(P0_i)
            
        except Exception as e:
            # TCL创建失败,重新采样
            continue
    
    # 检查是否生成了足够的TCL
    if len(tcl_objs_list) < num_households:
        raise RuntimeError(f"在{attempts}次尝试后,只生成了{len(tcl_objs_list)}/{num_households}个可行TCL")
    
    # 4. 计算聚合基线功率 (基于预测温度)
    P0_individual = np.array(P0_individual_list)  # (N, T)
    P0_agg = np.sum(P0_individual, axis=0)         # (T,)
    
    # 5. 返回数据字典
    return {
        'tcl_objs': tcl_objs_list,              # TCL对象列表
        'tcl_polytopes': tcl_polytope_list,      # Polytope表示(传统算法)
        'theta_a_forecast': theta_a_forecast,    # 预测温度 (名义值)
        'P0': P0_agg,                            # 聚合基线功率 (基于预测温度)
        'demands': P0_individual.T,              # 个体基线功率 (T, N) 格式
        'prices': prices,                        # 分时电价
        'periods': T,                            # 时间步数
        'households': num_households,            # TCL数量
        'dt': 1,                                 # 时间步长
        'objectives': ['cost', 'peak']           # 优化目标
    }

# --- 算法列表 ---
# 所有可用的算法
ALL_ALGORITHMS = {
    'Exact Minkowski': algo_exact,
    'No Flexibility': algo_no_flex,
    'Barot Outer': algo_Barot_wo_pc,
    'Inner Homothets': algo_Inner_Homothets,
    'Zonotope': algo_Zonotope,
    'Barot Inner': algo_Barot_Inner,
    'Inner Affine': algo_Inner_affine,
    'G-Poly-Transform-Det': algo_g_polymatroid_transform_det,  # 确定性坐标变换 (新框架)
    'JCC-SRO': algo_g_polymatroid_jcc_sro,                     # JCC-SRO (单阶段,坐标变换)
    'JCC-Re-SRO': algo_g_polymatroid_jcc_resro                 # JCC-Re-SRO (两阶段,坐标变换)
}

# 算法启用配置 - 设置为True以启用该算法
ALGORITHM_ENABLE = {
    'Exact Minkowski': True,      # 精确Minkowski和 (基准)
    'No Flexibility': True,       # 无灵活性 (基准)
    'Barot Outer': False,          # 传统算法
    'Inner Homothets': False,      # 传统算法
    'Zonotope': False,             # 传统算法
    'Barot Inner': False,          # 传统算法
    'Inner Affine': False,         # 传统算法
    'G-Poly-Transform-Det': True, # 【新】确定性g-polymatroid (坐标变换)
    'JCC-SRO': False,              # 【新】JCC-SRO (鲁棒优化)
    'JCC-Re-SRO': False            # 【新】JCC-Re-SRO (两阶段鲁棒优化)
}

# 需要不确定性数据的算法
JCC_ALGOS = {'JCC-SRO', 'JCC-Re-SRO'}

# 使用solve()接口且支持指定优化目标的算法
GPOLY_ALGOS = {'G-Poly-Transform-Det', 'JCC-SRO', 'JCC-Re-SRO'}

def get_enabled_algorithms():
    """返回启用的算法字典"""
    return {name: module for name, module in ALL_ALGORITHMS.items() 
            if ALGORITHM_ENABLE.get(name, False)}

def set_algorithm_enable(algo_names, enable=True):
    """
    批量设置算法启用状态
    
    Args:
        algo_names: 算法名称列表,或'all'表示所有算法,或'none'表示禁用所有
        enable: True启用, False禁用
    """
    if algo_names == 'all':
        for name in ALL_ALGORITHMS.keys():
            ALGORITHM_ENABLE[name] = enable
    elif algo_names == 'none':
        for name in ALL_ALGORITHMS.keys():
            ALGORITHM_ENABLE[name] = False
    else:
        for name in algo_names:
            if name in ALL_ALGORITHMS:
                ALGORITHM_ENABLE[name] = enable
            else:
                print(f"警告: 未知算法 '{name}'")

def enable_only(*algo_names):
    """只启用指定的算法,禁用其他所有算法"""
    set_algorithm_enable('none', False)
    set_algorithm_enable(algo_names, True)

def print_algorithm_status():
    """打印所有算法的启用状态"""
    print("\n算法启用状态:")
    print("-" * 50)
    for name in ALL_ALGORITHMS.keys():
        status = "✓ 启用" if ALGORITHM_ENABLE.get(name, False) else "✗ 禁用"
        print(f"  {name:<30} {status}")
    print("-" * 50)

# --- 数据生成 ---
def run_algorithm(algo_name, algo_module, data, objective=None):
    """
    运行单个算法
    
    Args:
        algo_name: 算法名称
        algo_module: 算法模块
        data: 输入数据
        objective: 优化目标 ('cost' 或 'peak')，仅对G-Poly算法有效
    
    Returns:
        结果字典
    """
    obj_suffix = f" ({objective})" if objective else ""
    print(f"  运行算法: {algo_name}{obj_suffix}...")
    try:
        start_time = time.time()
        
        # G-Polymatroid算法需要特殊处理,调用solve()并指定优化目标
        if algo_name in GPOLY_ALGOS:
            if objective is None:
                raise ValueError(f"{algo_name} 必须指定 objective 参数")
            
            result = algo_module.solve(data, objective=objective)
            total_time = time.time() - start_time
            cost = result.get('total_cost', np.nan)
            peak = result.get('peak_power', np.nan)
            comp_time = result.get('computation_time', total_time)
            
            print(f"    完成: cost={cost:.3f}, peak={peak:.3f} (t={comp_time:.3f}s)")
            return {
                'cost_value': cost, 'peak_value': peak,
                'cost_time': comp_time, 'peak_time': comp_time,
                'total_time': comp_time,
                'algorithm': algo_name, 'objective': objective, 'result': result
            }
        else:
            # 传统算法使用algo()接口（同时返回成本和峰值）
            result = algo_module.algo(data)
            total_time = time.time() - start_time
            cost = result.get('cost_value', np.nan)
            peak = result.get('peak_value', np.nan)
            cost_time = result.get('cost_time', 0.0)
            peak_time = result.get('peak_time', 0.0)
            print(f"    完成: cost={cost:.3f} (t={cost_time:.3f}s), peak={peak:.3f} (t={peak_time:.3f}s)")
            return {
                'cost_value': cost, 'peak_value': peak,
                'cost_time': cost_time, 'peak_time': peak_time,
                'algo_time': result.get('algo_time', total_time), 'total_time': total_time,
                'status': 'success'
            }
    except Exception as e:
        print(f"    错误: {e}")
        import traceback
        traceback.print_exc()
        return {'cost_value': np.nan, 'peak_value': np.nan, 'status': f'error: {e}', 'total_time': 0.0}
    
# --- UPR计算和结果保存 ---
def calculate_and_save_upr(df, algorithms, output_dir):
    """
    计算UPR并保存结果
    
    关键逻辑：
    - 成本UPR：使用以"成本"为目标优化的结果
    - 峰值UPR：使用以"峰值"为目标优化的结果
    - 这样可以公平评估每个算法在各自目标上的表现
    """
    print("\n" + "="*60)
    print("计算UPR (Unmet Performance Ratio)")
    print("="*60)
    
    for idx, row in df.iterrows():
        # 基准值：Exact Minkowski 和 No Flexibility
        exact_cost = row['Exact Minkowski_cost_value']
        exact_peak = row['Exact Minkowski_peak_value']
        noflex_cost = row['No Flexibility_cost_value']
        noflex_peak = row['No Flexibility_peak_value']
        
        # 计算每个算法的UPR
        for algo in algorithms:
            if algo in ['Exact Minkowski', 'No Flexibility']: 
                continue
            
            # 成本UPR：使用以成本为目标的优化结果
            algo_cost = row[f'{algo}_cost_value']
            if abs(noflex_cost - exact_cost) > 1e-6:
                cost_upr = ((algo_cost - exact_cost) / (noflex_cost - exact_cost) * 100.0)
            else:
                cost_upr = 0.0
            df.loc[idx, f'{algo}_cost_upr'] = cost_upr
            
            # 峰值UPR：使用以峰值为目标的优化结果
            algo_peak = row[f'{algo}_peak_value']
            if abs(noflex_peak - exact_peak) > 1e-6:
                peak_upr = ((algo_peak - exact_peak) / (noflex_peak - exact_peak) * 100.0)
            else:
                peak_upr = 0.0
            df.loc[idx, f'{algo}_peak_upr'] = peak_upr
    
    # 计算平均UPR并显示
    print("\n结果汇总:")
    print(f"{'算法':<25} {'成本UPR (%)':<15} {'峰值UPR (%)':<15} {'总时间 (s)':<15}")
    print("-" * 70)
    
    summary = []
    for name in algorithms:
        if name in ['Exact Minkowski', 'No Flexibility']: 
            continue
        
        avg_cost_upr = df[f'{name}_cost_upr'].mean()
        avg_peak_upr = df[f'{name}_peak_upr'].mean()
        avg_time = df[f'{name}_total_time'].mean()
        
        print(f"{name:<25} {avg_cost_upr:<15.2f} {avg_peak_upr:<15.2f} {avg_time:<15.3f}")
        summary.append({
            'Algorithm': name, 
            'Cost_UPR': avg_cost_upr, 
            'Peak_UPR': avg_peak_upr, 
            'Total_Time': avg_time
        })
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "advanced_comparison_results.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(output_dir, "advanced_summary.csv"), index=False)
    
    print(f"\n详细结果和摘要已保存到 '{output_dir}' 目录。")
    print("="*60)

# --- 主流程 ---
def run_advanced_comparison(num_samples, num_households, periods, num_days, num_tcls=None, t_horizon=None):
    print("=" * 60)
    print("高级版 G-POLYMATROID 性能对比框架")
    print("=" * 60)
    print(f"测试配置: {num_samples}个样本, {num_households}个家庭, {periods}个时间段, {num_days}天")
    
    if num_tcls is None:
        num_tcls = num_households
    if t_horizon is None:
        t_horizon = periods
    
    # 1. 生成TCL群体数据 (参数固定,仅温度不确定性)
    print("\n--- 阶段1: 生成TCL群体 ---")
    data = generate_realistic_tcl_data(num_households, periods, sample_seed=42)
    print(f"成功生成 {num_households} 个TCL")
    print(f"预测温度范围: [{data['theta_a_forecast'].min():.1f}, {data['theta_a_forecast'].max():.1f}]°C")
    print(f"聚合基线功率P0范围: [{data['P0'].min():.1f}, {data['P0'].max():.1f}] kW")
    
    # 2. 为JCC算法准备温度误差数据
    print("\n--- 准备不确定性数据 (JCC-SRO/Re-SRO) ---")
    
    # 检查是否需要生成温度误差数据
    file_suffix = "_summer"
    shape_file = f'omega_shape_set{file_suffix}.npy'
    calib_file = f'omega_calibration_set{file_suffix}.npy'
    
    if not all(os.path.exists(f) for f in [shape_file, calib_file]):
        print("需要生成温度误差数据...")
        
        # 生成夏季温度数据
        if not os.path.exists('summer_temps_by_day.npy'):
            from flexitroid.utils.extract_summer_high_temp_dataset import main as extract
            print("生成夏季温度原始数据...")
            extract()
        
        # 生成地面真实数据
        generate_ground_truth_data_summer(
            num_tcls=num_tcls, 
            t_horizon=t_horizon, 
            num_days=num_days, 
            use_summer_data=True
        )
        
        # 加载温度误差数据
        omega_combined = np.load('summer_all_errors_yesterday.npy')
        
        # 生成数据集 (针对SRO和Re-SRO分别准备)
        # SRO: 1/2 + 1/2 分割
        omega_shape_set, omega_calibration_set = generate_temperature_uncertainty_data(
            omega_combined=omega_combined,
            use_summer_data=True,
            for_resro=False  # SRO模式
        )
        
        # Re-SRO: 1/4 + 1/4 + 1/2 分割 (独立数据)
        omega_sro_shape, omega_sro_calib, omega_resro_calib = generate_temperature_uncertainty_data(
            omega_combined=omega_combined,
            use_summer_data=True,
            for_resro=True  # Re-SRO模式
        )
    else:
        print("温度误差数据集已存在,直接加载")
        omega_shape_set = np.load(shape_file)
        omega_calibration_set = np.load(calib_file)
        
        # 加载Re-SRO的3部分数据
        omega_sro_shape = np.load('omega_sro_shape_summer.npy')
        omega_sro_calib = np.load('omega_sro_calib_summer.npy')
        omega_resro_calib = np.load('omega_resro_calib_summer.npy')
    
    print(f"SRO数据: shape={omega_shape_set.shape}, calib={omega_calibration_set.shape}")
    print(f"Re-SRO数据: sro_shape={omega_sro_shape.shape}, sro_calib={omega_sro_calib.shape}, resro_calib={omega_resro_calib.shape}")
    
    # 3. 获取启用的算法
    ALGORITHMS = get_enabled_algorithms()
    print(f"\n启用的算法 ({len(ALGORITHMS)}个): {', '.join(ALGORITHMS.keys())}")
    
    # 4. 运行所有启用的算法
    all_results = []
    for i in range(num_samples):
        print(f"\n样本 {i + 1}/{num_samples}")
        sample_results = {'sample': i}
        
        for name, module in ALGORITHMS.items():
            run_data = data.copy()
            
            # 准备不确定性数据（JCC算法需要）
            if name in JCC_ALGOS:
                if name == 'JCC-Re-SRO':
                    run_data['uncertainty_data'] = {
                        'D_shape': omega_sro_shape,
                        'D_calib': omega_sro_calib,
                        'D_resro_calib': omega_resro_calib,
                        'epsilon': 0.05,
                        'delta': 0.05,
                        'use_full_cov': True
                    }
                    print(f"  [{name}] 使用独立Re-SRO数据: SRO({omega_sro_shape.shape[0]}+{omega_sro_calib.shape[0]}) + Re-SRO({omega_resro_calib.shape[0]})")
                else:
                    run_data['uncertainty_data'] = {
                        'D_shape': omega_shape_set,
                        'D_calib': omega_calibration_set,
                        'epsilon': 0.05,
                        'delta': 0.05,
                        'use_full_cov': True
                    }
                    print(f"  [{name}] 使用SRO数据: shape({omega_shape_set.shape[0]}) + calib({omega_calibration_set.shape[0]})")
            
            # 对于G-Poly算法，需要分别运行成本和峰值优化
            if name in GPOLY_ALGOS:
                # 运行成本优化
                print(f"\n  --- {name}: 成本优化 ---")
                result_cost = run_algorithm(name, module, run_data, objective='cost')
                
                # 运行峰值优化
                print(f"\n  --- {name}: 峰值优化 ---")
                result_peak = run_algorithm(name, module, run_data, objective='peak')
                
                # 合并结果：成本值来自成本优化，峰值来自峰值优化
                sample_results[f"{name}_cost_value"] = result_cost['cost_value']
                sample_results[f"{name}_cost_time"] = result_cost['cost_time']
                sample_results[f"{name}_peak_value"] = result_peak['peak_value']
                sample_results[f"{name}_peak_time"] = result_peak['peak_time']
                sample_results[f"{name}_total_time"] = result_cost['total_time'] + result_peak['total_time']
                sample_results[f"{name}_status"] = 'success'
                
            else:
                # 传统算法：一次运行同时得到成本和峰值
                result = run_algorithm(name, module, run_data)
                for key, value in result.items():
                    sample_results[f"{name}_{key}"] = value
        
        all_results.append(sample_results)
    
    # 5. 计算并显示结果
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("原始结果预览")
    print("="*60)
    print("\n各算法的成本和峰值结果:")
    for name in ALGORITHMS.keys():
        if f'{name}_cost_value' in df.columns and f'{name}_peak_value' in df.columns:
            avg_cost = df[f'{name}_cost_value'].mean()
            avg_peak = df[f'{name}_peak_value'].mean()
            avg_time = df[f'{name}_total_time'].mean()
            print(f"  {name:<25} Cost={avg_cost:<10.2f} Peak={avg_peak:<10.2f} Time={avg_time:<8.3f}s")
    
    calculate_and_save_upr(df, list(ALGORITHMS.keys()), "comparison_results")


if __name__ == "__main__":
    # ===================================================================
    # 算法选择配置
    # ===================================================================
    # 方法1: 直接修改ALGORITHM_ENABLE字典 (在文件顶部)
    # 方法2: 使用辅助函数动态配置
    
    # 示例: 只测试g-polymatroid算法
    # enable_only('Exact Minkowski', 'No Flexibility', 'G-Poly-Transform-Det', 'JCC-SRO', 'JCC-Re-SRO')
    
    # 示例: 禁用某些算法
    # set_algorithm_enable(['Barot Outer', 'Barot Inner'], enable=False)
    
    # 示例: 只测试基准和新算法
    # enable_only('Exact Minkowski', 'No Flexibility', 'G-Poly-Transform-Det')
    
    # 打印当前算法状态
    print_algorithm_status()
    
    # ===================================================================
    # 测试参数配置
    # ===================================================================
    num_samples = 1
    num_households = 20
    periods = 24
    num_days = 1000
    num_tcls = num_households
    t_horizon = periods
    
    run_advanced_comparison(
        num_samples=num_samples, 
        num_households=num_households, 
        periods=periods, 
        num_days=num_days, 
        num_tcls=num_tcls, 
        t_horizon=t_horizon
    )
