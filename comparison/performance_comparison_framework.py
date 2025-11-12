#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Comparison Framework for G-Polymatroid Methods

This framework implements systematic performance comparison between g-polymatroid
methods and existing aggregation approaches, focusing on operational benefits
through cost minimization and peak power reduction objectives.

Created for the g-polymatroid aggregation paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import comparison algorithms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'flexitroid')))
try:
    import lib.algo_exact as exact_algo
    import lib.algo_no_flex as no_flex_algo
    import lib.algo_Zonotope as zonotope_algo
    import lib.algo_Inner_Homothets as homothet_algo
    import lib.algo_Barot_wo_pc as barot_algo
    import lib.algo_g_polymatroid_approximate as gpoly_approx
    import lib.algo_g_polymatroid_initial_robust as gpoly_initial
    import lib.algo_g_polymatroid_final_robust as gpoly_final
    print("✓ 成功导入所有对比算法")
except ImportError as e:
    print(f"警告：导入算法模块失败: {e}")
    import traceback
    traceback.print_exc()

class OperationalBenefitComparator:
    """
    运营效益对比器：实现成本最小化和削峰填谷的系统性对比评估
    
    Key Features:
    1. Cost Minimization: f(x) = c^T(x+q)
    2. Peak Power Minimization: f(x) = ||x+q||_∞  
    3. UPR (Unused Potential Ratio) calculation
    4. Statistical significance testing
    """
    
    def __init__(self, config: Dict):
        """
        初始化对比器
        
        Args:
            config: 配置字典，包含算法列表、测试参数等
        """
        self.config = config
        self.algorithms = {
            # 基准算法
            'No Flexibility': no_flex_algo.algo,
            'Exact Minkowski': exact_algo.algo,
            
            # 经典近似方法
            'Zonotope': zonotope_algo.algo,
            'Inner Homothets': homothet_algo.algo,
            'Barot Outer': barot_algo.algo,
            
            # 我们的g-polymatroid方法
            'G-Poly Approximate': gpoly_approx.algo,
            'G-Poly Initial Robust': gpoly_initial.algo,
            'G-Poly Final Robust': gpoly_final.algo,
        }
        
        self.results = {}
        self.upr_results = {}
        
    def calculate_upr(self, z_exact: float, z_approx: float, z_no_flex: float) -> float:
        """
        计算未利用潜力比率 (Unused Potential Ratio)
        
        UPR = (z_approx - z_exact) / (z_no_flex - z_exact) × 100%
        
        Args:
            z_exact: 精确方法的目标值
            z_approx: 近似方法的目标值  
            z_no_flex: 无灵活性时的目标值
            
        Returns:
            UPR百分比
        """
        if abs(z_no_flex - z_exact) < 1e-6:
            return 0.0  # 避免除零
        
        upr = (z_approx - z_exact) / (z_no_flex - z_exact) * 100
        return max(0.0, upr)  # UPR不应为负
    
    def run_single_comparison(self, data: Dict, algorithms: List[str] = None) -> Dict:
        """
        运行单次对比实验
        
        Args:
            data: 测试数据字典
            algorithms: 要对比的算法列表，None表示使用全部
            
        Returns:
            结果字典
        """
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        results = {}
        
        print(f"运行对比实验: T={data['periods']}, H={data['households']}")
        
        # 运行所有算法
        for algo_name in algorithms:
            if algo_name not in self.algorithms:
                continue
                
            print(f"  正在运行: {algo_name}")
            
            try:
                t0 = time.time()
                # Pass the method name to the g-polymatroid algo dispatcher
                if algo_name.startswith('G-Poly'):
                    method = 'approximate'
                    if 'Initial' in algo_name:
                        method = 'initial_robust'
                    elif 'Final' in algo_name:
                        method = 'final_robust'
                    algo_result = self.algorithms[algo_name](data.copy(), method=method)
                else:
                    algo_result = self.algorithms[algo_name](data.copy())
                runtime = time.time() - t0
                
                results[algo_name] = {
                    'cost_value': algo_result.get('cost_value', np.nan),
                    'cost_time': algo_result.get('cost_time', np.nan),
                    'peak_value': algo_result.get('peak_value', np.nan), 
                    'peak_time': algo_result.get('peak_time', np.nan),
                    'algo_time': algo_result.get('algo_time', runtime),
                    'status': 'success'
                }
                
            except Exception as e:
                print(f"    错误: {e}")
                results[algo_name] = {
                    'cost_value': np.nan,
                    'cost_time': np.nan,
                    'peak_value': np.nan,
                    'peak_time': np.nan,
                    'algo_time': np.nan,
                    'status': f'error: {str(e)[:50]}'
                }
        
        return results
    
    def calculate_upr_metrics(self, results: Dict) -> Dict:
        """
        计算所有算法的UPR指标
        
        Args:
            results: 单次实验的结果字典
            
        Returns:
            UPR指标字典
        """
        upr_metrics = {}
        
        # 获取基准值
        z_exact_cost = results.get('Exact Minkowski', {}).get('cost_value', np.nan)
        z_exact_peak = results.get('Exact Minkowski', {}).get('peak_value', np.nan)
        z_no_flex_cost = results.get('No Flexibility', {}).get('cost_value', np.nan)
        z_no_flex_peak = results.get('No Flexibility', {}).get('peak_value', np.nan)
        
        # 计算每个算法的UPR
        for algo_name, algo_results in results.items():
            if algo_name in ['Exact Minkowski', 'No Flexibility']:
                continue
                
            cost_upr = self.calculate_upr(
                z_exact_cost, 
                algo_results.get('cost_value', np.nan),
                z_no_flex_cost
            )
            
            peak_upr = self.calculate_upr(
                z_exact_peak,
                algo_results.get('peak_value', np.nan), 
                z_no_flex_peak
            )
            
            upr_metrics[algo_name] = {
                'cost_upr': cost_upr,
                'peak_upr': peak_upr
            }
        
        return upr_metrics
    
    def run_systematic_comparison(self, 
                                periods_list: List[int] = [4, 8, 12, 16, 20, 24],
                                households_list: List[int] = [5, 10, 15, 20, 25, 30],
                                num_samples: int = 5) -> pd.DataFrame:
        """
        运行系统性对比实验
        
        Args:
            periods_list: 时间段列表
            households_list: 设备数量列表  
            num_samples: 每个配置的样本数
            
        Returns:
            结果DataFrame
        """
        results_list = []
        total_experiments = len(periods_list) * len(households_list) * num_samples
        experiment_count = 0
        
        print(f"开始系统性对比实验，总共{total_experiments}个实验")
        
        for periods in periods_list:
            for households in households_list:
                for sample in range(num_samples):
                    experiment_count += 1
                    progress = experiment_count / total_experiments * 100
                    print(f"\n进度: {progress:.1f}% - T={periods}, H={households}, Sample={sample}")
                    
                    try:
                        # 生成TCL的测试数据
                        data = self.generate_tcl_test_data(periods, households, sample)
                        
                        # 运行对比
                        results = self.run_single_comparison(data)
                        
                        # 计算UPR
                        upr_metrics = self.calculate_upr_metrics(results)
                        
                        # 保存结果
                        for algo_name, algo_results in results.items():
                            result_row = {
                                'periods': periods,
                                'households': households,
                                'sample': sample,
                                'algorithm': algo_name,
                                'cost_value': algo_results['cost_value'],
                                'cost_time': algo_results['cost_time'],
                                'peak_value': algo_results['peak_value'],
                                'peak_time': algo_results['peak_time'],
                                'algo_time': algo_results['algo_time'],
                                'status': algo_results['status'],
                                'cost_upr': upr_metrics.get(algo_name, {}).get('cost_upr', np.nan),
                                'peak_upr': upr_metrics.get(algo_name, {}).get('peak_upr', np.nan)
                            }
                            results_list.append(result_row)
                            
                    except Exception as e:
                        print(f"实验失败: {e}")
                        continue
        
        return pd.DataFrame(results_list)
    
    def generate_tcl_test_data(self, periods: int, households: int, sample: int) -> Dict:
        """
        生成基于TCL模型的测试数据 - 修正版，使用更现实的灵活性限制
        
        Args:
            periods: 时间段数
            households: TCL设备数量
            sample: 样本编号，用于保证可复现性
            
        Returns:
            测试数据字典
        """
        np.random.seed(sample)
        
        # 1. 模拟价格数据
        base_price = 0.05  # EUR/kWh
        price_variation = 0.02  # 减小价格变化
        time_hours = np.arange(periods)
        price_pattern = 1 + 0.3 * (np.sin(2*np.pi*time_hours/24 - np.pi/2) + 
                                  0.2 * np.sin(4*np.pi*time_hours/24))
        prices = base_price * price_pattern + np.random.normal(0, price_variation/3, periods)
        prices = np.maximum(prices, 0.01)

        # 2. 模拟室外温度预测 - 使用更保守的温度范围
        base_temp = 25.0  # 降低基础温度
        temp_amplitude = 3.0  # 减小温度变化幅度
        theta_a_forecast = base_temp + temp_amplitude * np.cos(2 * np.pi * (time_hours - 15) / 24)
        theta_a_forecast = np.clip(theta_a_forecast, 22.0, 28.0)  # 更窄的温度范围

        # 3. 生成TCL参数和基线需求 - 使用更严格的约束
        tcl_params_list = []
        demand_profiles = []
        
        # 修正的TCL参数分布范围 - 确保灵活性有限
        param_dist = {
            'R_th_range': (3.0, 6.0),     # 增大热阻
            'C_th_range': (2.0, 4.0),     # 减小热容，使状态约束更紧
            'P_m_range': (1.0, 2.5),      # 减小最大功率
            'eta_range': (0.9, 0.95), 
            'theta_r_range': (23.0, 24.0), # 窄设定温度范围
            'delta_val_range': (0.5, 1.5)  # 大幅减小容忍度
        }
        
        for h in range(households):
            R_th = np.random.uniform(*param_dist['R_th_range'])
            C_th = np.random.uniform(*param_dist['C_th_range'])
            P_m = np.random.uniform(*param_dist['P_m_range'])
            eta = np.random.uniform(*param_dist['eta_range'])
            theta_r = np.random.uniform(*param_dist['theta_r_range'])
            delta_val = np.random.uniform(*param_dist['delta_val_range'])
            
            # 计算衍生参数
            a = np.exp(-1.0 / (R_th * C_th))
            b = R_th * eta
            delta = (1-a) * C_th / (eta*b) if (eta*b) != 0 else 0
            
            # **关键修正：限制delta以确保状态约束起作用**
            delta = min(delta, 0.1)  # 限制delta的最大值

            params = {
                'T': periods, 'a': a, 'b': b, 'C_th': C_th, 'eta': eta,
                'delta': delta, 'P_m': P_m, 'theta_r': theta_r, 'x0': 0.0,
                'delta_val': delta_val, 'theta_a_forecast': theta_a_forecast
            }
            tcl_params_list.append(params)
            
            # 计算基线功率 (不可控部分)
            P0_unconstrained = (theta_a_forecast - theta_r) / b if b != 0 else np.zeros_like(theta_a_forecast)
            P0_forecast = np.maximum(0, P0_unconstrained)
            
            # **关键修正：添加最小基线需求，确保不能完全被抵消**
            min_baseline = 0.1  # 最小基线功率，代表不可调度的部分
            P0_forecast = np.maximum(P0_forecast, min_baseline)
            
            demand_profiles.append(P0_forecast)
            
        demands = np.array(demand_profiles).T  # 维度: (periods, households)
        
        return {
            'dt': 1,
            'periods': periods,
            'households': households,
            'sample': sample,
            'prices': prices,
            'demands': demands, # 基线功率消耗
            'tcls': tcl_params_list, # 每个TCL的详细参数
            'objectives': ['cost', 'peak']
        }

    def plot_upr_comparison(self, df: pd.DataFrame, save_path: str = None):
        """
        绘制UPR对比图
        
        Args:
            df: Results DataFrame.
            save_path: Path to save the plot.
        """
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 算法颜色映射
        g_poly_algorithms = ['G-Poly Approximate', 'G-Poly Initial Robust', 'G-Poly Final Robust']
        colors = {'G-Poly Approximate': '#1f77b4', 'G-Poly Initial Robust': '#ff7f0e', 
                 'G-Poly Final Robust': '#2ca02c', 'Zonotope': '#d62728', 
                 'Inner Homothets': '#9467bd', 'Barot Outer': '#8c564b'}
        
        # 1. Cost UPR vs Periods
        ax1 = axes[0, 0]
        for algo in g_poly_algorithms + ['Zonotope', 'Inner Homothets', 'Barot Outer']:
            algo_data = df[df['algorithm'] == algo]
            if not algo_data.empty:
                grouped = algo_data.groupby('periods')['cost_upr'].agg(['mean', 'std'])
                ax1.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                           label=algo, marker='o', color=colors.get(algo, 'gray'))
        
        ax1.set_xlabel('Time Periods')
        ax1.set_ylabel('Cost UPR (%)')
        ax1.set_title('Cost UPR vs Time Periods')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Peak UPR vs Periods  
        ax2 = axes[0, 1]
        for algo in g_poly_algorithms + ['Zonotope', 'Inner Homothets', 'Barot Outer']:
            algo_data = df[df['algorithm'] == algo]
            if not algo_data.empty:
                grouped = algo_data.groupby('periods')['peak_upr'].agg(['mean', 'std'])
                ax2.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                           label=algo, marker='s', color=colors.get(algo, 'gray'))
        
        ax2.set_xlabel('Time Periods')
        ax2.set_ylabel('Peak UPR (%)')
        ax2.set_title('Peak UPR vs Time Periods')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cost UPR vs Households
        ax3 = axes[1, 0]
        for algo in g_poly_algorithms + ['Zonotope', 'Inner Homothets', 'Barot Outer']:
            algo_data = df[df['algorithm'] == algo]
            if not algo_data.empty:
                grouped = algo_data.groupby('households')['cost_upr'].agg(['mean', 'std'])
                ax3.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                           label=algo, marker='o', color=colors.get(algo, 'gray'))
        
        ax3.set_xlabel('Number of Households')
        ax3.set_ylabel('Cost UPR (%)')
        ax3.set_title('Cost UPR vs Number of Households')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Peak UPR vs Households
        ax4 = axes[1, 1]
        for algo in g_poly_algorithms + ['Zonotope', 'Inner Homothets', 'Barot Outer']:
            algo_data = df[df['algorithm'] == algo]
            if not algo_data.empty:
                grouped = algo_data.groupby('households')['peak_upr'].agg(['mean', 'std'])
                ax4.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                           label=algo, marker='s', color=colors.get(algo, 'gray'))
        
        ax4.set_xlabel('Number of Households')
        ax4.set_ylabel('Peak UPR (%)')
        ax4.set_title('Peak UPR vs Number of Households')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """
        生成总结报告
        
        Args:
            df: 结果DataFrame
            
        Returns:
            报告字符串
        """
        report = []
        report.append("="*60)
        report.append("G-POLYMATROID PERFORMANCE COMPARISON REPORT")
        report.append("="*60)
        
        # 1. 基本统计
        algorithms = df['algorithm'].unique()
        g_poly_algos = [a for a in algorithms if a.startswith('G-Poly')]
        classical_algos = [a for a in algorithms if not a.startswith('G-Poly') and a not in ['Exact Minkowski', 'No Flexibility']]
        
        report.append(f"\n算法数量:")
        report.append(f"  - G-Polymatroid方法: {len(g_poly_algos)}")
        report.append(f"  - 经典方法: {len(classical_algos)}")
        
        # 2. 平均UPR对比
        report.append(f"\n平均UPR对比 (%):")
        report.append(f"{'算法':<30} {'Cost UPR':<12} {'Peak UPR':<12}")
        report.append("-" * 54)
        
        for algo in sorted(algorithms):
            if algo in ['Exact Minkowski', 'No Flexibility']:
                continue
            cost_upr = df[df['algorithm'] == algo]['cost_upr'].mean()
            peak_upr = df[df['algorithm'] == algo]['peak_upr'].mean()
            report.append(f"{algo:<30} {cost_upr:<12.2f} {peak_upr:<12.2f}")
        
        # 3. 最佳性能
        valid_df = df.dropna(subset=['cost_upr', 'peak_upr'])
        if not valid_df.empty:
            best_cost_algo = valid_df.loc[valid_df['cost_upr'].idxmin(), 'algorithm']
            best_peak_algo = valid_df.loc[valid_df['peak_upr'].idxmin(), 'algorithm']
            
            report.append(f"\n最佳性能:")
            report.append(f"  - 成本优化最佳: {best_cost_algo}")
            report.append(f"  - 削峰最佳: {best_peak_algo}")
        
        # 4. G-polymatroid方法内部对比
        if g_poly_algos:
            report.append(f"\nG-Polymatroid方法内部对比:")
            for algo in g_poly_algos:
                algo_data = df[df['algorithm'] == algo]
                if not algo_data.empty:
                    cost_upr = algo_data['cost_upr'].mean()
                    peak_upr = algo_data['peak_upr'].mean()
                    report.append(f"  {algo}: Cost={cost_upr:.2f}%, Peak={peak_upr:.2f}%")
        
        # 5. 计算时间对比
        report.append(f"\n平均计算时间 (秒):")
        for algo in sorted(algorithms):
            algo_time = df[df['algorithm'] == algo]['algo_time'].mean()
            report.append(f"  {algo}: {algo_time:.3f}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)

def main():
    """主函数：运行完整的性能对比"""
    print("开始G-Polymatroid性能对比实验")
    
    # 配置参数
    config = {
        'periods_list': [4, 8, 12],  # 简化测试
        'households_list': [5, 10, 15],  # 简化测试
        'num_samples': 3,  # 简化测试
        'output_dir': 'comparison_results'
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 初始化对比器
    comparator = OperationalBenefitComparator(config)
    
    # 运行系统性对比
    results_df = comparator.run_systematic_comparison(
        periods_list=config['periods_list'],
        households_list=config['households_list'], 
        num_samples=config['num_samples']
    )
    
    # 保存结果
    results_path = os.path.join(config['output_dir'], 'comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"结果已保存到: {results_path}")
    
    # 生成图表
    plot_path = os.path.join(config['output_dir'], 'upr_comparison.png')
    comparator.plot_upr_comparison(results_df, save_path=plot_path)
    
    # 生成报告
    report = comparator.generate_summary_report(results_df)
    report_path = os.path.join(config['output_dir'], 'comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n完整报告已保存到: {report_path}")

if __name__ == "__main__":
    main() 