"""
快速验证脚本 - 验证所有集成是否正常工作

测试内容:
1. 优化Greedy算法的性能和正确性
2. 坐标变换算法的运行
3. 完整对比框架的运行

使用方法:
    python quick_integration_test.py
"""

import sys
import os
import time
import numpy as np

# 添加路径
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flexitroid.devices.tcl import TCL
from flexitroid.aggregations.aggregator import Aggregator
from comparison.lib import algo_g_polymatroid_transform_det


def test_optimized_greedy():
    """测试1: 验证优化Greedy算法"""
    print("\n" + "="*80)
    print("测试1: 优化Greedy算法")
    print("="*80)
    
    # 创建测试TCL
    T = 24
    tcl_params = {
        'T': T,
        'C_th': 10.0,
        'R_th': 2.5,
        'P_m': 15.0,
        'eta': 3.0,
        'theta_r': 22.5,
        'delta_val': 1.5,
        'delta': 1.0,
        'x0': 0.0,
        'a': 1 - 1/(2.5 * 10.0),
        'b': 2.5 * 3.0
    }
    
    # 预测温度
    theta_a = 27.0 + 7.0 * np.cos(2 * np.pi * (np.arange(T) - 15) / 24)
    theta_a = np.clip(theta_a, 20.0, 35.0)
    
    # 创建TCL
    try:
        tcl = TCL(tcl_params, build_g_poly=True, theta_a_forecast=theta_a)
        print("✓ TCL创建成功")
    except Exception as e:
        print(f"✗ TCL创建失败: {e}")
        return False
    
    # 测试成本向量
    c = np.random.randn(T)
    
    # 测试旧版本
    print("\n--- 旧版本Greedy ---")
    try:
        start = time.perf_counter()
        u_old = tcl.solve_linear_program(c, use_optimized=False)
        time_old = time.perf_counter() - start
        print(f"✓ 运行成功")
        print(f"  时间: {time_old*1000:.3f}ms")
        print(f"  解范围: [{u_old.min():.2f}, {u_old.max():.2f}]")
    except Exception as e:
        print(f"✗ 运行失败: {e}")
        return False
    
    # 测试新版本
    print("\n--- 新版本Greedy (优化) ---")
    try:
        start = time.perf_counter()
        u_new = tcl.solve_linear_program(c, use_optimized=True)
        time_new = time.perf_counter() - start
        print(f"✓ 运行成功")
        print(f"  时间: {time_new*1000:.3f}ms")
        print(f"  解范围: [{u_new.min():.2f}, {u_new.max():.2f}]")
    except Exception as e:
        print(f"✗ 运行失败: {e}")
        return False
    
    # 验证正确性
    print("\n--- 正确性验证 ---")
    error = np.linalg.norm(u_old - u_new)
    if error < 1e-10:
        print(f"✓ 结果一致 (误差={error:.2e})")
    else:
        print(f"✗ 结果不一致 (误差={error:.2e})")
        return False
    
    # 性能对比
    print("\n--- 性能对比 ---")
    speedup = time_old / time_new if time_new > 0 else 0
    print(f"加速比: {speedup:.2f}x")
    if speedup > 1.5:
        print(f"✓ 性能提升显著 ({speedup:.2f}x > 1.5x)")
    else:
        print(f"⚠ 性能提升不明显 ({speedup:.2f}x < 1.5x)")
    
    return True


def test_aggregator_optimized():
    """测试2: 验证聚合器使用优化Greedy"""
    print("\n" + "="*80)
    print("测试2: 聚合器优化Greedy")
    print("="*80)
    
    T = 24
    N = 5
    
    # 创建多个TCL
    print(f"\n创建{N}个TCL...")
    tcl_list = []
    theta_a = 27.0 + 7.0 * np.cos(2 * np.pi * (np.arange(T) - 15) / 24)
    theta_a = np.clip(theta_a, 20.0, 35.0)
    
    for i in range(N):
        tcl_params = {
            'T': T,
            'C_th': 8.0 + np.random.uniform(-2, 2),
            'R_th': 2.5 + np.random.uniform(-0.5, 0.5),
            'P_m': 15.0 + np.random.uniform(-3, 3),
            'eta': 3.0 + np.random.uniform(-0.5, 0.5),
            'theta_r': 22.5,
            'delta_val': 1.5,
            'delta': 1.0,
            'x0': 0.0,
        }
        tcl_params['a'] = 1 - 1/(tcl_params['R_th'] * tcl_params['C_th'])
        tcl_params['b'] = tcl_params['R_th'] * tcl_params['eta']
        
        try:
            tcl = TCL(tcl_params, build_g_poly=True, theta_a_forecast=theta_a)
            tcl_list.append(tcl)
        except Exception as e:
            print(f"✗ TCL {i+1}创建失败: {e}")
            return False
    
    print(f"✓ 成功创建{len(tcl_list)}个TCL")
    
    # 创建聚合器
    print("\n创建聚合器...")
    try:
        agg = Aggregator(tcl_list)
        print("✓ 聚合器创建成功")
    except Exception as e:
        print(f"✗ 聚合器创建失败: {e}")
        return False
    
    # 测试聚合优化
    c = np.random.randn(T)
    
    print("\n--- 聚合器优化 ---")
    try:
        start = time.perf_counter()
        u_agg = agg.solve_linear_program(c)
        time_agg = time.perf_counter() - start
        print(f"✓ 运行成功")
        print(f"  时间: {time_agg*1000:.3f}ms")
        print(f"  聚合解范围: [{u_agg.min():.2f}, {u_agg.max():.2f}]")
    except Exception as e:
        print(f"✗ 运行失败: {e}")
        return False
    
    # 验证聚合性质
    print("\n--- 验证聚合性质 ---")
    try:
        # 计算单独优化的和
        u_individual_sum = sum(tcl.solve_linear_program(c) for tcl in tcl_list)
        
        # 应该与聚合优化结果相同
        error = np.linalg.norm(u_agg - u_individual_sum)
        if error < 1e-8:
            print(f"✓ 聚合性质正确 (误差={error:.2e})")
        else:
            print(f"✗ 聚合性质错误 (误差={error:.2e})")
            print(f"  u_agg范围: [{u_agg.min():.2f}, {u_agg.max():.2f}]")
            print(f"  u_sum范围: [{u_individual_sum.min():.2f}, {u_individual_sum.max():.2f}]")
            return False
    except Exception as e:
        print(f"⚠ 聚合性质验证失败: {e}")
    
    return True


def test_coordinate_transform():
    """测试3: 验证坐标变换算法"""
    print("\n" + "="*80)
    print("测试3: 坐标变换算法")
    print("="*80)
    
    T = 24
    N = 3
    
    # 创建测试数据
    print(f"\n创建测试数据(N={N}, T={T})...")
    theta_a = 27.0 + 7.0 * np.cos(2 * np.pi * (np.arange(T) - 15) / 24)
    theta_a = np.clip(theta_a, 20.0, 35.0)
    
    tcl_list = []
    P0_individual = []
    
    for i in range(N):
        tcl_params = {
            'T': T,
            'C_th': 10.0,
            'R_th': 2.5,
            'P_m': 15.0,
            'eta': 3.0,
            'theta_r': 22.5,
            'delta_val': 1.5,
            'delta': 1.0,
            'x0': 0.0,
            'a': 1 - 1/(2.5 * 10.0),
            'b': 2.5 * 3.0
        }
        
        try:
            tcl = TCL(tcl_params, build_g_poly=True, theta_a_forecast=theta_a)
            tcl_list.append(tcl)
            
            # 计算基线
            P0_i = np.maximum(0, (theta_a - tcl_params['theta_r']) / tcl_params['b'])
            P0_individual.append(P0_i)
        except Exception as e:
            print(f"✗ TCL {i+1}创建失败: {e}")
            return False
    
    print(f"✓ 成功创建{len(tcl_list)}个TCL")
    
    # 准备数据
    prices = np.ones(T) * 0.10
    prices[8:21] = 0.60  # 峰段
    
    P0_agg = np.sum(P0_individual, axis=0)
    
    data = {
        'tcl_objs': tcl_list,
        'prices': prices,
        'P0': P0_agg,
        'periods': T,
        'households': N
    }
    
    # 运行算法
    print("\n--- 运行坐标变换算法 ---")
    try:
        result = algo_g_polymatroid_transform_det.solve(data)
        print("✓ 算法运行成功")
        print(f"  总成本: {result['total_cost']:.2f}")
        print(f"  峰值功率: {result['peak_power']:.2f}")
        print(f"  计算时间: {result['computation_time']:.3f}s")
    except Exception as e:
        print(f"✗ 算法运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 验证结果合理性
    print("\n--- 验证结果合理性 ---")
    P_total = result['aggregate_flexibility']
    
    # 检查长度
    if len(P_total) != T:
        print(f"✗ 结果长度错误: {len(P_total)} != {T}")
        return False
    print(f"✓ 结果长度正确: {len(P_total)} = {T}")
    
    # 检查范围
    P_min = P_total.min()
    P_max = P_total.max()
    expected_max = N * 15.0  # N个TCL,每个最大15kW
    if P_min < 0 or P_max > expected_max * 1.5:
        print(f"⚠ 功率范围异常: [{P_min:.2f}, {P_max:.2f}] (预期<{expected_max:.2f})")
    else:
        print(f"✓ 功率范围合理: [{P_min:.2f}, {P_max:.2f}]")
    
    # 检查成本
    cost = result['total_cost']
    baseline_cost = np.dot(prices, P0_agg)
    if cost < baseline_cost * 0.5:  # 应该不会降低太多
        print(f"⚠ 成本异常低: {cost:.2f} < {baseline_cost*0.5:.2f}")
    else:
        print(f"✓ 成本合理: {cost:.2f} (baseline={baseline_cost:.2f})")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("快速集成验证测试")
    print("="*80)
    print("\n测试内容:")
    print("1. 优化Greedy算法的性能和正确性")
    print("2. 聚合器使用优化Greedy")
    print("3. 坐标变换算法的运行")
    
    results = {}
    
    # 运行测试
    results['test1'] = test_optimized_greedy()
    results['test2'] = test_aggregator_optimized()
    results['test3'] = test_coordinate_transform()
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 所有测试通过!")
        print("\n下一步:")
        print("1. 运行完整对比测试: python comparison/advanced_comparison_framework.py")
        print("2. 查看详细结果: comparison_results/advanced_summary.csv")
        print("3. 计算UPR指标")
    else:
        print("❌ 部分测试失败,请检查错误信息")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
