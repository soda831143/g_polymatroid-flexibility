"""
测试 Greedy 优化和顶点分解集成效果（使用 Gurobi 求解器）

验证内容:
1. Greedy 优化算法是否正确集成 (2x speedup)
2. 顶点分解是否正确工作 (signal = Σλ_j·v_j)
3. 所有算法是否正确使用顶点分解
"""
import numpy as np
import time
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flexitroid.devices.general_der import GeneralDER, DERParameters
from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.problems.signal_tracker import SignalTracker, GUROBI_AVAILABLE


def test_greedy_optimization():
    """测试1: Greedy优化算法性能"""
    print("\n" + "="*80)
    print("测试1: Greedy 优化算法")
    print("="*80)
    
    T = 96  # 较大的T来体现性能差异
    
    # 创建简单的DER
    params = DERParameters(
        u_min=np.full(T, -2.0),
        u_max=np.full(T, 3.0),
        x_min=np.linspace(-20, -10, T),
        x_max=np.linspace(30, 50, T)
    )
    der = GeneralDER(params)
    
    # 随机成本向量
    c = np.random.randn(T)
    
    # 测试原始版本
    print("\n运行原始 Greedy 算法...")
    start = time.perf_counter()
    u_original = der._solve_greedy_original(c)
    time_original = time.perf_counter() - start
    
    # 测试优化版本
    print("运行优化 Greedy 算法...")
    start = time.perf_counter()
    u_optimized = der._solve_greedy_optimized(c)
    time_optimized = time.perf_counter() - start
    
    # 验证结果一致性
    error = np.linalg.norm(u_original - u_optimized)
    speedup = time_original / time_optimized if time_optimized > 0 else float('inf')
    
    print(f"\n原始版本耗时: {time_original*1000:.3f} ms")
    print(f"优化版本耗时: {time_optimized*1000:.3f} ms")
    print(f"加速比: {speedup:.2f}x")
    print(f"结果误差: {error:.2e}")
    
    if error < 1e-10:
        print("✅ Greedy优化算法集成成功，结果完全一致")
    else:
        print(f"❌ 警告: 结果不一致，误差={error}")
    
    if speedup > 1.3:
        print(f"✅ 性能提升明显 ({speedup:.2f}x)")
    else:
        print(f"⚠️  性能提升不明显 ({speedup:.2f}x)")
    
    return error < 1e-10


def test_vertex_disaggregation():
    """测试2: 顶点分解功能（使用Gurobi）"""
    print("\n" + "="*80)
    print("测试2: 顶点分解 (Gurobi求解器)")
    print("="*80)
    
    if not GUROBI_AVAILABLE:
        print("❌ Gurobi未安装，跳过顶点分解测试")
        return False
    
    T = 24
    N = 5  # 5个设备
    
    # 创建异构设备（不同的参数）
    devices = []
    for i in range(N):
        params = DERParameters(
            u_min=np.full(T, -1.0 - 0.2*i),
            u_max=np.full(T, 2.0 + 0.3*i),
            x_min=np.linspace(-10-i, -5-i, T),
            x_max=np.linspace(20+i*2, 40+i*2, T)
        )
        devices.append(GeneralDER(params))
    
    # 创建聚合器
    aggregator = Aggregator(devices)
    
    # 目标信号（聚合最优解）
    c = np.random.randn(T)
    signal = aggregator.solve_linear_program(c)
    print(f"\n目标聚合信号范围: [{signal.min():.3f}, {signal.max():.3f}]")
    
    # 执行顶点分解
    print("\n执行顶点分解...")
    start = time.perf_counter()
    try:
        individual_signals = aggregator.disaggregate(signal)
        time_disagg = time.perf_counter() - start
        
        # 验证分解正确性
        reconstructed = np.sum(individual_signals, axis=0)
        error = np.linalg.norm(reconstructed - signal)
        
        print(f"\n分解耗时: {time_disagg:.3f}s")
        print(f"重构误差: {error:.2e}")
        print(f"各设备信号范围:")
        for i in range(N):
            u_i = individual_signals[i]
            print(f"  设备 {i}: [{u_i.min():.3f}, {u_i.max():.3f}]")
        
        # 验证每个个体信号在各自的可行域内
        all_feasible = True
        for i, (device, u_i) in enumerate(zip(devices, individual_signals)):
            # 简单检查: 是否满足功率约束
            u_min = device.params.u_min
            u_max = device.params.u_max
            
            if np.any(u_i < u_min - 1e-6) or np.any(u_i > u_max + 1e-6):
                print(f"❌ 设备 {i} 的信号超出可行域")
                all_feasible = False
        
        if error < 1e-3:
            print("✅ 顶点分解成功，重构误差极小")
        else:
            print(f"❌ 警告: 重构误差较大 ({error:.2e})")
        
        if all_feasible:
            print("✅ 所有个体信号均在各自可行域内")
        else:
            print("❌ 部分个体信号超出可行域")
        
        return error < 1e-3 and all_feasible
        
    except Exception as e:
        print(f"❌ 顶点分解失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_tracker():
    """测试3: SignalTracker 单独功能（使用Gurobi）"""
    print("\n" + "="*80)
    print("测试3: SignalTracker 凸组合求解 (Gurobi)")
    print("="*80)
    
    if not GUROBI_AVAILABLE:
        print("❌ Gurobi未安装，跳过SignalTracker测试")
        return False
    
    T = 24
    
    # 创建简单DER
    params = DERParameters(
        u_min=np.full(T, -1.5),
        u_max=np.full(T, 2.5),
        x_min=np.linspace(-15, -8, T),
        x_max=np.linspace(25, 45, T)
    )
    der = GeneralDER(params)
    
    # 生成可达信号（通过线性规划）
    c = np.sin(np.linspace(0, 4*np.pi, T))
    signal = der.solve_linear_program(c)
    print(f"\n目标信号范围: [{signal.min():.3f}, {signal.max():.3f}]")
    
    # 使用SignalTracker求解
    print("\n运行 SignalTracker...")
    tracker = SignalTracker(der, signal, max_iters=100)
    start = time.perf_counter()
    try:
        solution = tracker.solve()
        time_track = time.perf_counter() - start
        
        # 获取顶点和权重
        vertices, weights = tracker.get_vertices_and_weights()
        
        # 验证凸组合
        if len(weights) > 0:
            convex_comb = vertices.T @ weights
            error = np.linalg.norm(convex_comb - signal)
        else:
            error = np.linalg.norm(solution - signal)
        
        print(f"\n求解耗时: {time_track:.3f}s")
        print(f"找到顶点数: {len(weights)}")
        if len(weights) > 0:
            print(f"权重和: {np.sum(weights):.6f}")
            print(f"凸组合误差: {error:.2e}")
        print(f"最终解误差: {np.linalg.norm(solution - signal):.2e}")
        
        if error < 1e-6:
            print("✅ SignalTracker 成功找到精确凸组合表示")
            return True
        elif error < 1e-3:
            print(f"⚠️  SignalTracker 找到近似凸组合 (误差={error:.2e})")
            return True
        else:
            print(f"❌ SignalTracker 凸组合误差过大 ({error:.2e})")
            return False
            
    except Exception as e:
        print(f"❌ SignalTracker失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("Greedy优化和顶点分解集成测试 (Gurobi版本)")
    print("="*80)
    
    if not GUROBI_AVAILABLE:
        print("\n⚠️  警告: Gurobi未安装！")
        print("请安装Gurobi: pip install gurobipy")
        print("并确保有有效的Gurobi许可证")
        return False
    
    results = {}
    
    # 测试1: Greedy优化
    try:
        results['greedy'] = test_greedy_optimization()
    except Exception as e:
        print(f"\n❌ Greedy优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['greedy'] = False
    
    # 测试2: 顶点分解
    try:
        results['disaggregation'] = test_vertex_disaggregation()
    except Exception as e:
        print(f"\n❌ 顶点分解测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['disaggregation'] = False
    
    # 测试3: SignalTracker
    try:
        results['signal_tracker'] = test_signal_tracker()
    except Exception as e:
        print(f"\n❌ SignalTracker测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['signal_tracker'] = False
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("🎉 所有测试通过！集成成功！")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
