"""
算法: G-Polymatroid 确定性坐标变换聚合
流程: 名义边界 → 坐标变换 → 聚合 → 优化
特点: 使用名义物理边界(无鲁棒化),统一到坐标变换框架

这是三种g-polymatroid方法中的确定性版本:
1. 确定性 (本文件): Transform(h_nom) → Aggregate
2. SRO: Robustify-SRO(D) → Transform(h_sro) → Aggregate
3. Re-SRO: Robustify-ReSRO(D, u0) → Transform(h_resro) → Aggregate
"""
import numpy as np
from typing import List, Dict, Tuple
import sys
import os
import time

# 添加路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.devices.tcl import TCL
from flexitroid.devices.general_der import GeneralDER, DERParameters
from .peak_optimization import optimize_cost_column_generation, optimize_peak_column_generation
from .correct_tcl_gpoly import CorrectTCL_GPoly  # 【关键修正】使用正确的p/b实现


def _solve_cost_optimization(aggregator, prices, P0_agg, T, tcl_objs):
    """
    成本优化：使用列生成框架
    
    【正确的方法】：列生成（Dantzig-Wolfe分解）
    
    问题：异构坐标变换导致的目标函数耦合
    - 物理成本：J = Σ_t c[t]·(P0[t] + Σ_i (γ_i[t]·ũ_i[t]))
    - 可行集：ũ_agg = Σ_i ũ_i ∈ F_agg
    - 问题：不存在只关于ũ_agg的"虚拟目标函数"
    
    解决方案：列生成迭代
    1. 主问题：在F_agg的顶点凸组合上优化物理成本
    2. 子问题：通过贪心算法生成改善顶点
    3. 迭代直到收敛
    """
    u0_physical_individual, u0_physical_agg, total_cost = optimize_cost_column_generation(
        aggregator, prices, P0_agg, tcl_objs, T
    )
    
    # 从物理个体信号反推虚拟个体信号（用于其他需要虚拟信号的地方）
    N = len(tcl_objs)
    u0_virtual_individual = np.zeros((N, T))
    for i, tcl in enumerate(tcl_objs):
        a_i = tcl.a
        delta_i = tcl.delta
        for t in range(T):
            scale = (a_i ** t) / delta_i if delta_i > 1e-10 else 1.0
            u0_virtual_individual[i, t] = u0_physical_individual[i, t] / scale if abs(scale) > 1e-10 else 0.0
    
    # 虚拟聚合信号
    u0_virtual_agg = np.sum(u0_virtual_individual, axis=0)
    
    return u0_virtual_individual, u0_virtual_agg


def _solve_peak_optimization(aggregator, P0_agg, tcl_objs, T):
    """
    峰值优化：使用列生成框架
    
    【正确的方法】：列生成（Dantzig-Wolfe分解）
    
    问题同成本优化，但目标函数是L-infinity
    - 物理峰值：P_peak = max_t(P0[t] + Σ_i (γ_i[t]·ũ_i[t]))
    - 可行集：ũ_agg = Σ_i ũ_i ∈ F_agg
    - 同样需要列生成来保证物理最优
    """
    u0_physical_individual, u0_physical_agg, peak_value = optimize_peak_column_generation(
        aggregator, P0_agg, tcl_objs, T
    )
    
    # 从物理信号反推虚拟信号
    N = len(tcl_objs)
    u0_virtual_individual = np.zeros((N, T))
    for i, tcl in enumerate(tcl_objs):
        a_i = tcl.a
        delta_i = tcl.delta
        for t in range(T):
            scale = (a_i ** t) / delta_i if delta_i > 1e-10 else 1.0
            u0_virtual_individual[i, t] = u0_physical_individual[i, t] / scale if abs(scale) > 1e-10 else 0.0
    
    u0_virtual_agg = np.sum(u0_virtual_individual, axis=0)
    
    return u0_virtual_individual, u0_virtual_agg


def solve(data: dict, tcl_objs: List = None, objective='cost') -> dict:
    """
    确定性坐标变换G-Polymatroid聚合算法
    
    流程:
    1. 获取名义物理边界 (无鲁棒化)
    2. 物理坐标变换到虚拟坐标 (有损→无损)
    3. 聚合所有TCL的虚拟g-polymatroid
    4. 在虚拟坐标下优化 (成本或峰值)
    5. 逆变换回物理坐标
    
    Args:
        data: 包含以下键的字典
            - 'prices': 电价 (T,)
            - 'P0': 基线功率 (T,) 
            - 'tcl_objs': TCL对象列表
            - 'periods': 时间步数
            - 'households': TCL数量
        tcl_objs: TCL对象列表(可选)
        objective: 优化目标 ('cost' 或 'peak')
    
    Returns:
        结果字典 {
            'aggregate_flexibility': 聚合后的物理功率序列,
            'total_cost': 总成本,
            'peak_power': 峰值功率,
            'computation_time': 计算时间,
            'objective': 优化目标
        }
    """
    start_time = time.time()
    
    # 1. 解析输入数据
    prices = data['prices']
    P0_agg = data['P0']
    T = data['periods']
    N = data['households']
    
    if tcl_objs is None:
        tcl_objs = data.get('tcl_objs', None)
    
    if tcl_objs is None:
        raise ValueError("必须提供tcl_objs(TCL对象列表)")
    
    print("\n" + "="*80)
    print("确定性坐标变换 G-Polymatroid 聚合算法")
    print("="*80)
    print(f"TCL数量: {N}, 时间步数: {T}")
    print("使用名义边界 (无鲁棒化)")
    
    # 2. 获取名义物理边界 (跳过鲁棒化步骤)
    print("\n--- 阶段1: 使用名义物理边界 (确定性) ---")
    print("跳过鲁棒化,直接使用预测的名义边界")
    
    # 3. 坐标变换: Physical → Virtual (使用名义边界)
    print("\n--- 阶段2: 坐标变换 (Physical → Virtual) ---")
    tcl_virtual_list = []
    
    for i, tcl in enumerate(tcl_objs):
        # 获取TCL物理参数
        a = tcl.a
        delta = tcl.delta
        x0 = tcl.tcl_params.get('x0', 0.0)
        
        # 物理状态约束: -x_plus <= x(t) <= x_plus
        x_plus = (tcl.C_th * tcl.tcl_params['delta_val']) / tcl.eta if tcl.eta > 0 else 100.0
        
        # 计算虚拟坐标下的功率边界
        P0_i = (tcl.theta_a_forecast - tcl.theta_r) / tcl.b_coef
        P0_i = np.maximum(0, P0_i)
        u_min_phys = -P0_i
        u_max_phys = tcl.P_m - P0_i
        
        # 虚拟功率边界: ũ(t) = δ·u(t)/a^t
        u_min_virt = np.zeros(T)
        u_max_virt = np.zeros(T)
        for t in range(T):
            scale = delta / (a ** t) if abs(a ** t) > 1e-10 else delta
            u_min_virt[t] = u_min_phys[t] * scale
            u_max_virt[t] = u_max_phys[t] * scale
        
        # 虚拟状态约束修正!
        # 物理: x(t) = a^t·x0 + δ·Σ_{s=0}^{t-1} a^{t-1-s}·u(s)
        # 虚拟: x̃(t) = x(t)/a^t = x0 + Σ_{s=0}^{t-1} ũ(s)
        # 
        # 物理约束: -x_plus <= x(t) <= x_plus (对所有t)
        # ∴ 虚拟约束: -x_plus/a^t <= x̃(t) <= x_plus/a^t
        # ∴ 累积约束: -x_plus/a^t - x0 <= Σ_{s=0}^{t-1} ũ(s) <= x_plus/a^t - x0
        #
        # GeneralDER中的x_min[t]表示Σ_{s=0}^{t} ũ(s)(包括第t个)的范围
        # 但这里我们保持与GeneralDER一致:
        # x_min[t] = -x_plus/a^(t+1) - x0 (至时刻t的累积,对应虚拟状态x̃(t+1))
        # x_max[t] = x_plus/a^(t+1) - x0
        #
        # 关键修正: 确保约束严格来自**物理状态约束**,不混合功率累积
        
        y_lower_virt = np.zeros(T)
        y_upper_virt = np.zeros(T)
        
        for t in range(T):
            # GeneralDER期望x_min[t], x_max[t]表示Σ_{s=0}^{t} ũ(s)的范围
            # 这对应虚拟状态在时刻t+1: x̃(t+1) = x0 + Σ_{s=0}^{t} ũ(s)
            # 由物理约束: -x_plus/a^(t+1) <= x̃(t+1) <= x_plus/a^(t+1)
            # 所以: -x_plus/a^(t+1) - x0 <= Σ_{s=0}^{t} ũ(s) <= x_plus/a^(t+1) - x0
            
            power_denom = a ** (t+1) if abs(a ** (t+1)) > 1e-10 else 1e-10
            y_lower_virt[t] = -x_plus / power_denom - x0
            y_upper_virt[t] = x_plus / power_denom - x0
        
        # 【关键修正】使用正确的p/b实现,而不是GeneralDER的DP算法
        # GeneralDER的DP算法无法正确处理指数衰减的虚拟边界
        params_virtual = DERParameters(
            u_min=u_min_virt,
            u_max=u_max_virt,
            x_min=y_lower_virt,
            x_max=y_upper_virt
        )
        tcl_virtual = CorrectTCL_GPoly(params_virtual)  # 使用显式LP求解p/b
        
        # 【调试】第一个TCL时输出虚拟边界信息
        if i == 0:
            print(f"  [DEBUG] TCL 0: 创建 CorrectTCL_GPoly 实例")
            print(f"  [DEBUG] TCL 0 虚拟功率边界: u_min_virt=[{u_min_virt.min():.3f}, {u_min_virt.max():.3f}], u_max_virt=[{u_max_virt.min():.3f}, {u_max_virt.max():.3f}]")
            print(f"  [DEBUG] TCL 0 虚拟能量边界: y_lower_virt=[{y_lower_virt.min():.3f}, {y_lower_virt.max():.3f}], y_upper_virt=[{y_upper_virt.min():.3f}, {y_upper_virt.max():.3f}]")
            print(f"  [DEBUG] TCL 0 物理参数: a={a:.4f}, delta={tcl.delta:.4f}, x_plus={x_plus:.2f}")
        
        tcl_virtual_list.append(tcl_virtual)
    
    print(f"完成 {len(tcl_virtual_list)} 个TCL的坐标变换")
    print(f"  [DEBUG] 所有TCL类型: {[type(tcl).__name__ for tcl in tcl_virtual_list[:3]]}...")
    
    # 4. 聚合虚拟g-polymatroid
    print("\n--- 阶段3: 聚合虚拟g-polymatroid ---")
    aggregator = Aggregator(tcl_virtual_list)
    agg_virtual = aggregator
    print("虚拟坐标聚合完成")
    print(f"  [DEBUG] Aggregator.fleet 中设备类型: {[type(d).__name__ for d in aggregator.fleet[:3]]}...")
    
    # 5. 虚拟坐标下优化 (根据目标选择算法)
    print(f"\n--- 阶段4: 虚拟坐标优化 (目标: {objective}) ---")
    
    # 初始化（防止未赋值）
    u0_virtual_individual = np.zeros((N, T))
    u0_virtual_agg = np.zeros(T)
    
    if objective == 'cost':
        print("目标: 最小化成本")
        u0_virtual_individual_cost, u0_virtual_agg_cost = _solve_cost_optimization(
            agg_virtual, prices, P0_agg, T, tcl_objs
        )
        print("成本优化完成 (使用列生成框架)")
        
        # 【关键修正】将优化结果赋给主变量
        u0_virtual_individual = u0_virtual_individual_cost
        u0_virtual_agg = u0_virtual_agg_cost
        
    elif objective == 'peak':
        print("目标: 最小化峰值功率")
        u0_virtual_individual_peak, u0_virtual_agg_peak = _solve_peak_optimization(
            agg_virtual, P0_agg, tcl_objs, T
        )
        print("峰值优化完成 (使用列生成框架)")
        
        # 【关键修正】将优化结果赋给主变量
        u0_virtual_individual = u0_virtual_individual_peak
        u0_virtual_agg = u0_virtual_agg_peak
    else:
        raise ValueError(f"未知的优化目标: {objective}")
    
    # 6.5 【新增】顶点分解: 将聚合信号分解为个体信号
    if objective == 'cost':
        print("\n--- 阶段5.5: 虚拟信号验证 ---")
        print(f"虚拟个体信号数: {u0_virtual_individual.shape[0]}")
        print(f"各TCL虚拟信号范围:")
        for i in range(min(3, len(u0_virtual_individual))):
            u_i = u0_virtual_individual[i]
            print(f"  TCL {i}: [{u_i.min():.3f}, {u_i.max():.3f}]")
        if len(u0_virtual_individual) > 3:
            print(f"  ... (共{len(u0_virtual_individual)}个)")
    
    # 7. 逆变换到物理坐标 (对每个TCL单独进行)
    print("\n--- 阶段6: 逆变换到物理坐标 (Individual) ---")
    
    u0_physical_individual = []
    for i, (tcl, u_virt_i) in enumerate(zip(tcl_objs, u0_virtual_individual)):
        # 使用各TCL自己的参数进行逆变换
        a_i = tcl.a
        delta_i = tcl.delta
        
        u_phys_i = np.zeros(T)
        for t in range(T):
            scale = (a_i ** t) / delta_i if delta_i > 1e-10 else 1.0
            u_phys_i[t] = u_virt_i[t] * scale
        
        u0_physical_individual.append(u_phys_i)
    
    u0_physical_individual = np.array(u0_physical_individual)
    print(f"逆变换完成: {u0_physical_individual.shape}")
    
    # 聚合物理信号
    u0_physical_agg = np.sum(u0_physical_individual, axis=0)
    
    print(f"使用个体逆变换 (每个TCL使用自己的a_i, delta_i参数)")
    
    # 转换为实际功率
    P_total = u0_physical_agg + P0_agg
    
    # 8. 计算指标
    total_cost = np.dot(prices, P_total)
    peak_power = np.max(P_total)
    computation_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"确定性坐标变换算法完成 (目标: {objective})")
    print(f"总成本: {total_cost:.2f}")
    print(f"峰值功率: {peak_power:.2f}")
    print(f"计算时间: {computation_time:.3f}s")
    print("="*80)
    
    return {
        'aggregate_flexibility': P_total,
        'individual_flexibility': u0_physical_individual,
        'total_cost': total_cost,
        'peak_power': peak_power,
        'computation_time': computation_time,
        'objective': objective,
        'algorithm': f'G-Polymatroid-Transform-Det-{objective.upper()}'
    }


def algo(data: dict) -> dict:
    """
    兼容性包装函数
    """
    return solve(data)
