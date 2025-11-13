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
from .peak_optimization import optimize_peak_l_infinity


def _solve_cost_optimization(aggregator, prices, P0_agg, T, tcl_objs):
    """
    成本优化在虚拟聚合坐标中：保证物理最优
    
    关键修正：使用求和而非平均！
    
    物理成本（所有TCL相同电费）：
        J = Σ_t c[t]·P[t] = Σ_t c[t]·(P0[t] + Σ_i u_i[t])
    
    虚拟变换：u_i[t] = (a_i^t/δ_i)·ũ_i[t]
    
    虚拟聚合成本：
        J = const + Σ_i Σ_t c[t]·(a_i^t/δ_i)·ũ_i[t]
          = const + Σ_t c[t]·(Σ_i a_i^t/δ_i)·ũ_agg[t]
    
    因此虚拟聚合目标函数系数（关键！）：
        c̃_agg[t] = c[t]·(Σ_i a_i^t/δ_i)  
        
    注意：是对所有i求和，不是平均！
    """
    # 计算虚拟聚合目标系数：c̃_agg[t] = c[t]·Σ_i(a_i^t/δ_i)
    N = len(tcl_objs) if tcl_objs else 1
    c_virtual_agg = np.zeros(T)
    
    for t in range(T):
        # 计算该时刻所有TCL的变换系数之和
        scale_sum = 0
        for i in range(N):
            tcl = tcl_objs[i]
            scale = (tcl.a ** t) / tcl.delta if tcl.delta > 1e-10 else 1.0
            scale_sum += scale
        
        # 虚拟聚合目标系数 = 电费 × 变换系数之和
        c_virtual_agg[t] = prices[t] * scale_sum
    
    # 在虚拟聚合空间优化
    u0_virtual_agg = aggregator.solve_linear_program(c_virtual_agg)
    
    # 分解为个体虚拟信号
    u0_virtual_individual = aggregator.disaggregate(u0_virtual_agg)
    
    return u0_virtual_individual, u0_virtual_agg


def _solve_peak_optimization(aggregator, P0_agg, tcl_objs, T):
    """
    峰值优化: 在虚拟坐标中求解L∞规划
    """
    return optimize_peak_l_infinity(aggregator, P0_agg, tcl_objs, T)


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
        
        # 虚拟累积能量边界 (关键修正!)
        # 物理: x(t) = a^t·x0 + δ·Σ_{s=0}^{t-1} a^{t-1-s}·u(s)
        # 虚拟: x̃(t) = x(t)/a^t = x0 + Σ_{s=0}^{t-1} ũ(s)
        # 物理约束: -x_plus <= x(t) <= x_plus
        # 虚拟约束: -x_plus/a^t <= x̃(t) <= x_plus/a^t
        # 
        # 对于GeneralDER中的x_min[t], x_max[t] (表示Σ_{s=0}^{t} ũ(s)的范围):
        # x̃(t+1) = x0 + Σ_{s=0}^{t} ũ(s)
        # 所以: Σ_{s=0}^{t} ũ(s) = x̃(t+1) - x0
        # 边界: -x_plus/a^(t+1) - x0 <= Σ_{s=0}^{t} ũ(s) <= x_plus/a^(t+1) - x0
        #
        # 关键: 状态约束通常比功率累积约束更严格,应该**以状态约束为主**
        
        y_lower_virt = np.zeros(T)
        y_upper_virt = np.zeros(T)
        
        for t in range(T):
            # 基于物理状态约束的累积和
            # x̃(t+1) 的范围决定了 Σ_{s=0}^{t} ũ(s) 的范围
            x_tilde_t_plus_1_min = -x_plus / (a ** (t+1)) if abs(a ** (t+1)) > 1e-10 else -x_plus
            x_tilde_t_plus_1_max = x_plus / (a ** (t+1)) if abs(a ** (t+1)) > 1e-10 else x_plus
            
            y_lower_state = x_tilde_t_plus_1_min - x0
            y_upper_state = x_tilde_t_plus_1_max - x0
            
            # 基于功率边界的累积和(作为备选)
            y_lower_power = np.sum(u_min_virt[:t+1])
            y_upper_power = np.sum(u_max_virt[:t+1])
            
            # 优先使用状态约束(更严格且准确)
            # 只有当状态约束范围过小(<功率累积的10%)时才回退到功率边界
            state_range = y_upper_state - y_lower_state
            power_range = y_upper_power - y_lower_power
            
            if state_range < 0.01 * abs(power_range) and power_range > 0:
                # 状态约束过于严格,使用功率边界
                y_lower_virt[t] = y_lower_power
                y_upper_virt[t] = y_upper_power
            else:
                # 使用状态约束
                y_lower_virt[t] = y_lower_state
                y_upper_virt[t] = y_upper_state
        
        # 验证边界有效性
        if not np.all(y_lower_virt <= y_upper_virt):
            print(f"警告: TCL {i} 虚拟边界无效,使用保守边界")
            # 回退到功率边界
            y_lower_virt = np.cumsum(u_min_virt)
            y_upper_virt = np.cumsum(u_max_virt)
        
        # 创建虚拟坐标下的TCL
        params_virtual = DERParameters(
            u_min=u_min_virt,
            u_max=u_max_virt,
            x_min=y_lower_virt,
            x_max=y_upper_virt
        )
        tcl_virtual = GeneralDER(params_virtual)
        tcl_virtual_list.append(tcl_virtual)
    
    print(f"完成 {len(tcl_virtual_list)} 个TCL的坐标变换")
    
    # 4. 聚合虚拟g-polymatroid
    print("\n--- 阶段3: 聚合虚拟g-polymatroid ---")
    aggregator = Aggregator(tcl_virtual_list)
    agg_virtual = aggregator
    print("虚拟坐标聚合完成")
    
    # 5. 虚拟坐标下优化 (根据目标选择算法)
    print(f"\n--- 阶段4: 虚拟坐标优化 (目标: {objective}) ---")
    
    if objective == 'cost':
        print("目标: 最小化成本")
        u0_virtual_individual, u0_virtual_agg = _solve_cost_optimization(
            agg_virtual, prices, P0_agg, T, tcl_objs
        )
    elif objective == 'peak':
        print("目标: 最小化峰值功率")
        u0_virtual_individual, u0_virtual_agg = _solve_peak_optimization(
            agg_virtual, P0_agg, tcl_objs, T
        )
    else:
        raise ValueError(f"未知的优化目标: {objective}")
    
    # 6.5 【新增】顶点分解: 将聚合信号分解为个体信号
    if objective == 'cost':
        print("\n--- 阶段5.5: 顶点分解 (Vertex Disaggregation) ---")
        print(f"分解为 {u0_virtual_individual.shape[0]} 个个体信号")
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
