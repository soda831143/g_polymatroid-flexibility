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


def solve(data: dict, tcl_objs: List = None) -> dict:
    """
    确定性坐标变换G-Polymatroid聚合算法
    
    流程:
    1. 获取名义物理边界 (无鲁棒化)
    2. 物理坐标变换到虚拟坐标 (有损→无损)
    3. 聚合所有TCL的虚拟g-polymatroid
    4. 在虚拟坐标下优化
    5. 逆变换回物理坐标
    
    Args:
        data: 包含以下键的字典
            - 'prices': 电价 (T,)
            - 'P0': 基线功率 (T,) 
            - 'tcl_objs': TCL对象列表
            - 'periods': 时间步数
            - 'households': TCL数量
        tcl_objs: TCL对象列表(可选)
    
    Returns:
        结果字典 {
            'aggregate_flexibility': 聚合后的物理功率序列,
            'total_cost': 总成本,
            'peak_power': 峰值功率,
            'computation_time': 计算时间
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
        # 获取物理g-polymatroid的名义边界
        y_lower_phys = np.zeros(T)
        y_upper_phys = np.zeros(T)
        
        for t in range(T):
            A_prefix = frozenset(range(t + 1))  # {0, 1, ..., t}
            try:
                y_lower_phys[t] = tcl.p(A_prefix)  # 下界
                y_upper_phys[t] = tcl.b(A_prefix)  # 上界
            except Exception:
                # 如果g-polymatroid计算失败，使用保守的边界
                P0_i = np.maximum(0, (tcl.theta_a_forecast - tcl.theta_r) / tcl.b_coef)
                y_lower_phys[t] = np.sum(-P0_i[:t+1])  # 累积最小值
                y_upper_phys[t] = np.sum((tcl.P_m - P0_i)[:t+1])  # 累积最大值
        
        # 变换到虚拟坐标
        a = tcl.a
        delta = tcl.delta
        
        # 虚拟能量边界: x̃(k) = x(k) / a^(k+1)
        y_lower_virt = np.zeros(T)
        y_upper_virt = np.zeros(T)
        for t in range(T):
            scale = a ** (t + 1)
            y_lower_virt[t] = y_lower_phys[t] / scale if scale > 1e-10 else y_lower_phys[t]
            y_upper_virt[t] = y_upper_phys[t] / scale if scale > 1e-10 else y_upper_phys[t]
        
        # 计算虚拟坐标下的功率边界
        P0_i = (tcl.theta_a_forecast - tcl.theta_r) / tcl.b_coef
        P0_i = np.maximum(0, P0_i)
        u_min_phys = -P0_i
        u_max_phys = tcl.P_m - P0_i
        
        # 虚拟功率边界: ũ(k) = δ·u(k)/a^k
        u_min_virt = np.zeros(T)
        u_max_virt = np.zeros(T)
        for t in range(T):
            scale = delta / (a ** t) if abs(a ** t) > 1e-10 else delta
            u_min_virt[t] = u_min_phys[t] * scale
            u_max_virt[t] = u_max_phys[t] * scale
        
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
    
    # 5. 虚拟坐标下优化
    print("\n--- 阶段4: 虚拟坐标优化 ---")
    c_virtual = prices
    u0_virtual_agg = agg_virtual.solve_linear_program(c_virtual)
    print(f"虚拟坐标最优解范围: [{u0_virtual_agg.min():.3f}, {u0_virtual_agg.max():.3f}]")
    
    # 6. 逆变换到物理坐标
    print("\n--- 阶段5: 逆变换到物理坐标 ---")
    # 平均分配聚合解到各个TCL
    u0_virtual_individual = np.array([u0_virtual_agg / N for _ in range(N)])
    
    # 逆变换: u(k) = (a^k / δ) * ũ(k)
    u0_physical_individual = []
    for i, tcl in enumerate(tcl_objs):
        a = tcl.a
        delta = tcl.delta
        u_phys = np.zeros(T)
        for t in range(T):
            scale = (a ** t) / delta if delta > 1e-10 else 1.0
            u_phys[t] = u0_virtual_individual[i][t] * scale
        u0_physical_individual.append(u_phys)
    
    u0_physical_individual = np.array(u0_physical_individual)
    
    # 聚合物理解
    u0_physical_agg = np.sum(u0_physical_individual, axis=0)
    
    # 转换为实际功率
    P_total = u0_physical_agg + P0_agg
    
    # 7. 计算指标
    total_cost = np.dot(prices, P_total)
    peak_power = np.max(P_total)
    computation_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("确定性坐标变换算法完成")
    print(f"总成本: {total_cost:.2f}")
    print(f"峰值功率: {peak_power:.2f}")
    print(f"计算时间: {computation_time:.3f}s")
    print("="*80)
    
    return {
        'aggregate_flexibility': P_total,
        'total_cost': total_cost,
        'peak_power': peak_power,
        'computation_time': computation_time,
        'algorithm': 'G-Polymatroid-Transform-Det'
    }


def algo(data: dict) -> dict:
    """
    兼容性包装函数
    """
    return solve(data)
