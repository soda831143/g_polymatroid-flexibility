"""
算法: JCC-SRO G-Polymatroid 聚合
流程: JCC不确定性处理 → SRO鲁棒边界 → 坐标变换 → 聚合 → 优化
特点: 使用椭球不确定性集,较保守但计算高效
"""
import numpy as np
from typing import List, Dict, Tuple
import sys
import os
# 添加路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.utils.coordinate_transform import CoordinateTransformer
from flexitroid.problems.jcc_robust_bounds import JCCRobustCalculator
from flexitroid.devices.tcl import TCL
from flexitroid.devices.general_der import GeneralDER, DERParameters


def solve(data: dict, tcl_objs: List = None) -> dict:
    """
    JCC-SRO算法主函数
    
    流程:
    1. 使用JCC处理不确定性数据,构建椭球不确定性集
    2. 计算SRO鲁棒边界(每个TCL基于椭球集)
    3. 物理坐标变换到虚拟坐标(有损→无损)
    4. 聚合所有TCL的虚拟g-polymatroid
    5. 在虚拟坐标下优化
    6. 逆变换回物理坐标
    
    Args:
        data: 包含以下键的字典
            - 'prices': 电价 (T,)
            - 'P0': 基线功率 (T,) 
            - 'tcl_objs': TCL对象列表 (如果tcl_objs参数为None)
            - 'periods': 时间步数
            - 'households': TCL数量
            - 'uncertainty_data': {
                'D_shape': 形状集数据 (n1, T)
                'D_calib': 校准集数据 (n2, T)
                'epsilon': JCC违反概率
                'delta': 统计置信度
              }
        tcl_objs: TCL对象列表(可选)
    
    Returns:
        结果字典 {
            'aggregate_flexibility': 聚合后的物理功率序列,
            'total_cost': 总成本,
            'peak_power': 峰值功率,
            'computation_time': 计算时间,
            'sro_bounds': SRO鲁棒边界,
            'U_initial': 椭球不确定性集参数
        }
    """
    import time
    start_time = time.time()
    
    # 1. 解析输入数据
    prices = data['prices']
    P0_agg = data['P0']
    T = data['periods']
    N = data['households']
    uncertainty_data = data.get('uncertainty_data', None)
    
    if tcl_objs is None:
        tcl_objs = data.get('tcl_objs', None)
    
    if tcl_objs is None:
        raise ValueError("必须提供tcl_objs(TCL对象列表)")
    
    if uncertainty_data is None:
        raise ValueError("JCC-SRO算法必须提供uncertainty_data")
    
    print("\n" + "="*80)
    print("JCC-SRO G-Polymatroid 聚合算法")
    print("="*80)
    print(f"TCL数量: {N}, 时间步数: {T}")
    print(f"不确定性样本: 形状集 {uncertainty_data['D_shape'].shape[0]} + 校准集 {uncertainty_data['D_calib'].shape[0]}")
    
    # 2. JCC不确定性处理 + SRO鲁棒边界计算
    print("\n--- 阶段1: JCC不确定性处理与SRO鲁棒边界计算 ---")
    jcc_calculator = JCCRobustCalculator(tcl_objs, uncertainty_data)
    
    # 计算SRO阶段的鲁棒边界(基于椭球不确定性集)
    jcc_calculator.compute_sro_bounds()
    b_robust_sro = jcc_calculator.b_robust_initial  # (N, constraint_num)
    U_initial = jcc_calculator.U_initial
    
    print(f"SRO鲁棒边界计算完成: {b_robust_sro.shape}")
    print(f"椭球不确定性集: μ范围[{U_initial['mu'].min():.3f}, {U_initial['mu'].max():.3f}], s*={U_initial['s_star']:.4f}")
    
    # 3. 为每个TCL创建带SRO鲁棒边界的g-polymatroid
    print("\n--- 阶段2: 构建鲁棒g-polymatroid ---")
    tcl_robust_list = []
    for i, tcl in enumerate(tcl_objs):
        # 使用SRO鲁棒边界重新构建TCL的g-polymatroid
        tcl_params_robust = tcl.tcl_params.copy()
        tcl_params_robust['b_robust_override'] = b_robust_sro[i]
        
        # 创建新的TCL对象(带鲁棒边界)
        tcl_robust = TCL(
            tcl_params=tcl_params_robust,
            build_g_poly=True,
            theta_a_forecast=tcl.theta_a_forecast,
            use_provable_inner=True
        )
        tcl_robust_list.append(tcl_robust)
    
    print(f"创建了 {len(tcl_robust_list)} 个鲁棒TCL g-polymatroid")
    
    # 4. 坐标变换: Physical → Virtual
    print("\n--- 阶段3: 坐标变换 (Physical → Virtual) ---")
    tcl_virtual_list = []
    
    # 直接实现变换逻辑,不使用CoordinateTransformer类
    for i, tcl_robust in enumerate(tcl_robust_list):
        # 获取物理g-polymatroid的边界
        # 使用TCL的g-polymatroid (b和p函数) 计算前缀集合的能量约束
        y_lower_phys = np.zeros(T)
        y_upper_phys = np.zeros(T)
        
        for t in range(T):
            A_prefix = frozenset(range(t + 1))  # {0, 1, ..., t}
            try:
                y_lower_phys[t] = tcl_robust.p(A_prefix)  # 下界
                y_upper_phys[t] = tcl_robust.b(A_prefix)  # 上界
            except Exception:
                # 如果g-polymatroid计算失败，使用保守的边界
                P0_i = np.maximum(0, (tcl_robust.theta_a_forecast - tcl_robust.theta_r) / tcl_robust.b_coef)
                y_lower_phys[t] = np.sum(-P0_i[:t+1])  # 累积最小值
                y_upper_phys[t] = np.sum((tcl_robust.P_m - P0_i)[:t+1])  # 累积最大值
        
        # 变换到虚拟坐标 (简化实现: 假设变换主要影响功率边界)
        a = tcl_robust.a
        delta = tcl_robust.delta
        
        # 虚拟能量边界: x̃(k) = x(k) / a^k
        y_lower_virt = np.zeros(T)
        y_upper_virt = np.zeros(T)
        for t in range(T):
            scale = a ** (t + 1)
            y_lower_virt[t] = y_lower_phys[t] / scale if scale > 1e-10 else y_lower_phys[t]
            y_upper_virt[t] = y_upper_phys[t] / scale if scale > 1e-10 else y_upper_phys[t]
        
        # 计算虚拟坐标下的功率边界
        P0_i = (tcl_robust.theta_a_forecast - tcl_robust.theta_r) / tcl_robust.b_coef
        P0_i = np.maximum(0, P0_i)
        u_min_phys = -P0_i
        u_max_phys = tcl_robust.P_m - P0_i
        
        # 虚拟功率边界: ũ(k) = δ·u(k)/a^k
        u_min_virt = np.zeros(T)
        u_max_virt = np.zeros(T)
        for t in range(T):
            # ũ(k) = δ·u(k)/a^k → 变换系数 = δ/a^k
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
    
    # 5. 聚合虚拟g-polymatroid
    print("\n--- 阶段4: 聚合虚拟g-polymatroid ---")
    aggregator = Aggregator(tcl_virtual_list)
    agg_virtual = aggregator  # aggregator就是聚合后的对象
    print("虚拟坐标聚合完成")
    
    # 6. 虚拟坐标下优化
    print("\n--- 阶段5: 虚拟坐标优化 ---")
    # 虚拟坐标下的成本向量(价格不变)
    c_virtual = prices
    u0_virtual_agg = agg_virtual.solve_linear_program(c_virtual)
    print(f"虚拟坐标最优解范围: [{u0_virtual_agg.min():.3f}, {u0_virtual_agg.max():.3f}]")
    
    # 7. 逆变换到物理坐标
    print("\n--- 阶段6: 逆变换到物理坐标 ---")
    # 平均分配聚合解到各个TCL
    u0_virtual_individual = np.array([u0_virtual_agg / N for _ in range(N)])
    
    # 逆变换: u(k) = (a^k / δ) * ũ(k)
    u0_physical_individual = []
    for i, tcl_robust in enumerate(tcl_robust_list):
        a = tcl_robust.a
        delta = tcl_robust.delta
        u_phys = np.zeros(T)
        for t in range(T):
            scale = (a ** t) / delta if delta > 1e-10 else 1.0
            u_phys[t] = u0_virtual_individual[i][t] * scale
        u0_physical_individual.append(u_phys)
    
    u0_physical_individual = np.array(u0_physical_individual)
    
    # 聚合物理解
    u0_physical_agg = np.sum(u0_physical_individual, axis=0)
    
    # 调试: 检查形状
    print(f"\n[调试] u0_physical_agg 形状: {u0_physical_agg.shape}")
    print(f"[调试] P0_agg 形状: {P0_agg.shape}")
    print(f"[调试] P0_agg 范围: [{P0_agg.min():.3f}, {P0_agg.max():.3f}]")
    
    # 转换为实际功率
    P_total = u0_physical_agg + P0_agg
    
    # 8. 计算指标
    total_cost = np.dot(prices, P_total)
    peak_power = np.max(P_total)
    computation_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("JCC-SRO算法完成")
    print(f"总成本: {total_cost:.2f}")
    print(f"峰值功率: {peak_power:.2f}")
    print(f"计算时间: {computation_time:.3f}s")
    print("="*80)
    
    return {
        'aggregate_flexibility': P_total,
        'total_cost': total_cost,
        'peak_power': peak_power,
        'computation_time': computation_time,
        'sro_bounds': b_robust_sro,
        'U_initial': U_initial,
        'algorithm': 'JCC-SRO'
    }
