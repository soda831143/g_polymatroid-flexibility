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
from .peak_optimization import optimize_cost_column_generation, optimize_peak_column_generation


def solve(data: dict, tcl_objs: List = None, objective='cost') -> dict:
    """
    JCC-SRO算法主函数
    
    流程:
    1. 使用JCC处理不确定性数据,构建椭球不确定性集
    2. 计算SRO鲁棒边界(每个TCL基于椭球集)
    3. 物理坐标变换到虚拟坐标(有损→无损)
    4. 聚合所有TCL的虚拟g-polymatroid
    5. 在虚拟坐标下优化 (成本或峰值)
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
        objective: 优化目标 ('cost' 或 'peak')
    
    Returns:
        结果字典 {
            'aggregate_flexibility': 聚合后的物理功率序列,
            'total_cost': 总成本,
            'peak_power': 峰值功率,
            'computation_time': 计算时间,
            'objective': 优化目标,
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
    
    # 6. 虚拟坐标下优化 (根据目标选择算法)
    print(f"\n--- 阶段5: 虚拟坐标优化 (目标: {objective}) ---")
    
    if objective == 'cost':
        print("目标: 最小化成本 (使用列生成框架)")
        # 成本优化：使用列生成框架处理异构变换
        u0_physical_individual, u0_physical_agg, total_cost = optimize_cost_column_generation(
            agg_virtual, prices, P0_agg, tcl_objs, T
        )
        
        # 从物理信号反推虚拟信号（如果需要）
        N = len(tcl_objs)
        u0_virtual_individual = np.zeros((N, T))
        for i, tcl in enumerate(tcl_objs):
            a_i = tcl.a
            delta_i = tcl.delta
            for t in range(T):
                scale = (a_i ** t) / delta_i if delta_i > 1e-10 else 1.0
                u0_virtual_individual[i, t] = u0_physical_individual[i, t] / scale if abs(scale) > 1e-10 else 0.0
        
        u0_virtual_agg = np.sum(u0_virtual_individual, axis=0)
        
    elif objective == 'peak':
        print("目标: 最小化峰值 (使用列生成框架)")
        # 峰值优化: 使用列生成框架
        u0_physical_individual, u0_physical_agg, peak_power = optimize_peak_column_generation(
            agg_virtual, P0_agg, tcl_objs, T
        )
        
        # 反推虚拟信号
        N = len(tcl_objs)
        u0_virtual_individual = np.zeros((N, T))
        for i, tcl in enumerate(tcl_objs):
            a_i = tcl.a
            delta_i = tcl.delta
            for t in range(T):
                scale = (a_i ** t) / delta_i if delta_i > 1e-10 else 1.0
                u0_virtual_individual[i, t] = u0_physical_individual[i, t] / scale if abs(scale) > 1e-10 else 0.0
        
        u0_virtual_agg = np.sum(u0_virtual_individual, axis=0)
    else:
        raise ValueError(f"未知的优化目标: {objective}")
    
    # 7. 虚拟信号验证
    print("\n--- 阶段6: 虚拟信号验证 ---")
    print(f"虚拟个体信号数: {u0_virtual_individual.shape[0]}")
    print(f"各TCL虚拟信号范围:")
    for i in range(min(3, len(u0_virtual_individual))):
        u_i = u0_virtual_individual[i]
        print(f"  TCL {i}: [{u_i.min():.3f}, {u_i.max():.3f}]")
    if len(u0_virtual_individual) > 3:
        print(f"  ... (共{len(u0_virtual_individual)}个)")
    
    # 8. 转换为实际功率（物理个体信号已由列生成函数返回）
    print("\n--- 阶段7: 最终结果处理 ---")
    print(f"物理个体信号: {u0_physical_individual.shape}")
    print(f"物理聚合信号范围: [{u0_physical_agg.min():.3f}, {u0_physical_agg.max():.3f}]")
    
    # 转换为实际功率
    P_total = u0_physical_agg + P0_agg
    print(f"总功率范围: [{P_total.min():.3f}, {P_total.max():.3f}]")
    
    # 9. 计算指标
    if objective == 'cost':
        peak_power = np.max(P_total)
    else:
        # peak_power已由列生成函数返回
        pass
    
    total_cost = np.dot(prices, P_total)
    computation_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"JCC-SRO算法完成 (目标: {objective})")
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
        'sro_bounds': b_robust_sro,
        'U_initial': U_initial,
        'algorithm': f'JCC-SRO-{objective.upper()}'
    }
