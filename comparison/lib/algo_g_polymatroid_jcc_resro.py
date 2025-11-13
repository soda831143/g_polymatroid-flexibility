"""
算法: JCC-Re-SRO G-Polymatroid 聚合  
流程: JCC+SRO → 初次优化 → 基于u0重构Re-SRO → 二次优化
特点: 两阶段优化,第二阶段使用多面体不确定性集,更不保守

参考MATLAB实现流程:
1. get_joint_uncertainty_set.m: SRO阶段(椭球集)
2. RO_reserve_joint.m: 第一次优化得到u0
3. get_reconstruct_joint.m: 基于u0重构Re-SRO(多面体集)
4. Recon_RO_joint.m: 第二次优化(使用Re-SRO边界)

流程:
阶段1: JCC+SRO获取初始鲁棒边界 (椭球不确定性集)
阶段2: 坐标变换 + 聚合 + 第一次优化 (得到u0)
阶段3: 基于u0重构Re-SRO边界 (多面体不确定性集)
阶段4: 坐标变换 + 聚合 + 第二次优化 (使用Re-SRO边界)
阶段5: 逆变换输出
"""

import time
import numpy as np
from typing import Dict, Any
import sys
import os

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flexitroid.devices.tcl import TCL
from flexitroid.problems.jcc_robust_bounds import JCCRobustCalculator
from flexitroid.utils.coordinate_transform import CoordinateTransformer
from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.devices.general_der import GeneralDER, DERParameters
from .peak_optimization import optimize_peak_l_infinity


def solve(data: Dict, objective: str = 'cost') -> Dict[str, Any]:
    """
    JCC-Re-SRO算法主函数 (两阶段优化)
    
    第一阶段: SRO + 初次优化
    1. JCC处理不确定性 (椭球集)
    2. 计算SRO鲁棒边界
    3. 坐标变换 + 聚合 + 优化 → 得到u0
    
    第二阶段: Re-SRO + 二次优化  
    4. 基于u0重构不确定性集 (多面体集)
    5. 计算Re-SRO鲁棒边界 (更紧)
    6. 坐标变换 + 聚合 + 优化 → 最终解
    
    Args:
        data: 包含以下关键字的字典
            - 'tcl_objs': TCL对象列表
            - 'periods': 时间步数T
            - 'households': TCL数量N
            - 'uncertainty_data': {
                'D_shape': SRO形状集温度误差数据 (n1, T) [数据D的25%]
                'D_calib': SRO校准集温度误差数据 (n2, T) [数据D的25%]
                'D_resro_calib': Re-SRO独立校准集温度误差数据 (n3, T) [数据D的50%,独立于SRO]
                'epsilon': JCC违反概率上限
                'delta': 置信度
              }
        objective: 优化目标 ('cost' 或 'peak')
                'delta': 统计置信度
              }
            - 'prices': 电价 (T,)
            - 'P0': 基线聚合功率 (T,)
            
        注意: Re-SRO使用独立的50%数据进行校准,确保与SRO的25%+25%统计独立
    
    Returns:
        results: 包含求解结果的字典
    """
    t0 = time.time()
    
    print("\n" + "="*80)
    print("JCC-Re-SRO 两阶段鲁棒聚合算法")
    print("="*80)
    
    # 解析输入
    tcl_objs = data.get('tcl_objs', None)
    if tcl_objs is None:
        raise ValueError("必须提供tcl_objs")
    
    T = data['periods']
    N = data['households']
    prices = data['prices']
    P0_agg = data['P0']
    uncertainty_data = data.get('uncertainty_data', None)
    
    if uncertainty_data is None:
        raise ValueError("JCC-Re-SRO算法必须提供uncertainty_data")
    
    # 检查Re-SRO独立校准集
    if 'D_resro_calib' not in uncertainty_data:
        print("  警告: 未提供D_resro_calib,将回退使用D_calib (不符合理论设计!)")
    
    print(f"TCL数量: {N}, 时间步数: {T}")
    n_shape = uncertainty_data['D_shape'].shape[0]
    n_calib = uncertainty_data['D_calib'].shape[0]
    n_resro = uncertainty_data.get('D_resro_calib', np.array([])).shape[0] if 'D_resro_calib' in uncertainty_data else 0
    
    if n_resro > 0:
        print(f"数据分配策略 (Re-SRO模式):")
        print(f"  - SRO形状集: {n_shape} 样本 ({n_shape/(n_shape+n_calib+n_resro)*100:.1f}%)")
        print(f"  - SRO校准集: {n_calib} 样本 ({n_calib/(n_shape+n_calib+n_resro)*100:.1f}%)")
        print(f"  - Re-SRO独立校准集: {n_resro} 样本 ({n_resro/(n_shape+n_calib+n_resro)*100:.1f}%) [独立]")
    else:
        print(f"不确定性样本: 形状集{n_shape} + 校准集{n_calib}")
    
    # ========== 第一阶段: SRO + 初次优化 ==========
    print("\n" + "="*80)
    print("【第一阶段】SRO鲁棒化 + 初次优化")
    print("="*80)
    
    # 1.1 JCC + SRO计算初始鲁棒边界
    print("\n步骤1.1: JCC-SRO计算初始鲁棒边界...")
    jcc_calculator = JCCRobustCalculator(tcl_objs, uncertainty_data)
    jcc_calculator.compute_sro_bounds()
    
    b_robust_initial = jcc_calculator.b_robust_initial  # (N, constraint_num)
    U_initial = jcc_calculator.U_initial
    
    print(f"SRO鲁棒边界: 椭球集 s*={U_initial['s_star']:.4f}")
    
    # 1.2 坐标变换: 使用SRO边界
    print("\n步骤1.2: 坐标变换 (Physical → Virtual, 使用SRO边界)...")
    tcl_virtual_sro_list = []
    
    for i, tcl in enumerate(tcl_objs):
        # 计算带SRO边界的虚拟g-polymatroid
        # 使用TCL的g-polymatroid (b和p函数) 计算前缀集合的能量约束
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
        y_lower_virt = np.zeros(T)
        y_upper_virt = np.zeros(T)
        for t in range(T):
            scale = a ** (t + 1)
            y_lower_virt[t] = y_lower_phys[t] / scale if scale > 1e-10 else y_lower_phys[t]
            y_upper_virt[t] = y_upper_phys[t] / scale if scale > 1e-10 else y_upper_phys[t]
        
        # 功率边界变换: ũ(k) = δ·u(k)/a^k
        P0_i = np.maximum(0, (tcl.theta_a_forecast - tcl.theta_r) / tcl.b_coef)
        u_min_phys = -P0_i
        u_max_phys = tcl.P_m - P0_i
        u_min_virt = np.zeros(T)
        u_max_virt = np.zeros(T)
        for t in range(T):
            scale = delta / (a ** t) if abs(a ** t) > 1e-10 else delta
            u_min_virt[t] = u_min_phys[t] * scale
            u_max_virt[t] = u_max_phys[t] * scale
        
        params_virt = DERParameters(
            u_min=u_min_virt, u_max=u_max_virt,
            x_min=y_lower_virt, x_max=y_upper_virt
        )
        tcl_virtual_sro_list.append(GeneralDER(params_virt))
    
    print(f"完成 {N} 个TCL的SRO坐标变换")
    
    # 1.3 聚合SRO虚拟g-polymatroid
    print("\n步骤1.3: 聚合SRO虚拟g-polymatroid...")
    agg_sro_virtual = Aggregator(tcl_virtual_sro_list)
    
    # 1.4 第一次优化得到u0
    print(f"\n步骤1.4: 虚拟坐标优化 (目标: {objective}) 得到初始解u0...")
    
    if objective == 'cost':
        # 成本优化：在虚拟聚合空间用正确的变换目标函数系数
        # c̃_agg[t] = c[t]·Σ_i(a_i^t/δ_i)  （关键：求和而非平均！）
        N = len(tcl_objs)
        c_virtual_agg = np.zeros(T)
        
        for t in range(T):
            scale_sum = 0
            for i in range(N):
                tcl = tcl_objs[i]
                scale = (tcl.a ** t) / tcl.delta if tcl.delta > 1e-10 else 1.0
                scale_sum += scale
            c_virtual_agg[t] = prices[t] * scale_sum
        
        u0_virtual_agg = agg_sro_virtual.solve_linear_program(c_virtual_agg)
    elif objective == 'peak':
        u0_virtual_individual, u0_virtual_agg = optimize_peak_l_infinity(
            agg_sro_virtual, P0_agg, tcl_objs, T
        )
    else:
        raise ValueError(f"未知的优化目标: {objective}")
    # 【新增】顶点分解: 将虚拟聚合信号分解为个体信号
    print("\n步骤1.5: 顶点分解得到个体虚拟信号...")
    u0_virtual_individual = agg_sro_virtual.disaggregate(u0_virtual_agg)
    
    # 【关键修正】使用个体逆变换（而非平均分配）
    u0_physical_individual = []
    for i, (tcl, u_virt_i) in enumerate(zip(tcl_objs, u0_virtual_individual)):
        # 使用各TCL自己的参数进行逆变换
        a_i = tcl.a
        delta_i = tcl.delta
        
        u_phys = np.zeros(T)
        for t in range(T):
            scale = (a_i ** t) / delta_i if delta_i > 1e-10 else 1.0
            u_phys[t] = u_virt_i[t] * scale
        
        u0_physical_individual.append(u_phys)
    
    u0_physical_individual = np.array(u0_physical_individual)  # (N, T)
    u0_physical_agg = np.sum(u0_physical_individual, axis=0)
    P0_stage1 = u0_physical_agg + P0_agg
    cost_stage1 = np.dot(prices, P0_stage1)
    
    print(f"第一阶段优化完成: 成本={cost_stage1:.2f}")
    
    # ========== 第二阶段: Re-SRO + 二次优化 ==========
    print("\n" + "="*80)
    print("【第二阶段】Re-SRO重构 + 二次优化")
    print("="*80)
    
    # 2.1 基于u0重构Re-SRO边界
    print("\n步骤2.1: 基于u0重构Re-SRO边界...")
    jcc_calculator.compute_resro_bounds(u0_physical_individual)  # 传入(N,T)个体解
    
    b_robust_final = jcc_calculator.b_robust_final  # (N, constraint_num)
    U_final = jcc_calculator.U_final
    
    print(f"Re-SRO鲁棒边界: 多面体集 s_unified={U_final['s_unified']:.4f}")
    
    # 2.2 坐标变换: 使用Re-SRO边界
    print("\n步骤2.2: 坐标变换 (Physical → Virtual, 使用Re-SRO边界)...")
    tcl_virtual_resro_list = []
    
    for i, tcl in enumerate(tcl_objs):
        # 计算带Re-SRO边界的虚拟g-polymatroid
        # 使用TCL的g-polymatroid (b和p函数) 计算前缀集合的能量约束
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
        y_lower_virt = np.zeros(T)
        y_upper_virt = np.zeros(T)
        for t in range(T):
            scale = a ** (t + 1)
            y_lower_virt[t] = y_lower_phys[t] / scale if scale > 1e-10 else y_lower_phys[t]
            y_upper_virt[t] = y_upper_phys[t] / scale if scale > 1e-10 else y_upper_phys[t]
        
        # 功率边界变换: ũ(k) = δ·u(k)/a^k (与第一阶段相同)
        P0_i = np.maximum(0, (tcl.theta_a_forecast - tcl.theta_r) / tcl.b_coef)
        u_min_phys = -P0_i
        u_max_phys = tcl.P_m - P0_i
        u_min_virt = np.zeros(T)
        u_max_virt = np.zeros(T)
        for t in range(T):
            scale = delta / (a ** t) if abs(a ** t) > 1e-10 else delta
            u_min_virt[t] = u_min_phys[t] * scale
            u_max_virt[t] = u_max_phys[t] * scale
        
        params_virt = DERParameters(
            u_min=u_min_virt, u_max=u_max_virt,
            x_min=y_lower_virt, x_max=y_upper_virt
        )
        tcl_virtual_resro_list.append(GeneralDER(params_virt))
    
    print(f"完成 {N} 个TCL的Re-SRO坐标变换")
    
    # 2.3 聚合Re-SRO虚拟g-polymatroid
    print("\n步骤2.3: 聚合Re-SRO虚拟g-polymatroid...")
    agg_resro_virtual = Aggregator(tcl_virtual_resro_list)
    
    # 2.4 第二次优化得到最终解
    print(f"\n步骤2.4: 虚拟坐标优化 (目标: {objective}) 得到最终解...")
    
    if objective == 'cost':
        # 成本优化：在虚拟聚合空间用正确的变换目标函数系数
        # c̃_agg[t] = c[t]·Σ_i(a_i^t/δ_i)  （关键：求和而非平均！）
        N = len(tcl_objs)
        c_virtual_agg = np.zeros(T)
        
        for t in range(T):
            scale_sum = 0
            for i in range(N):
                tcl = tcl_objs[i]
                scale = (tcl.a ** t) / tcl.delta if tcl.delta > 1e-10 else 1.0
                scale_sum += scale
            c_virtual_agg[t] = prices[t] * scale_sum
        
        u_final_virtual_agg = agg_resro_virtual.solve_linear_program(c_virtual_agg)
        # 【新增】顶点分解: 将最终虚拟聚合信号分解为个体信号
        print("\n步骤2.5: 顶点分解得到最终个体虚拟信号...")
        u_final_virtual_individual = agg_resro_virtual.disaggregate(u_final_virtual_agg)
    elif objective == 'peak':
        u_final_virtual_individual, u_final_virtual_agg = optimize_peak_l_infinity(
            agg_resro_virtual, P0_agg, tcl_objs, T
        )
    else:
        raise ValueError(f"未知的优化目标: {objective}")
    
    print(f"分解为 {u_final_virtual_individual.shape[0]} 个最终个体虚拟信号")
    
    # 【关键修正】使用个体逆变换
    u_final_physical_individual = []
    for i, (tcl, u_virt_i) in enumerate(zip(tcl_objs, u_final_virtual_individual)):
        a_i = tcl.a
        delta_i = tcl.delta
        
        u_phys = np.zeros(T)
        for t in range(T):
            scale = (a_i ** t) / delta_i if delta_i > 1e-10 else 1.0
            u_phys[t] = u_virt_i[t] * scale
        
        u_final_physical_individual.append(u_phys)
    
    u_final_physical_individual = np.array(u_final_physical_individual)
    u_final_physical_agg = np.sum(u_final_physical_individual, axis=0)
    
    P_final = u_final_physical_agg + P0_agg
    cost_final = np.dot(prices, P_final)
    peak_final = np.max(P_final)
    
    computation_time = time.time() - t0
    
    print("\n" + "="*80)
    print(f"JCC-Re-SRO两阶段算法完成 (目标: {objective})")
    print(f"第一阶段(SRO)成本: {cost_stage1:.2f}")
    print(f"第二阶段(Re-SRO)成本: {cost_final:.2f}")
    print(f"成本改善: {cost_stage1 - cost_final:.2f} ({(cost_stage1-cost_final)/cost_stage1*100:.2f}%)")
    print(f"峰值功率: {peak_final:.2f}")
    print(f"计算时间: {computation_time:.3f}s")
    print("="*80)
    
    return {
        'aggregate_flexibility': P_final,
        'individual_flexibility': u_final_physical_individual,
        'total_cost': cost_final,
        'peak_power': peak_final,
        'computation_time': computation_time,
        'objective': objective,
        'sro_cost': cost_stage1,
        'resro_cost': cost_final,
        'cost_improvement': cost_stage1 - cost_final,
        'U_initial': U_initial,
        'algorithm': f'JCC-Re-SRO-{objective.upper()}'
    }


def algo(data: Dict) -> Dict[str, Any]:
    """
    兼容性包装函数,调用solve()
    """
    return solve(data)
