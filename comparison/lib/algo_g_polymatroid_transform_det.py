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
            # 【修正】使用a^(t+1)与正向变换一致
            time_index = t + 1
            scale = (a_i ** time_index) / delta_i if delta_i > 1e-10 else 1.0
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
            # 【修正】使用a^(t+1)与正向变换一致
            time_index = t + 1
            scale = (a_i ** time_index) / delta_i if delta_i > 1e-10 else 1.0
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
        
        # 虚拟功率边界: ũ(k) = δ·u(k)/a^k (TEX Eq. 4)
        # 
        # 【关键】时间索引约定（方案B，与虚拟状态边界一致）:
        # - TEX公式: k ∈ {1, 2, ..., T}
        # - Python: t ∈ {0, 1, ..., T-1}
        # - 映射: t对应TEX中的k=t+1
        # 
        # 因此: ũ[t] = δ·u[t] / a^(t+1)
        # 
        # 这与虚拟状态约束的索引一致：
        # - x̃[t+1] = x[t+1] / a^(t+1)
        # - x̃[t+1] = x̃[t] + ũ[t]
        u_min_virt = np.zeros(T)
        u_max_virt = np.zeros(T)
        for t in range(T):
            time_index = t + 1  # TEX中的k
            scale = delta / (a ** time_index) if abs(a ** time_index) > 1e-10 else delta
            u_min_virt[t] = u_min_phys[t] * scale
            u_max_virt[t] = u_max_phys[t] * scale
        
        # 【关键修正】虚拟状态约束 - 严格遵循TEX文件公式
        # 
        # === 物理模型 (TEX Eq. 1) ===
        # x(k) = a·x(k-1) + δ·u(k)
        # 展开: x(t) = a^t·x(0) + δ·Σ_{k=1}^{t} a^(t-k)·u(k)
        # 
        # === 虚拟坐标变换 (TEX Eq. 3, 4) ===
        # x̃(t) := x(t) / a^t
        # ũ(k) := δ·u(k) / a^k
        # 
        # === 虚拟动力学 (TEX Eq. 5) ===
        # x̃(t) = x̃(t-1) + ũ(t)  (无损系统!)
        # 展开: x̃(t) = x̃(0) + Σ_{k=1}^{t} ũ(k)
        # 
        # === 虚拟状态约束 (TEX Section 3.1) ===
        # 物理约束: -x_plus <= x(t) <= x_plus
        # 除以a^t: -x_plus/a^t <= x̃(t) <= x_plus/a^t
        # 代入展开式: -x_plus/a^t <= x̃(0) + Σ_{k=1}^{t} ũ(k) <= x_plus/a^t
        # 移项: -x_plus/a^t - x̃(0) <= Σ_{k=1}^{t} ũ(k) <= x_plus/a^t - x̃(0)
        # 
        # === GeneralDER约定 ===
        # x_min[t] 和 x_max[t] 表示累积和 Σ_{k=1}^{t+1} ũ(k) 的边界
        # 即：x_min[t] <= Σ_{k=1}^{t+1} ũ(k) <= x_max[t]
        # 这对应虚拟状态 x̃(t+1) - x̃(0)
        # 
        # 因此正确的映射是:
        # x_min[t] = -x_plus/a^(t+1) - x̃(0)
        # x_max[t] = x_plus/a^(t+1) - x̃(0)
        # 
        # 注意: x̃(0) = x(0) = x0 (初始状态)
        
        y_lower_virt = np.zeros(T)
        y_upper_virt = np.zeros(T)
        
        for t in range(T):
            # t 从 0 到 T-1
            # x_min[t] 对应 Σ_{k=1}^{t+1} ũ(k)，即虚拟状态 x̃(t+1)
            # 物理约束在时刻 t+1: -x_plus <= x(t+1) <= x_plus
            # 虚拟形式: -x_plus/a^(t+1) <= x̃(t+1) <= x_plus/a^(t+1)
            
            time_index = t + 1  # 对应物理时刻
            power_denom = a ** time_index if abs(a ** time_index) > 1e-10 else 1e-10
            
            # TEX公式: x̃_lower(t) = x_lower(t)/a^t - x̃(0)
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
            print(f"  [DEBUG] TCL 0 物理参数: a={a:.4f}, delta={delta:.4f}, x0={x0:.4f}, x_plus={x_plus:.2f}")
            print(f"  [DEBUG] TCL 0 物理功率范围: u_min_phys=[{u_min_phys.min():.3f}, {u_min_phys.max():.3f}], u_max_phys=[{u_max_phys.min():.3f}, {u_max_phys.max():.3f}]")
            print(f"  [DEBUG] TCL 0 物理状态约束: x ∈ [{-x_plus:.2f}, {x_plus:.2f}]")
        
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
            # 【修正】逆变换: u[t] = (a^(t+1) / δ) · ũ[t]
            # 与正向变换一致: ũ[t] = δ·u[t] / a^(t+1)
            time_index = t + 1
            scale = (a_i ** time_index) / delta_i if delta_i > 1e-10 else 1.0
            u_phys_i[t] = u_virt_i[t] * scale
        
        u0_physical_individual.append(u_phys_i)
    
    u0_physical_individual = np.array(u0_physical_individual)
    print(f"逆变换完成: {u0_physical_individual.shape}")
    
    # 聚合物理信号
    u0_physical_agg = np.sum(u0_physical_individual, axis=0)
    
    print(f"使用个体逆变换 (每个TCL使用自己的a_i, delta_i参数)")
    
    # 转换为实际功率
    P_total = u0_physical_agg + P0_agg
    
    # 【关键验证】检查每个TCL的物理约束是否满足
    print("\n--- 阶段7: 物理约束验证 ---")
    all_constraints_satisfied = True
    constraint_violations = []
    
    # 从data中获取个体基线功率（如果有的话）
    P0_individual = data.get('demands', None)  # (T, N) 格式
    if P0_individual is not None:
        P0_individual = P0_individual.T  # 转换为 (N, T) 格式
    
    for i, tcl in enumerate(tcl_objs):
        u_phys_i = u0_physical_individual[i]  # 物理控制信号
        
        # 获取个体基线功率
        if P0_individual is not None:
            P0_i = P0_individual[i]
        else:
            # 如果没有提供，根据TCL参数重新计算
            theta_a = tcl.tcl_params.get('theta_a_forecast', data.get('theta_a_forecast', np.zeros(T)))
            theta_r = tcl.tcl_params['theta_r']
            b_coef = tcl.tcl_params['R_th'] * tcl.tcl_params['eta']
            P0_i = np.maximum(0, (theta_a - theta_r) / b_coef)
        
        P_i = u_phys_i + P0_i  # 总功率
        
        # 获取TCL参数
        params = tcl.tcl_params
        a = params['a']
        delta = params['delta']
        P_m = params['P_m']
        P_min = params.get('P_min', 0.0)
        x0 = params.get('x0', 0.0)
        
        # 计算物理状态边界
        x_plus = (params['C_th'] * params['delta_val']) / params['eta'] if params['eta'] > 0 else 0
        x_min_phys = -x_plus
        x_max_phys = x_plus
        
        # 1. 功率边界约束：P_min <= P_i[t] <= P_m
        power_min_violations = np.sum(P_i < P_min - 1e-6)
        power_max_violations = np.sum(P_i > P_m + 1e-6)
        
        # 2. 状态约束：通过动力学方程计算状态轨迹
        x_trajectory = np.zeros(T + 1)
        x_trajectory[0] = x0
        state_violations = 0
        
        for t in range(T):
            x_trajectory[t + 1] = a * x_trajectory[t] + delta * u_phys_i[t]
            if x_trajectory[t + 1] < x_min_phys - 1e-6 or x_trajectory[t + 1] > x_max_phys + 1e-6:
                state_violations += 1
        
        # 统计违反情况
        total_violations = power_min_violations + power_max_violations + state_violations
        
        if total_violations > 0:
            all_constraints_satisfied = False
            violation_info = {
                'tcl_id': i,
                'power_min_violations': int(power_min_violations),
                'power_max_violations': int(power_max_violations),
                'state_violations': int(state_violations),
                'P_i_min': float(P_i.min()),
                'P_i_max': float(P_i.max()),
                'P_min': float(P_min),
                'P_m': float(P_m),
                'x_min': float(x_trajectory.min()),
                'x_max': float(x_trajectory.max()),
                'x_min_phys': float(x_min_phys),
                'x_max_phys': float(x_max_phys)
            }
            constraint_violations.append(violation_info)
    
    # 输出验证结果
    if all_constraints_satisfied:
        print(f"[OK] 所有 {N} 个TCL的物理约束均满足")
    else:
        print(f"[FAIL] 发现约束违反！{len(constraint_violations)}/{N} 个TCL违反约束")
        for violation in constraint_violations[:5]:  # 只显示前5个
            print(f"  TCL {violation['tcl_id']}:")
            if violation['power_min_violations'] > 0:
                print(f"    功率下界违反: {violation['power_min_violations']}次, P_i_min={violation['P_i_min']:.4f} < P_min={violation['P_min']:.4f}")
            if violation['power_max_violations'] > 0:
                print(f"    功率上界违反: {violation['power_max_violations']}次, P_i_max={violation['P_i_max']:.4f} > P_m={violation['P_m']:.4f}")
            if violation['state_violations'] > 0:
                print(f"    状态约束违反: {violation['state_violations']}次, x范围=[{violation['x_min']:.4f}, {violation['x_max']:.4f}], 限制=[{violation['x_min_phys']:.4f}, {violation['x_max_phys']:.4f}]")
        if len(constraint_violations) > 5:
            print(f"  ... 还有 {len(constraint_violations) - 5} 个TCL违反约束")
    
    # 8. 计算指标
    total_cost = np.dot(prices, P_total)
    # 【关键修正】峰值功率应该使用L-infinity范数（绝对值的最大值），与Exact和No-Flex一致
    peak_power = np.linalg.norm(P_total, ord=np.inf)  # max(abs(P_total))
    computation_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"确定性坐标变换算法完成 (目标: {objective})")
    print(f"总成本: {total_cost:.2f}")
    print(f"峰值功率: {peak_power:.2f}")
    print(f"约束满足: {'是' if all_constraints_satisfied else '否 (不可行解!)'}")
    print(f"计算时间: {computation_time:.3f}s")
    print("="*80)
    
    return {
        'aggregate_flexibility': P_total,
        'individual_flexibility': u0_physical_individual,
        'total_cost': total_cost,
        'peak_power': peak_power,
        'computation_time': computation_time,
        'objective': objective,
        'algorithm': f'G-Polymatroid-Transform-Det-{objective.upper()}',
        'constraints_satisfied': all_constraints_satisfied,
        'constraint_violations': constraint_violations
    }


def algo(data: dict) -> dict:
    """
    兼容性包装函数
    """
    return solve(data)
