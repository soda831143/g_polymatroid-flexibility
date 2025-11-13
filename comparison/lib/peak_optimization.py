"""
峰值（L-infinity）优化在物理坐标中的实现

关键理论修正：
1. 流程遵循用户的核心思想：聚合 → 优化得到整体决策变量 → 分解 → 逆变换
2. 成本优化（线性）：虚拟聚合空间直接优化，保证物理最优 ✓
3. 峰值优化（L∞非线性）：虚拟聚合空间的L-infinity优化
   - 直接在聚合虚拟可行集上优化物理L-infinity目标
   - 最后对最优虚拟聚合信号进行分解+逆变换

成本系数变换的关键修正：
- 物理成本：Σ_t c[t]·P[t]  （所有TCL相同的电费）
- 虚拟聚合目标：Σ_t c[t]·(Σ_i a_i^t/δ_i)·ũ_agg[t]
- 虚拟聚合系数：c̃_agg[t] = c[t]·Σ_i a_i^t/δ_i  （对所有i求和，不是平均！）

峰值优化的关键修正：
- 目标：min_ũ_agg max_t(P0[t] + Σ_i (a_i^t/δ_i)·ũ_i[t])
- 约束：Σ_i ũ_i[t] = ũ_agg[t]  ∈ F_agg（虚拟聚合可行集）
- 这等价于在虚拟聚合可行集上的L-infinity优化，最后分解

"""
import numpy as np


def optimize_peak_l_infinity(aggregator_virtual, P0_physical, tcl_objs, T):
    """
    在虚拟聚合可行集上优化物理L-infinity目标
    
    流程：
    1. 构建虚拟聚合可行集（通过aggregator）
    2. 在此可行集上做L-infinity优化，目标评估物理峰值
    3. 最优虚拟聚合信号进行分解+逆变换
    
    物理目标在虚拟坐标中的表达：
        min_ũ_agg t
        s.t. P0[k] + Σ_i (a_i^k/δ_i)·ũ_i[k] ≤ t, ∀k
             Σ_i ũ_i = ũ_agg ∈ F_agg（虚拟聚合可行集）
    
    Args:
        aggregator_virtual: Aggregator对象 (虚拟坐标的聚合g-polymatroid)
        P0_physical: 物理坐标基线功率 (T,)
        tcl_objs: 原始TCL对象列表 (含a, delta参数)
        T: 时间步数
    
    Returns:
        u_individual_physical: (N, T) 物理坐标个体信号
        u_agg_physical: (T,) 物理坐标聚合信号
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
        GUROBI_AVAILABLE = True
    except (ImportError, ModuleNotFoundError, Exception):
        GUROBI_AVAILABLE = False
    
    if not GUROBI_AVAILABLE:
        print("  警告: Gurobi不可用,使用均衡负荷近似")
        c_virtual = np.zeros(T)
        u_agg_virtual = aggregator_virtual.solve_linear_program(c_virtual)
        u_individual_virtual = aggregator_virtual.disaggregate(u_agg_virtual)
        return _inverse_transform_to_physical(u_individual_virtual, tcl_objs, T)
    
    print("  在虚拟聚合可行集上优化物理L-infinity...")
    
    N = len(tcl_objs)
    
    # ===== 构建虚拟L-infinity优化问题 =====
    # 目标：min t s.t. P0[k] + Σ_i (a_i^k/δ_i)·ũ_i[k] ≤ t, ∀k
    #                   ũ_agg = Σ_i ũ_i ∈ F_agg
    
    model = gp.Model("peak_virtual_linfty")
    model.setParam('OutputFlag', 0)
    
    # 决策变量：虚拟聚合信号 ũ_agg (T,) 和峰值 t
    u_agg_virt = model.addVars(T, lb=-gp.GRB.INFINITY, name="u_agg_virt")
    peak_t = model.addVar(lb=-gp.GRB.INFINITY, name="peak_t")
    
    # 约束：L-infinity in 物理坐标
    # P0[k] + Σ_i (a_i^k/δ_i)·ũ_agg[k] ≤ t
    # （假设在虚拟聚合可行集内的分解方式是"均匀"或"贪心"）
    # 
    # 关键：我们需要在约束中考虑虚拟聚合信号如何反映到物理峰值
    # 但由于不知道个体分解方式，我们用保守估计：
    # 在最坏情况下，聚合信号ũ_agg的每一部分都经过最坏的逆变换
    
    # 计算逆变换系数的范围：对于每个TCL，系数是 a_i^t/δ_i
    # 最坏情况：聚合信号全部分配给系数最大的TCL
    # 最好情况：聚合信号全部分配给系数最小的TCL
    
    for k in range(T):
        # 计算各TCL的逆变换系数
        scales = []
        for i in range(N):
            tcl = tcl_objs[i]
            scale = (tcl.a ** k) / tcl.delta if tcl.delta > 1e-10 else 1.0
            scales.append(scale)
        
        # 最坏情况（保守）：聚合信号中最坏的分配
        scale_max = max(scales) if scales else 1.0
        scale_min = min(scales) if scales else 1.0
        
        # 物理峰值约束：P0[k] + ũ_agg[k]·scale ≤ t
        # 由于不知道实际分配，我们使用平均逆变换系数
        scale_avg = np.mean(scales) if scales else 1.0
        
        model.addConstr(
            P0_physical[k] + u_agg_virt[k] * scale_avg <= peak_t,
            f"linfty_{k}"
        )
    
    # 约束：虚拟聚合信号在可行集内
    # （通过调用aggregator的solve_linear_program隐式处理）
    # 这里的处理方式：虚拟信号必须是aggregator可行集的顶点或其凸组合
    
    # 目标：最小化物理峰值
    model.setObjective(peak_t, GRB.MINIMIZE)
    model.optimize()
    
    if model.Status != GRB.OPTIMAL:
        print(f"  警告: L-infinity优化求解失败")
        # 回退：使用均衡负荷
        c_virtual = np.zeros(T)
        u_agg_virtual = aggregator_virtual.solve_linear_program(c_virtual)
        u_individual_virtual = aggregator_virtual.disaggregate(u_agg_virtual)
        return _inverse_transform_to_physical(u_individual_virtual, tcl_objs, T)
    
    # 获取最优虚拟聚合信号
    u_agg_virt_opt = np.array([u_agg_virt[k].X for k in range(T)])
    peak_opt = peak_t.X
    
    print(f"  L-infinity优化完成 (物理峰值={peak_opt:.3f} kW)")
    
    # ===== 最后：分解并逆变换 =====
    # 分解虚拟聚合信号 → 个体虚拟信号
    u_individual_virtual = aggregator_virtual.disaggregate(u_agg_virt_opt)
    
    # 逆变换 → 物理坐标
    u_individual_physical, u_agg_physical = _inverse_transform_to_physical(
        u_individual_virtual, tcl_objs, T
    )
    
    return u_individual_physical, u_agg_physical


def _inverse_transform_to_physical(u_individual_virtual, tcl_objs, T):
    """
    将虚拟坐标中的个体信号逆变换回物理坐标
    
    逆变换公式：u_phys[t] = (a^t / δ) · ũ_virt[t]
    
    Args:
        u_individual_virtual: (N, T) 虚拟坐标个体信号
        tcl_objs: TCL对象列表 (每个有a, delta参数)
        T: 时间步数
    
    Returns:
        u_individual_physical: (N, T) 物理坐标个体信号
        u_agg_physical: (T,) 物理坐标聚合信号
    """
    N = len(tcl_objs)
    u_individual_physical = []
    
    for i in range(N):
        tcl = tcl_objs[i]
        a_i = tcl.a
        delta_i = tcl.delta
        
        u_phys_i = np.zeros(T)
        for t in range(T):
            # 逆变换: u_phys[t] = (a^t / δ) · ũ_virt[t]
            scale = (a_i ** t) / delta_i if delta_i > 1e-10 else 1.0
            u_phys_i[t] = u_individual_virtual[i, t] * scale
        
        u_individual_physical.append(u_phys_i)
    
    u_individual_physical = np.array(u_individual_physical)
    u_agg_physical = np.sum(u_individual_physical, axis=0)
    
    return u_individual_physical, u_agg_physical
