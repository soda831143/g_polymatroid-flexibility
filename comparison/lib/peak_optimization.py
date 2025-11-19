"""
峰值（L-infinity）优化的列生成框架实现

【核心理论修正】
===================================================
问题：异构坐标变换导致的目标函数"耦合"

物理聚合负载：P_agg[t] = P_0[t] + Σ_i (γ_i[t] · ũ_i[t])
其中 γ_i[t] = a_i^t / δ_i 是异构变换系数（每个TCL不同）

虚拟可行集约束：ũ_agg = Σ_i ũ_i ∈ F_agg

【问题所在】
不能简单地在F_agg上优化ũ_agg，因为：
1. 物理目标涉及Σ_i (γ_i · ũ_i)，而可行集约束的是Σ_i ũ_i
2. 同一个ũ_agg可以有多种个体分解方式，产生完全不同的物理目标值
3. 因此不存在只关于ũ_agg的"虚拟目标函数"

【正确方法】: 列生成 (Column Generation) / Dantzig-Wolfe分解

主问题（Master Problem）：
    min_λ  J(λ)  （物理目标作为λ的函数）
    s.t.   ũ_agg = Σ_j λ_j v_j   （v_j是F_agg的顶点）
           Σ_j λ_j = 1
           λ_j ≥ 0

子问题（Subproblem）：
    给定对偶变量π，生成新顶点 v_new = argmin π^T v, s.t. v ∈ F_agg
    这可以通过贪心算法（greedy algorithm）高效求解

【流程】
1. 初始化：用一个初始顶点（如v_0 = F_agg的贪心顶点）
2. 迭代：
   - 用当前顶点集合求解主问题 → 最优λ和对偶变量π
   - 用π调用子问题（贪心算法）→ 检查是否可以改善
   - 若改善，加入新顶点，继续迭代
   - 否则收敛，停止
3. 最终：通过最优λ得到最优ũ_agg，分解+逆变换得到物理坐标结果

"""
import os
import time
import numpy as np
import concurrent.futures as cf


def optimize_cost_column_generation(aggregator_virtual, prices, P0_physical, tcl_objs, T, max_iterations=200, tolerance=1e-2):
    """
    使用列生成算法优化物理成本
    
    主问题：
        min_λ  Σ_t c[t] · (P_0[t] + Σ_j λ_j Σ_i γ_i[t] v_ij[t])
        s.t.   Σ_j λ_j = 1
               λ_j ≥ 0
    
    子问题：给定对偶变量π，求解
        min  π^T v, s.t. v ∈ F_agg （通过贪心算法求解）
    
    Args:
        aggregator_virtual: Aggregator对象
        prices: 分时电价 (T,)
        P0_physical: 物理基线功率 (T,)
        tcl_objs: TCL对象列表
        T: 时间步数
        max_iterations: 最大迭代次数
        tolerance: 对偶间隙容差
    
    Returns:
        u_individual_physical: (N, T) 物理坐标个体信号
        u_agg_physical: (T,) 物理坐标聚合信号
        total_cost: 优化后的物理成本
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
        GUROBI_AVAILABLE = True
    except (ImportError, ModuleNotFoundError, Exception):
        GUROBI_AVAILABLE = False
    
    if not GUROBI_AVAILABLE:
        print("  警告: Gurobi不可用,使用初始顶点近似")
        c_virtual = np.zeros(T)
        u_agg_virtual = aggregator_virtual.solve_linear_program(c_virtual)
        u_individual_virtual = aggregator_virtual.disaggregate(u_agg_virtual)
        u_individual_physical, u_agg_physical = _inverse_transform_to_physical(
            u_individual_virtual, tcl_objs, T
        )
        total_cost = np.sum(prices * (P0_physical + u_agg_physical))
        return u_individual_physical, u_agg_physical, total_cost
    
    print("  使用列生成算法优化成本...")
    
    N = len(tcl_objs)
    
    # ===== 步骤1：初始化 - 生成初始顶点 =====
    vertices_virtual = []  # 存储虚拟聚合顶点 v_j
    vertices_individual = []  # 存储个体顶点分解 v_ij (关键!)
    vertices_physical = []  # 存储对应的物理聚合信号 Σ_i γ_i · v_ij
    
    # 【修正】初始顶点：使用零价格向量生成
    # 这确保初始顶点不是针对特定目标的最优解，允许列生成过程探索
    v_init_individual = np.zeros((N, T))
    for i, device in enumerate(aggregator_virtual.fleet):
        # 使用零价格向量生成初始顶点
        c_i_virtual = np.zeros(T)
        v_init_individual[i] = device.solve_linear_program(c_i_virtual)
    vertices_individual.append(v_init_individual)
    
    # 聚合得到初始虚拟顶点
    v_init = np.sum(v_init_individual, axis=0)
    vertices_virtual.append(v_init)
    
    # 【修正】使用真实的个体分解计算物理信号
    v_phys_init = np.zeros(T)
    for i in range(N):
        tcl = tcl_objs[i]
        for t in range(T):
            # 【关键修正】使用a^(t+1)与逆变换一致
            time_index = t + 1
            gamma_it = (tcl.a ** time_index) / tcl.delta
            v_phys_init[t] += gamma_it * v_init_individual[i, t]
    vertices_physical.append(v_phys_init)
    
    cost_init = np.sum(prices * (P0_physical + v_phys_init))
    print(f"  初始顶点生成完毕")
    print(f"  [DEBUG] 初始虚拟聚合信号范围: [{v_init.min():.3f}, {v_init.max():.3f}]")
    print(f"  [DEBUG] 初始物理聚合信号范围: [{v_phys_init.min():.3f}, {v_phys_init.max():.3f}]")
    print(f"  [DEBUG] 初始物理成本={cost_init:.3f}")
    
    # ===== 步骤2：列生成迭代 =====
    iteration = 0
    best_cost = np.inf
    
    while iteration < max_iterations:
        iteration += 1
        
        # --- 2.1: 求解主问题 ---
        master = gp.Model("cost_master")
        master.setParam('OutputFlag', 0)
        
        num_vertices = len(vertices_virtual)
        lambda_vars = master.addVars(num_vertices, lb=0.0, name="lambda")
        
        # 目标：最小化物理成本
        obj_expr = 0.0
        for j, v_phys in enumerate(vertices_physical):
            # 这个顶点对应的物理聚合负载：P_agg = P_0 + v_phys
            physical_load = P0_physical + v_phys
            cost_j = np.sum(prices * physical_load)
            obj_expr += lambda_vars[j] * cost_j
        
        master.setObjective(obj_expr, GRB.MINIMIZE)
        
        # 约束：λ之和为1
        master.addConstr(gp.quicksum(lambda_vars[j] for j in range(num_vertices)) == 1.0, "convex_comb")
        
        master.optimize()
        
        if master.Status != GRB.OPTIMAL:
            print(f"  迭代{iteration}: 主问题求解失败")
            break
        
        best_cost = master.ObjVal
        lambda_opt = np.array([lambda_vars[j].X for j in range(num_vertices)])
        
        convex_constr = master.getConstrByName("convex_comb")
        mu = convex_constr.Pi if convex_constr is not None else 0.0
        
        print(f"  迭代{iteration}: 目标值={best_cost:.3f}, 顶点数={num_vertices}, μ={mu:.6f}")
        
        # --- 2.2: 求解子问题 ---
        # 【正确的Reduced Cost计算】
        # 主问题: min Σ_j λ_j · Cost_j, s.t. Σ_j λ_j = 1
        # 其中: Cost_j = Σ_t c[t] · (P0[t] + Σ_i γ_i[t]·v_ij[t])
        # 
        # 对偶变量π对应约束 Σ_j λ_j = 1
        # 
        # 新顶点的Reduced Cost:
        # RC = Cost_new - π = Σ_t c[t]·Σ_i γ_i[t]·v_i,new[t] - π
        # 
        # 子问题: min RC ⇔ min Σ_i Σ_t (c[t]·γ_i[t])·v_i[t]
        # 
        # 对每个设备i: min Σ_t (c[t]·γ_i[t])·v_i[t]
        # 虚拟坐标下: min Σ_t (c[t]·γ_i[t])·ṽ_i[t]
        
        # 【关键修正】为每个设备独立生成顶点,使用正确的价格向量
        v_new_individual = np.zeros((N, T))
        for i, device in enumerate(aggregator_virtual.fleet):
            tcl = tcl_objs[i]
            # 设备i在虚拟坐标下的价格向量
            gamma_i = np.array([(tcl.a ** t) / tcl.delta for t in range(T)])
            c_i_virtual = gamma_i * prices  # 物理价格·逆变换系数
            
            # 【调试】第一次迭代时输出
            if iteration == 1 and i == 0:
                print(f"    [DEBUG] 设备0的价格向量: gamma_0·c_phys")
                print(f"    [DEBUG] gamma_0范围: [{gamma_i.min():.3f}, {gamma_i.max():.3f}]")
                print(f"    [DEBUG] c_phys范围: [{prices.min():.3f}, {prices.max():.3f}]")
                print(f"    [DEBUG] c_i_virtual范围: [{c_i_virtual.min():.3f}, {c_i_virtual.max():.3f}]")
            
            v_new_individual[i] = device.solve_linear_program(c_i_virtual)
            
            # 【调试】第一次迭代第一个设备时，验证solve_linear_program是否调用了正确的b/p
            if iteration == 1 and i == 0:
                print(f"    [DEBUG] 设备0 solve_linear_program 返回的信号范围: [{v_new_individual[i].min():.3f}, {v_new_individual[i].max():.3f}]")
        
        # 聚合得到新的虚拟聚合顶点
        v_new = np.sum(v_new_individual, axis=0)
        
        # 【修正】使用真实的个体分解计算物理信号
        v_phys_new = np.zeros(T)
        for i in range(N):
            tcl = tcl_objs[i]
            for t in range(T):
                # 【关键修正】使用a^(t+1)与逆变换一致
                time_index = t + 1
                gamma_it = (tcl.a ** time_index) / tcl.delta
                v_phys_new[t] += gamma_it * v_new_individual[i, t]
        
        cost_new = np.sum(prices * (P0_physical + v_phys_new))
        
        # 【调试】第一次迭代时输出
        if iteration == 1:
            print(f"    [DEBUG] 物理信号范围: [{v_phys_new.min():.3f}, {v_phys_new.max():.3f}]")
            print(f"    [DEBUG] 新顶点物理成本: {cost_new:.3f}")
        
        # 计算改善程度
        reduced_cost = cost_new - mu
        print(f"    子问题：新顶点成本={cost_new:.3f}, ReducedCost={reduced_cost:.6f}")
        
        if iteration == 1:
            print(f"    [DEBUG] 当前最优成本: {best_cost:.3f}")
            print(f"    [DEBUG] 对偶μ: {mu:.6f}")
        
        # 检查收敛（Reduced Cost >= 0 表示无改进列）
        # 【修正】增加顶点限制并添加相对容差检查
        relative_gap = abs(reduced_cost) / max(abs(best_cost), 1e-6)
        if reduced_cost >= -tolerance or num_vertices >= 200:
            if num_vertices >= 200:
                print(f"  列生成达到最大顶点数限制 (vertices={num_vertices}, RelGap={relative_gap:.6f})")
            else:
                print(f"  列生成收敛 (ReducedCost={reduced_cost:.6e}, RelGap={relative_gap:.6f})")
            break
        
        # 添加新顶点及其分解
        vertices_virtual.append(v_new)
        vertices_individual.append(v_new_individual)
        vertices_physical.append(v_phys_new)
    
    # ===== 步骤3：解析最优解 =====
    # 重新求解最终主问题以获得最优λ
    final_master = gp.Model("cost_master_final")
    final_master.setParam('OutputFlag', 0)
    
    num_vertices = len(vertices_virtual)
    lambda_vars_final = final_master.addVars(num_vertices, lb=0.0, name="lambda")
    
    obj_expr_final = 0.0
    for j, v_phys in enumerate(vertices_physical):
        physical_load = P0_physical + v_phys
        cost_j = np.sum(prices * physical_load)
        obj_expr_final += lambda_vars_final[j] * cost_j
    
    final_master.setObjective(obj_expr_final, GRB.MINIMIZE)
    final_master.addConstr(gp.quicksum(lambda_vars_final[j] for j in range(num_vertices)) == 1.0)
    
    final_master.optimize()
    
    lambda_final = np.array([lambda_vars_final[j].X for j in range(num_vertices)])
    
    # ===== 【关键修正】按照g-polymatroid理论进行分解 =====
    # 根据 Corollary 2 和 Theorem 7:
    # 最优个体虚拟信号 = Σ_j λ_j* · v_ij
    # 其中 v_ij 是聚合顶点 v_j 的个体分解
    
    u_individual_virtual = np.zeros((N, T))
    for j in range(num_vertices):
        # vertices_individual[j] 是 (N, T) 的个体顶点分解
        u_individual_virtual += lambda_final[j] * vertices_individual[j]
    
    # 计算虚拟聚合信号（验证用）
    u_agg_virt_opt = np.sum(u_individual_virtual, axis=0)
    
    # 逆变换到物理坐标
    u_individual_physical, u_phys_agg_opt = _inverse_transform_to_physical(
        u_individual_virtual, tcl_objs, T
    )
    
    total_cost = final_master.ObjVal
    print(f"  成本优化完成: 物理成本={total_cost:.3f} (顶点数={num_vertices})")
    
    return u_individual_physical, u_phys_agg_opt, total_cost


def optimize_peak_column_generation(aggregator_virtual, P0_physical, tcl_objs, T, prices=None,
                                   max_iterations=200, tolerance=1e-2):
    """
    使用列生成算法优化物理L-infinity目标（峰值）
    
    主问题：
        min_λ,t  t
        s.t.   P_0[k] + Σ_j λ_j Σ_i γ_i[k] v_ij[k] ≤ t, ∀k
               Σ_j λ_j = 1
               λ_j ≥ 0
    
    子问题：给定对偶变量π_k，求解
        min  Σ_k π_k · v[k], s.t. v ∈ F_agg
    
    Args:
        aggregator_virtual: Aggregator对象
        P0_physical: 物理基线功率 (T,)
        tcl_objs: TCL对象列表
        T: 时间步数
        prices: 电价向量（保留以兼容并行版本，未使用）
        max_iterations: 最大迭代次数
        tolerance: 对偶间隙容差
    
    Returns:
        u_individual_physical: (N, T) 物理坐标个体信号
        u_agg_physical: (T,) 物理坐标聚合信号
        peak_value: 优化后的物理峰值
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
        GUROBI_AVAILABLE = True
    except (ImportError, ModuleNotFoundError, Exception):
        GUROBI_AVAILABLE = False
    
    if not GUROBI_AVAILABLE:
        print("  警告: Gurobi不可用,使用初始顶点近似")
        c_virtual = np.zeros(T)
        u_agg_virtual = aggregator_virtual.solve_linear_program(c_virtual)
        u_individual_virtual = aggregator_virtual.disaggregate(u_agg_virtual)
        u_individual_physical, u_agg_physical = _inverse_transform_to_physical(
            u_individual_virtual, tcl_objs, T
        )
        # 使用L-infinity范数（与algo_exact.py一致）
        peak_value = np.linalg.norm(P0_physical + u_agg_physical, ord=np.inf)
        return u_individual_physical, u_agg_physical, peak_value
    
    print("  使用列生成算法优化峰值...")
    
    N = len(tcl_objs)
    
    # ===== 步骤1：启发式温启动 - 生成T+2个智能顶点 =====
    # 
    # 【优化温启动策略】减少初始顶点数以加速
    # 
    # 从2T+2减少到T+2个"智能"顶点:
    # - v1: 全局最大充电 (降低整体峰值)
    # - v2: 全局最小充电 (对比基准)
    # - v3...v_{T+2}: 只在时间t最大充电 (T个单时刻峰值抑制顶点)
    # 
    # 效果: 减少温启动时间，同时保留关键顶点
    # 
    vertices_virtual = []
    vertices_physical = []
    vertices_individual = []  # 存储个体顶点分解 v_ij (Corollary 2)
    
    print("  [温启动] 生成T+2个智能初始顶点...")
    
    initial_prices = []
    
    # v1: 全局最大充电 (所有时间步都尽量降低功率)
    initial_prices.append(-np.ones(T) * 10.0)
    
    # v2: 全局最小充电 (对比基准,探索边界)
    initial_prices.append(np.ones(T) * 10.0)
    
    # v3...v_{T+2}: 单时刻最大充电顶点
    # 为每个时间步t生成一个"只在t时刻大力降低功率"的顶点
    for t in range(T):
        c_t = np.zeros(T)
        c_t[t] = -100.0  # 只在时间t给予强负价格,鼓励降低该时刻峰值
        initial_prices.append(c_t)
    
    print(f"  [温启动] 共{len(initial_prices)}个初始价格向量 (T+2={T+2})")
    
    for c_init in initial_prices:
        # 直接为每个设备生成个体顶点（Corollary 2实现）
        v_init_individual = np.zeros((N, T))
        for i, device in enumerate(aggregator_virtual.fleet):
            tcl = tcl_objs[i]
            gamma_i = np.array([(tcl.a ** t) / tcl.delta for t in range(T)])
            c_i_virtual = gamma_i * c_init  # 虚拟坐标价格向量
            v_init_individual[i] = device.solve_linear_program(c_i_virtual)
        vertices_individual.append(v_init_individual)
        
        # 聚合得到初始聚合顶点
        v_init = np.sum(v_init_individual, axis=0)
        vertices_virtual.append(v_init)
        
        # 【修正】使用个体顶点精确计算物理信号
        v_phys_init = np.zeros(T)
        for i in range(N):
            tcl = tcl_objs[i]
            for t in range(T):
                # 【关键修正】使用a^(t+1)与逆变换一致
                time_index = t + 1
                gamma_it = (tcl.a ** time_index) / tcl.delta
                v_phys_init[t] += gamma_it * v_init_individual[i, t]
        vertices_physical.append(v_phys_init)
    
    print(f"  初始顶点生成完毕")
    
    # ===== 步骤2：列生成迭代 =====
    iteration = 0
    best_peak = np.inf
    
    while iteration < max_iterations:
        iteration += 1
        
        # --- 2.1: 求解主问题 ---
        master = gp.Model("peak_master")
        master.setParam('OutputFlag', 0)
        
        num_vertices = len(vertices_virtual)
        lambda_vars = master.addVars(num_vertices, lb=0.0, name="lambda")
        peak_var = master.addVar(lb=0.0, name="peak")
        
        # 目标：最小化峰值
        master.setObjective(peak_var, GRB.MINIMIZE)
        
        # 约束1：λ的凸组合
        master.addConstr(gp.quicksum(lambda_vars[j] for j in range(num_vertices)) == 1.0, "convex_comb")
        
        # 约束2：L-infinity范数约束 (|P0(t)+Σ_j λ_j·v_j(t)| <= peak_var)
        for k in range(T):
            agg_expr = gp.quicksum(lambda_vars[j] * vertices_physical[j][k] for j in range(num_vertices))
            total_expr = agg_expr + P0_physical[k]
            master.addConstr(total_expr - peak_var <= 0, name=f"linfty_upper_{k}")
            master.addConstr(-total_expr - peak_var <= 0, name=f"linfty_lower_{k}")
        
        master.optimize()
        
        if master.Status != GRB.OPTIMAL:
            print(f"  迭代{iteration}: 主问题求解失败")
            break
        
        best_peak = peak_var.X
        lambda_opt = np.array([lambda_vars[j].X for j in range(num_vertices)])
        convex_constr = master.getConstrByName("convex_comb")
        mu = convex_constr.Pi if convex_constr is not None else 0.0
        
        # 获取对偶变量（L-infinity约束的对偶）
        # 约束形式：
        #  (P0 + Σ λ·v) - peak_var <= 0   (对偶 π_upper >= 0)
        # -(P0 + Σ λ·v) - peak_var <= 0   (对偶 π_lower >= 0)
        # Reduced Cost: RC = Σ_t (π_upper - π_lower)·v_new(t) - μ
        # 有效对偶向量: π = π_upper - π_lower
        
        constrs_upper = [master.getConstrByName(f"linfty_upper_{k}") for k in range(T)]
        constrs_lower = [master.getConstrByName(f"linfty_lower_{k}") for k in range(T)]
        
        # Gurobi对最小化问题的 "<=" 约束返回的Pi通常为非正值
        # 将它们取相反数以得到标准的非负对偶
        pi_upper_raw = np.array([c.Pi if c is not None else 0.0 for c in constrs_upper])
        pi_lower_raw = np.array([c.Pi if c is not None else 0.0 for c in constrs_lower])
        pi_upper = -pi_upper_raw
        pi_lower = -pi_lower_raw
        pi_vec = pi_upper - pi_lower
        
        print(f"  迭代{iteration}: 峰值={best_peak:.3f}, 顶点数={num_vertices}, μ={mu:.6f}")
        print(f"    对偶变量: π_upper range=[{pi_upper.min():.6f}, {pi_upper.max():.6f}]")
        print(f"    对偶变量: π_lower range=[{pi_lower.min():.6f}, {pi_lower.max():.6f}]")
        print(f"    有效对偶: π range=[{pi_vec.min():.6f}, {pi_vec.max():.6f}]")
        
        # 【调试】第一次迭代时输出更多信息
        if iteration == 1:
            print(f"    [DEBUG] P0范围: [{P0_physical.min():.3f}, {P0_physical.max():.3f}]")
            print(f"    [DEBUG] 对偶变量π (前5个): {pi_vec[:5]}")
            active_dual = pi_vec[np.abs(pi_vec) > 1e-6]
            if len(active_dual) > 0:
                print(f"    [DEBUG] 活跃对偶变量 (|π| > 1e-6): {len(active_dual)}个, 范围=[{active_dual.min():.6f}, {active_dual.max():.6f}]")
        
        # --- 2.2: 求解子问题 ---
        # 【正确的列生成子问题推导】
        # 主问题: min peak_var, s.t. |P0(t) + Σ_j λ_j·v_j(t)| <= peak_var
        # 两个不等式约束分别有对偶 π_upper, π_lower >= 0
        # Reduced Cost: RC = Σ_t (π_upper - π_lower) · v_new(t) - μ
        # 子问题目标: min Σ_t π_t · v_new(t)，其中 π_t = π_upper_t - π_lower_t
        # v_new(t) = Σ_i γ_i(t)·ũ_i(t)，⇒ 每个设备的成本是 π_t·γ_i(t)
        
        # 直接为每个设备生成新顶点（Corollary 2实现）
        v_new_individual = np.zeros((N, T))
        for i, device in enumerate(aggregator_virtual.fleet):
            tcl = tcl_objs[i]
            gamma_i = np.array([(tcl.a ** t) / tcl.delta for t in range(T)])
            c_i_virtual = gamma_i * pi_vec
            v_new_individual[i] = device.solve_linear_program(c_i_virtual)
            
            # 【调试】第一次迭代第一个设备时输出
            if iteration == 1 and i == 0:
                print(f"    [DEBUG] 设备0: gamma范围=[{gamma_i.min():.3f}, {gamma_i.max():.3f}]")
                print(f"    [DEBUG] 设备0: c_i_virtual范围=[{c_i_virtual.min():.6f}, {c_i_virtual.max():.6f}]")
                print(f"    [DEBUG] 设备0: v_new范围=[{v_new_individual[i].min():.3f}, {v_new_individual[i].max():.3f}]")
        
        # 聚合得到新的聚合顶点
        v_new = np.sum(v_new_individual, axis=0)
        
        # 【修正】使用个体顶点精确计算物理信号
        v_phys_new = np.zeros(T)
        for i in range(N):
            tcl = tcl_objs[i]
            for t in range(T):
                # 【关键修正】使用a^(t+1)与逆变换一致
                time_index = t + 1
                gamma_it = (tcl.a ** time_index) / tcl.delta
                v_phys_new[t] += gamma_it * v_new_individual[i, t]
        
        # 使用L-infinity范数计算峰值（与algo_exact.py一致）
        peak_new = np.linalg.norm(P0_physical + v_phys_new, ord=np.inf)
        
        reduced_cost = np.dot(pi_vec, v_phys_new) - mu
        print(f"    子问题：新顶点峰值={peak_new:.3f}, ReducedCost={reduced_cost:.6f}")
        
        # 【调试】第一次迭代时输出
        if iteration == 1:
            print(f"    [DEBUG] v_phys_new范围: [{v_phys_new.min():.3f}, {v_phys_new.max():.3f}]")
            print(f"    [DEBUG] P0+v_phys_new范围: [{(P0_physical+v_phys_new).min():.3f}, {(P0_physical+v_phys_new).max():.3f}]")
            print(f"    [DEBUG] 当前最优峰值: {best_peak:.3f}, ReducedCost: {reduced_cost:.6f}")
        
        # 检查收敛（Reduced Cost >= 0 表示无改进列）
        # 【修正】增加顶点限制并添加相对容差检查
        relative_gap = abs(reduced_cost) / max(abs(best_peak), 1e-6)
        if reduced_cost >= -tolerance or num_vertices >= 200:
            if num_vertices >= 200:
                print(f"  列生成达到最大顶点数限制 (vertices={num_vertices}, RelGap={relative_gap:.6f})")
            else:
                print(f"  列生成收敛 (ReducedCost={reduced_cost:.6e}, RelGap={relative_gap:.6f})")
            break
        
        # 添加新顶点
        vertices_virtual.append(v_new)
        vertices_physical.append(v_phys_new)
        vertices_individual.append(v_new_individual)  # 已在上面生成
    
    # ===== 步骤3：解析最优解 =====
    final_master = gp.Model("peak_master_final")
    final_master.setParam('OutputFlag', 0)
    
    num_vertices = len(vertices_virtual)
    lambda_vars_final = final_master.addVars(num_vertices, lb=0.0, name="lambda")
    peak_var_final = final_master.addVar(lb=0.0, name="peak")
    
    final_master.setObjective(peak_var_final, GRB.MINIMIZE)
    final_master.addConstr(gp.quicksum(lambda_vars_final[j] for j in range(num_vertices)) == 1.0)
    
    # L-infinity范数约束（与主问题一致）
    for k in range(T):
        agg_expr = gp.quicksum(lambda_vars_final[j] * vertices_physical[j][k] for j in range(num_vertices))
        total_expr = agg_expr + P0_physical[k]
        final_master.addConstr(total_expr - peak_var_final <= 0, name=f"linfty_upper_{k}")
        final_master.addConstr(-total_expr - peak_var_final <= 0, name=f"linfty_lower_{k}")
    
    final_master.optimize()
    
    lambda_final = np.array([lambda_vars_final[j].X for j in range(num_vertices)])
    
    # ===== 【关键修正】按照g-polymatroid理论进行分解 =====
    # 根据 Corollary 2 和 Theorem 7:
    # 最优个体虚拟信号 = Σ_j λ_j* · v_ij
    # 其中 v_ij 是聚合顶点 v_j 的个体分解
    
    u_individual_virtual = np.zeros((N, T))
    for j in range(num_vertices):
        # vertices_individual[j] 是 (N, T) 的个体顶点分解
        u_individual_virtual += lambda_final[j] * vertices_individual[j]
    
    # 计算虚拟聚合信号（验证用）
    u_agg_virt_opt = np.sum(u_individual_virtual, axis=0)
    
    # 逆变换到物理坐标
    u_individual_physical, u_phys_agg_opt = _inverse_transform_to_physical(
        u_individual_virtual, tcl_objs, T
    )
    
    peak_value = final_master.ObjVal
    print(f"  峰值优化完成: 物理峰值={peak_value:.3f} (顶点数={num_vertices})")
    
    return u_individual_physical, u_phys_agg_opt, peak_value


def _compute_physical_signal(u_virtual_agg, tcl_objs, T):
    """
    给定虚拟聚合信号，计算对应的物理聚合信号
    
    这是一个启发式的计算，用于列生成中评估顶点的物理成本/峰值
    假设虚拟聚合信号通过"均衡分配"分解为个体：
    ũ_i = u_agg / N （简单平均）
    
    然后逆变换：u_phys_i = γ_i · ũ_i = γ_i · (u_agg / N)
    
    物理聚合：P_agg = Σ_i u_phys_i = Σ_i γ_i · (u_agg / N) = (Σ_i γ_i / N) · u_agg
    
    注意：这是一个启发式方法。完整的方法需要知道实际的分解方式。
    
    Args:
        u_virtual_agg: (T,) 虚拟聚合信号
        tcl_objs: TCL对象列表
        T: 时间步数
    
    Returns:
        u_physical_agg: (T,) 物理聚合信号（启发式估计）
    """
    N = len(tcl_objs)
    u_physical_agg = np.zeros(T)
    
    for t in range(T):
        # 计算时间t的平均逆变换系数
        gamma_avg_t = 0.0
        for i in range(N):
            tcl = tcl_objs[i]
            gamma_it = (tcl.a ** t) / tcl.delta
            gamma_avg_t += gamma_it
        gamma_avg_t /= N
        
        # 物理聚合（假设均衡分配）
        u_physical_agg[t] = u_virtual_agg[t] * gamma_avg_t
    
    return u_physical_agg


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
            # 【修正】逆变换: u_phys[t] = (a^(t+1) / δ) · ũ_virt[t]
            # 与正向变换一致: ũ[t] = δ·u[t] / a^(t+1)
            time_index = t + 1
            scale = (a_i ** time_index) / delta_i if delta_i > 1e-10 else 1.0
            u_phys_i[t] = u_individual_virtual[i, t] * scale
        
        u_individual_physical.append(u_phys_i)
    
    u_individual_physical = np.array(u_individual_physical)
    u_agg_physical = np.sum(u_individual_physical, axis=0)
    
    return u_individual_physical, u_agg_physical
