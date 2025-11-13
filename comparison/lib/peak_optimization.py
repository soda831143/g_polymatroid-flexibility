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
import numpy as np


def optimize_cost_column_generation(aggregator_virtual, prices, P0_physical, tcl_objs, T, max_iterations=100, tolerance=1e-3):
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
    
    # 【关键修正】初始顶点：使用物理价格生成（而不是零向量）
    # 这样初始顶点就已经是一个合理的解
    v_init_individual = np.zeros((N, T))
    for i, device in enumerate(aggregator_virtual.fleet):
        tcl = tcl_objs[i]
        gamma_i = np.array([(tcl.a ** t) / tcl.delta for t in range(T)])
        c_i_virtual = gamma_i * prices  # 设备i的虚拟坐标价格
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
            gamma_it = (tcl.a ** t) / tcl.delta
            v_phys_init[t] += gamma_it * v_init_individual[i, t]
    vertices_physical.append(v_phys_init)
    
    cost_init = np.sum(prices * (P0_physical + v_phys_init))
    print(f"  初始顶点生成完毕")
    print(f"  [DEBUG] 初始虚拟聚合信号范围: [{v_init.min():.3f}, {v_init.max():.3f}]")
    print(f"  [DEBUG] 初始物理聚合信号范围: [{v_phys_init.min():.3f}, {v_phys_init.max():.3f}]")
    print(f"  [DEBUG] 初始物理成本={cost_init:.3f} (与Exact Minkowski {46.21:.2f}比较)")
    
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
        
        # 获取对偶变量
        pi = master.getAttr("pi", master.getConstrs())[0] if len(master.getConstrs()) > 0 else 0.0
        
        print(f"  迭代{iteration}: 目标值={best_cost:.3f}, 顶点数={num_vertices}, 对偶={pi:.6f}")
        
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
                gamma_it = (tcl.a ** t) / tcl.delta
                v_phys_new[t] += gamma_it * v_new_individual[i, t]
        
        cost_new = np.sum(prices * (P0_physical + v_phys_new))
        
        # 【调试】第一次迭代时输出
        if iteration == 1:
            print(f"    [DEBUG] 物理信号范围: [{v_phys_new.min():.3f}, {v_phys_new.max():.3f}]")
            print(f"    [DEBUG] 新顶点物理成本: {cost_new:.3f}")
        
        # 计算改善程度
        improvement = best_cost - cost_new
        print(f"    子问题：新顶点成本={cost_new:.3f}, 改善={improvement:.6f}")
        
        # 【调试】第一次迭代时详细输出
        if iteration == 1:
            print(f"    [DEBUG] 当前最优成本: {best_cost:.3f}")
            print(f"    [DEBUG] 新顶点成本: {cost_new:.3f}")
            print(f"    [DEBUG] 理论上应该收敛到 Exact Minkowski 的成本 (~46.21)")
        
        # 检查收敛（改善不足或迭代上限）
        if improvement < tolerance or num_vertices > 50:  # 防止顶点过多
            print(f"  列生成收敛 (改善={improvement:.6e})")
            print(f"  [INFO] 如果收敛成本远小于46.21，说明子问题求解有误")
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


def optimize_peak_column_generation(aggregator_virtual, P0_physical, tcl_objs, T, max_iterations=100, tolerance=1e-3):
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
        peak_value = np.max(P0_physical + u_agg_physical)
        return u_individual_physical, u_agg_physical, peak_value
    
    print("  使用列生成算法优化峰值...")
    
    N = len(tcl_objs)
    
    # ===== 步骤1：初始化 - 生成多个初始顶点 =====
    vertices_virtual = []
    vertices_physical = []
    vertices_individual = []  # 存储个体顶点分解 v_ij (Corollary 2)
    
    # 【策略修正】生成多个初始顶点以覆盖不同的极端情况
    # 1. 最小化所有功率 (尽可能使用负功率降低峰值)
    # 2. 针对每个时间步单独优化 (降低该时间步的峰值)
    
    initial_prices = [
        -np.ones(T),  # 全局最小化
        np.ones(T),   # 全局最大化 (作为对比)
    ]
    
    # 为前几个高峰时间步生成专门的初始顶点
    # 找出P0最大的几个时间步
    peak_times = np.argsort(P0_physical)[-3:]  # 最高的3个时间步
    for t_peak in peak_times:
        c_targeted = np.zeros(T)
        c_targeted[t_peak] = -10.0  # 专门降低这个时间步的功率
        initial_prices.append(c_targeted)
    
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
                gamma_it = (tcl.a ** t) / tcl.delta
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
        peak_var = master.addVar(lb=-gp.GRB.INFINITY, name="peak")
        
        # 目标：最小化峰值
        master.setObjective(peak_var, GRB.MINIMIZE)
        
        # 约束1：λ的凸组合
        master.addConstr(gp.quicksum(lambda_vars[j] for j in range(num_vertices)) == 1.0, "convex_comb")
        
        # 约束2：L-infinity在每个时间步
        # 改写为标准形式: Σ_j λ_j·v_j(t) - z <= -P0(t)
        # 这样对偶变量π(t)的符号就是正确的
        for k in range(T):
            lhs = gp.LinExpr()
            for j in range(num_vertices):
                lhs += lambda_vars[j] * vertices_physical[j][k]
            lhs -= peak_var
            master.addConstr(lhs <= -P0_physical[k], f"linfty_{k}")
        
        master.optimize()
        
        if master.Status != GRB.OPTIMAL:
            print(f"  迭代{iteration}: 主问题求解失败")
            break
        
        best_peak = peak_var.X
        lambda_opt = np.array([lambda_vars[j].X for j in range(num_vertices)])
        
        # 获取对偶变量（L-infinity约束的对偶）
        constrs = [master.getConstrByName(f"linfty_{k}") for k in range(T)]
        pi_vec = master.getAttr("pi", constrs)
        pi_vec = np.array(pi_vec) if pi_vec else np.zeros(T)
        
        print(f"  迭代{iteration}: 峰值={best_peak:.3f}, 顶点数={num_vertices}, 对偶range=[{pi_vec.min():.6f}, {pi_vec.max():.6f}]")
        
        # --- 2.2: 求解子问题 ---
        # 【关键修正】每个设备独立求解虚拟子问题
        # 物理坐标: u_i^phys(t) = (a_i^t / δ_i) · u_i^virt(t)
        # 子问题: max Σ_t π(t) · u_i^phys(t) = max Σ_t [(a_i^t / δ_i) · π(t)] · u_i^virt(t)
        # 转换为最小化: min Σ_t [-(a_i^t / δ_i) · π(t)] · u_i^virt(t)
        # 所以设备i的虚拟价格向量: c_i^virt(t) = -(a_i^t / δ_i) · π(t)
        
        # 直接为每个设备生成新顶点（Corollary 2实现）
        v_new_individual = np.zeros((N, T))
        for i, device in enumerate(aggregator_virtual.fleet):
            tcl = tcl_objs[i]
            gamma_i = np.array([(tcl.a ** t) / tcl.delta for t in range(T)])
            c_i_virtual = -gamma_i * pi_vec  # 注意负号:贪心算法求min,但子问题要max
            v_new_individual[i] = device.solve_linear_program(c_i_virtual)
        
        # 聚合得到新的聚合顶点
        v_new = np.sum(v_new_individual, axis=0)
        
        # 【修正】使用个体顶点精确计算物理信号
        v_phys_new = np.zeros(T)
        for i in range(N):
            tcl = tcl_objs[i]
            for t in range(T):
                gamma_it = (tcl.a ** t) / tcl.delta
                v_phys_new[t] += gamma_it * v_new_individual[i, t]
        
        peak_new = np.max(P0_physical + v_phys_new)
        
        improvement = best_peak - peak_new
        print(f"    子问题：新顶点峰值={peak_new:.3f}, 改善={improvement:.6f}")
        
        # 检查收敛
        if improvement < tolerance or num_vertices > 50:
            print(f"  列生成收敛 (改善={improvement:.6e})")
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
    peak_var_final = final_master.addVar(lb=-gp.GRB.INFINITY, name="peak")
    
    final_master.setObjective(peak_var_final, GRB.MINIMIZE)
    final_master.addConstr(gp.quicksum(lambda_vars_final[j] for j in range(num_vertices)) == 1.0)
    
    for k in range(T):
        lhs = gp.LinExpr()
        for j in range(num_vertices):
            lhs += lambda_vars_final[j] * vertices_physical[j][k]
        lhs -= peak_var_final
        final_master.addConstr(lhs <= -P0_physical[k], f"linfty_{k}")
    
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
            # 逆变换: u_phys[t] = (a^t / δ) · ũ_virt[t]
            scale = (a_i ** t) / delta_i if delta_i > 1e-10 else 1.0
            u_phys_i[t] = u_individual_virtual[i, t] * scale
        
        u_individual_physical.append(u_phys_i)
    
    u_individual_physical = np.array(u_individual_physical)
    u_agg_physical = np.sum(u_individual_physical, axis=0)
    
    return u_individual_physical, u_agg_physical
