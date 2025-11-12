import numpy as np
import cvxpy as cp

def compute_optimal_y_bounds_for_tcl(
    T_horizon: int,                  # 总时间步长 T
    a_thermal: float,                # 热耗散系数 a
    x0_transformed: float,           # 变换后的初始状态 x(0)
    u_min_tcl: np.ndarray,           # TCL变换后输入 u(t) 的下限 (长度为 T_horizon)
    u_max_tcl: np.ndarray,           # TCL变换后输入 u(t) 的上限 (长度为 T_horizon)
    x_min_physical_const: float,     # TCL物理状态的恒定下限 (如最低温度)
    x_max_physical_const: float      # TCL物理状态的恒定上限 (如最高温度)
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算TCL的最大内近似g-polymatroid的累积能量边界 y_star_lower 和 y_star_upper。
    对应TCL论文 Section III-B。
    """

    y_star_lower_np = np.zeros(T_horizon)
    y_star_upper_np = np.zeros(T_horizon)

    # 论文中的时间索引 t 是从 1 到 T
    # Python中循环 t_paper 从 1 到 T_horizon
    for t_paper_idx in range(1, T_horizon + 1):
        # 当前LP的变量维度是 t_paper_idx (对应 u(1)...u(t_paper_idx))
        u_lp_vars = cp.Variable(t_paper_idx, name=f"u_t{t_paper_idx}")

        # --- 定义通用约束 F_i^t (TCL真实可行运行直到 t_paper_idx) ---
        constraints_Fi_t = []
        # 1. 功率约束 u_min_tcl <= u(s) <= u_max_tcl
        # u_lp_vars[s_code_idx] 对应论文中的 u(s_paper_idx = s_code_idx + 1)
        for s_code_idx in range(t_paper_idx):
            constraints_Fi_t.append(u_lp_vars[s_code_idx] >= u_min_tcl[s_code_idx])
            constraints_Fi_t.append(u_lp_vars[s_code_idx] <= u_max_tcl[s_code_idx])

        # 2. 状态约束: x_min_physical_const <= x(k) <= x_max_physical_const
        #    其中 x(k) = a^k * x0 + sum_{s=1 to k} a^(k-s) * u(s)
        #    对于每个 k_paper 从 1 到 t_paper_idx
        for k_paper in range(1, t_paper_idx + 1):
            # sum_term_val = sum(a_thermal**(k_paper - (s_code_idx + 1)) * u_lp_vars[s_code_idx] 
            #                    for s_code_idx in range(k_paper))
            
            # 使用CVXPY的向量化表达会更高效
            # powers_of_a = np.array([a_thermal**(k_paper - (s_code_idx + 1)) for s_code_idx in range(k_paper)])
            # sum_term_cvx = powers_of_a @ u_lp_vars[:k_paper]
            
            # 直接构建表达式
            current_sum_expr = 0
            for s_code_idx in range(k_paper): # s_code_idx from 0 to k_paper-1
                                              # paper's s from 1 to k_paper
                current_sum_expr += (a_thermal**(k_paper - (s_code_idx + 1))) * u_lp_vars[s_code_idx]
            
            # 论文中 x_i(k) 是指物理状态上下限减去初始状态影响
            # x_k_lower_bound_on_sum = x_min_physical_const - (a_thermal**k_paper) * x0_transformed
            # x_k_upper_bound_on_sum = x_max_physical_const - (a_thermal**k_paper) * x0_transformed
            # constraints_Fi_t.append(current_sum_expr >= x_k_lower_bound_on_sum)
            # constraints_Fi_t.append(current_sum_expr <= x_k_upper_bound_on_sum)
            
            # 或者，更直接地，计算状态 x(k) 并约束它：
            # x_k_val = (a_thermal**k_paper) * x0_transformed + current_sum_expr
            # constraints_Fi_t.append(x_k_val >= x_min_physical_const)
            # constraints_Fi_t.append(x_k_val <= x_max_physical_const)

            # 采用论文定义1中的形式： x_i(k) <= sum a^(k-s)u(s) <= x_bar_i(k)
            # where x_underline_i(k) = x_underline_phys - a^k x_0
            # and   x_bar_i(k)     = x_bar_phys     - a^k x_0
            state_sum_lower_bound = x_min_physical_const - (a_thermal**k_paper) * x0_transformed
            state_sum_upper_bound = x_max_physical_const - (a_thermal**k_paper) * x0_transformed
            constraints_Fi_t.append(current_sum_expr >= state_sum_lower_bound)
            constraints_Fi_t.append(current_sum_expr <= state_sum_upper_bound)


        # --- 计算 y_star_lower[t_paper_idx-1] ---
        # Objective: maximize sum_{s=1 to t_paper_idx} u(s)
        objective_lower = cp.Maximize(cp.sum(u_lp_vars))
        
        # Hyperplane constraint H_lower: sum_{s=1 to t_paper_idx} a^(t_paper_idx-s)u(s) = z_lower_i(t_paper_idx)
        sum_for_z_lower_hyperplane = 0
        for s_code_idx in range(t_paper_idx):
            sum_for_z_lower_hyperplane += (a_thermal**(t_paper_idx - (s_code_idx + 1))) * u_lp_vars[s_code_idx]

        sum_for_z_lower_u_min_part = sum(
            a_thermal**(t_paper_idx - (s_code_idx + 1)) * u_min_tcl[s_code_idx] 
            for s_code_idx in range(t_paper_idx)
        )
        z_lower_i_t = np.maximum(
            x_min_physical_const - (a_thermal**t_paper_idx) * x0_transformed,
            sum_for_z_lower_u_min_part
        )
        
        constraints_lower = constraints_Fi_t + [sum_for_z_lower_hyperplane == z_lower_i_t]
        problem_lower = cp.Problem(objective_lower, constraints_lower)

        # 尝试不同的求解器
        available_solvers = cp.installed_solvers()
        preferred_solvers = ['CLARABEL', 'OSQP', 'SCS', 'SCIPY']
        solvers_to_try = []
        
        for solver_name in preferred_solvers:
            if solver_name in available_solvers:
                if solver_name == 'CLARABEL':
                    solvers_to_try.append(cp.CLARABEL)
                elif solver_name == 'OSQP':
                    solvers_to_try.append(cp.OSQP)
                elif solver_name == 'SCS':
                    solvers_to_try.append(cp.SCS)
                elif solver_name == 'SCIPY':
                    solvers_to_try.append(cp.SCIPY)
        
        if not solvers_to_try:
            print("  警告: 没有找到可用的求解器，使用默认求解器")
            problem_lower.solve(verbose=True)
        else:
            for solver in solvers_to_try:
                try:
                    problem_lower.solve(solver=solver)
                    if problem_lower.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        break
                except Exception as e:
                    continue
            else:
                problem_lower.solve(verbose=True)  # 使用默认求解器

        if problem_lower.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Warning: LP for y_lower at t={t_paper_idx} not solved to optimality ({problem_lower.status})")
            # 可以根据需要处理这种情况，例如使用NaN或抛出错误
            y_star_lower_np[t_paper_idx-1] = np.nan 
        else:
            y_star_lower_np[t_paper_idx-1] = problem_lower.value

        # --- 计算 y_star_upper[t_paper_idx-1] ---
        # Objective: minimize sum_{s=1 to t_paper_idx} u(s)
        objective_upper = cp.Minimize(cp.sum(u_lp_vars))

        # Hyperplane constraint H_upper: sum_{s=1 to t_paper_idx} a^(t_paper_idx-s)u(s) = z_upper_i(t_paper_idx)
        sum_for_z_upper_hyperplane = sum_for_z_lower_hyperplane # 表达式相同

        sum_for_z_upper_u_max_part = sum(
            a_thermal**(t_paper_idx - (s_code_idx + 1)) * u_max_tcl[s_code_idx]
            for s_code_idx in range(t_paper_idx)
        )
        z_upper_i_t = np.minimum(
            x_max_physical_const - (a_thermal**t_paper_idx) * x0_transformed,
            sum_for_z_upper_u_max_part
        )

        constraints_upper = constraints_Fi_t + [sum_for_z_upper_hyperplane == z_upper_i_t]
        problem_upper = cp.Problem(objective_upper, constraints_upper)

        # 尝试不同的求解器
        available_solvers = cp.installed_solvers()
        preferred_solvers = ['CLARABEL', 'OSQP', 'SCS', 'SCIPY']
        solvers_to_try = []
        
        for solver_name in preferred_solvers:
            if solver_name in available_solvers:
                if solver_name == 'CLARABEL':
                    solvers_to_try.append(cp.CLARABEL)
                elif solver_name == 'OSQP':
                    solvers_to_try.append(cp.OSQP)
                elif solver_name == 'SCS':
                    solvers_to_try.append(cp.SCS)
                elif solver_name == 'SCIPY':
                    solvers_to_try.append(cp.SCIPY)
        
        if not solvers_to_try:
            print("  警告: 没有找到可用的求解器，使用默认求解器")
            problem_upper.solve(verbose=True)
        else:
            for solver in solvers_to_try:
                try:
                    problem_upper.solve(solver=solver)
                    if problem_upper.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        break
                except Exception as e:
                    continue
            else:
                problem_upper.solve(verbose=True)  # 使用默认求解器

        if problem_upper.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Warning: LP for y_upper at t={t_paper_idx} not solved to optimality ({problem_upper.status})")
            y_star_upper_np[t_paper_idx-1] = np.nan
        else:
            y_star_upper_np[t_paper_idx-1] = problem_upper.value
            
    return y_star_lower_np, y_star_upper_np