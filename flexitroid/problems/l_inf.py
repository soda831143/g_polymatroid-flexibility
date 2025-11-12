import numpy as np
import cvxpy as cp
from flexitroid.flexitroid import Flexitroid
'''
这些文件实现了在 Flexitroid 对象（代表可行域，即g-polymatroid）上求解不同类型优化问题的类。
LinearProgram：使用Dantzig-Wolfe分解方法来求解线性规划问题，其子问题是调用 feasible_set.solve_linear_program(d)（即在g-polymatroid上用贪心算法求解LP）。
dantzig_wolfe 方法使用贪心算法迭代求解子问题，直到收敛。
form_initial_set 方法用于初始化子问题集合。
initial_vertex_dual 方法用于求解初始顶点的对偶问题。
solve_dual 方法用于求解子问题的对偶问题。
求解 L∞范数最小化问题，也使用了Dantzig-Wolfe分解。
'''
class L_inf():
    def __init__(self, feasible_set: Flexitroid):
        """Initialize the linear program with Dantzig-Wolfe decomposition.
        
        Args:
            X: Flexitroid object representing the feasible set
            A: Constraint matrix
            b: Constraint bounds
            c: Cost vector
        """
        """初始化L-无穷范数最小化问题。
        
        Args:
            feasible_set: Flexitroid对象，代表可行集 Q(p,b)。
        """
        self.feasible_set = feasible_set
        self.epsilon = 1e-6  # Convergence tolerance
        self.max_iter = 1000  # Maximum iterations

        self.lmda = None # 存储主问题解 (权重λ)
        self.v_subset = None # 存储已生成的顶点子集
        self.solution = None # 存储最终解 u*


    def solve(self):
        """Solve the linear program using Dantzig-Wolfe decomposition.
        
        Returns:
            Optimal solution vector
        """
        """使用类似Dantzig-Wolfe的方法求解L-无穷范数最小化问题。"""
        if self.lmda == None:
            lmda, v_subset = self.dantzig_wolfe()
            self.lmda = lmda
            self.v_subset = v_subset
            self.solution = lmda@v_subset # 解 u* 是顶点的凸组合
    
    def dantzig_wolfe(self):
        """L-无穷范数最小化算法的主体。"""
        v_subset = self.feasible_set.form_box() # 初始化顶点子集

        i = 0 # 迭代计数
        while True:
            i+=1
            print(i, end='\r')

             # 定义主问题变量
            t = cp.Variable(nonneg=True) # 对应 L-无穷范数的界 t
            lmda = cp.Variable(v_subset.shape[0], nonneg=True) # 顶点的权重 λ

            # 主问题约束
            # u = lmda @ v_subset (u是顶点的凸组合)
            # 1. sum(λ_k) = 1
            con_convex = [cp.sum(lmda) == 1]
            # 2. u_j <= t  => (lmda @ v_subset)_j <= t
            con_upper = [lmda@v_subset <= t]
            # 3. -u_j <= t  => -(lmda @ v_subset)_j <= t
            con_lower = [-lmda@v_subset <= t,] # 即 lmda @ v_subset >= -t_var
            constraints = con_convex + con_upper + con_lower

            objective = cp.Minimize(t)  # 目标：最小化 t
            prob = cp.Problem(objective, constraints)
            prob.solve() # 使用cvxpy求解主问题

            # 获取对偶变量以构造子问题
            # mu 是 sum(λ) = 1 约束的对偶变量
            mu = con_convex[0].dual_value
            # pi_plus 是 u_j <= t 约束的对偶变量
            pi_plus = con_upper[0].dual_value
            # pi_minus 是 -u_j <= t 约束的对偶变量
            pi_minus = con_lower[0].dual_value
            # 子问题的目标函数中的成本向量 pi = pi_plus - pi_minus
            # 子问题是 min (pi_plus - pi_minus)^T v - mu
            pi = pi_plus - pi_minus

            # 求解子问题：在 feasible_set (Q(p,b)) 上最小化 pi_vec^T v
            new_vertex = self.feasible_set.solve_linear_program(pi)

            # 计算 reduced_cost
            # reduced_cost = (pi_plus - pi_minus)^T new_vertex - mu
            # 如果 reduced_cost >= 0 (或接近0)，则收敛
            # 注意：这里的 reduced_cost 定义与标准Dantzig-Wolfe的 c_j - z_j 可能略有不同，
            # 具体取决于对偶变量的符号和主问题形式。
            # 此处的条件 < 1e-9 用于判断是否继续迭代 (如果 < 阈值，表示还可以改进)。
            # 通常是 reduced_cost > -epsilon。这里的判断条件是 `reduced_cost < 1e-9` 然后 break。
            # 这似乎意味着如果 reduced_cost 已经“足够负”（或接近0从负方向），
            # 或者说 new_vertex 不能带来显著改善，就停止。
            # 标准的停止条件是 min_v {c_sub^T v} - alpha >= 0.
            # 这里是 (pi_vec^T new_vertex) - (-mu).
            # 所以是 pi_vec^T new_vertex + mu.
            # 如果这个值很小（例如接近0或负数），则可能表示可以继续。
            # 如果 `pi_vec.dot(new_vertex) + mu` (这里的mu是上面con_convex[0].dual_value) >= 0, 则停止。
            # 代码中的 `reduced_cost = - mu - np.dot(new_vertex, pi)`
            # 如果 `reduced_cost < 1e-9` (即 `-mu - new_vertex @ pi < 1e-9`)，则终止。
            # 这等价于 `new_vertex @ pi + mu > -1e-9`。
            # 所以，当新顶点的贡献不能使得对偶目标值进一步降低（或降低很少）时，算法终止。
            reduced_cost = - mu - np.dot(new_vertex, pi)

            if reduced_cost < 1e-9: # 这个条件需要仔细核对，通常是 reduced_cost > -epsilon 时终止
                                    # 或者像这里，如果它不够负，就终止。
                print('Terminating')
                break
            else: # 如果 reduced_cost 足够负，说明 new_vertex 可以改进解
                v_subset = np.vstack([v_subset, new_vertex]) # 将新顶点加入

        if i > self.max_iter: # 修正: 应该是 i >= self.max_iter
            raise Exception('Did not converge')
        return lmda.value, v_subset
    