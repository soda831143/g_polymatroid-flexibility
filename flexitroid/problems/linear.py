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
'''

class LinearProgram():
    def __init__(self, feasible_set: Flexitroid, A: np.ndarray, b: np.ndarray, c: np.ndarray):
        """Initialize the linear program with Dantzig-Wolfe decomposition.
        
        Args:
            X: Flexitroid object representing the feasible set
            A: Constraint matrix
            b: Constraint bounds
            c: Cost vector
        """
        self.feasible_set = feasible_set # 存储g-polymatroid可行集
        self.A = A # 存储额外约束矩阵
        self.b = b # 存储额外约束边界
        self.c = c # 存储目标函数成本向量
        self.epsilon = 1e-6  # Convergence tolerance
        self.max_iter = 1000  # Maximum iterations

        self.lmda = None # 用于存储主问题的解 (权重λ)
        self.v_subset = None # 用于存储已生成的 Q(p,b) 的顶点子集 V_subset
        self.solution = None # 用于存储最终解 u*

    def solve(self):
        """Solve the linear program using Dantzig-Wolfe decomposition.
        
        Returns:
            Optimal solution vector
        """
        if self.lmda == None: # 如果还没求解过
            lmda, v_subset = self.dantzig_wolfe() # 执行Dantzig-Wolfe算法
            self.lmda = lmda
            self.v_subset = v_subset
            self.solution = lmda@v_subset # 最优解 u* 是顶点的凸组合
    
    def dantzig_wolfe(self):
        V_subset = self.form_initial_set() # 初始化顶点子集 (生成初始的可行基)
        i=0 # 迭代计数

        while i<self.max_iter:
            print(i, end='\r') # 打印当前迭代次数
            A_V = np.einsum('ij,kj->ik', self.A, V_subset)
            # A_V 矩阵的每一列是 A @ v_k，其中 v_k 是 V_subset 中的一个顶点
            c_V = np.einsum('j,kj->k', self.c, V_subset)
            # c_V 向量是 c 和 V_subset 的点积，表示每个顶点 v_k 的代价

            y, alpha, lmda = self.solve_dual(A_V, c_V)
            # 求解限制主问题 (Restricted Master Problem - RMP) 的对偶问题
            # RMP: min c_V^T λ s.t. A_V λ <= b, sum(λ) = 1, λ >= 0
            # 通过求解其对偶问题得到对偶变量 y (对应 A_V λ <= b) 和 alpha (对应 sum(λ) = 1)
            # lmda 是原RMP的解，即顶点的权重

            # 计算子问题的目标函数中的 c' (即 d 在代码中的表示)
            # 子问题是 min (c - A^T y)^T v s.t. v in Q(p,b)
            # d 对应 c - A^T y
            d = self.c - np.einsum('i,ij->j', y, self.A)

            # 求解子问题：在 feasible_set (Q(p,b)) 上最小化 d_vec^T v
            # 这会返回一个新的顶点 new_vertex
            new_vertex = self.feasible_set.solve_linear_program(d)

            # 计算 reduced_cost = d_vec^T new_vertex - alpha
            # 如果 reduced_cost >= 0 (或非常接近0)，则已找到最优解
            if d@new_vertex - alpha > -self.epsilon:
                break
            V_subset = np.vstack([V_subset, new_vertex]) # 将新顶点加入到顶点子集中
            i += 1
        if not i < self.max_iter: # 如果达到最大迭代次数仍未收敛
            raise Exception('Did not converge')
        return lmda, V_subset # 返回顶点的权重和最终的顶点集
    
    def form_initial_set(self):
        """为Dantzig-Wolfe算法生成一个初始的顶点集合，确保初始主问题可行。"""
        V_subset = self.feasible_set.form_box() # form_box() 可能返回Q(p,b)的一些基本顶点，如坐标轴方向最远点
        # 这个循环是为了找到一个初始的顶点集，使得主问题有可行解
        while True:
            A_V = np.einsum('ij,kj->ik', self.A, V_subset) 

            y, alpha = self.initial_vertex_dual(A_V) # 求解一个用于寻找可行初始解的对偶问题

            d = - np.einsum('i,ij->j', y, self.A) # 子问题的目标 c' = -A^T y
            new_vertex = self.feasible_set.solve_linear_program(d) # 求解子问题，找到新的顶点

            if d@new_vertex - alpha > -1e-6: # 判断是否还需要添加顶点以保证初始可行性
                break
            V_subset = np.vstack([V_subset, new_vertex])
        return V_subset

    
    def initial_vertex_dual(self, A_V):
        """求解用于 form_initial_set 的对偶问题（一个LP可行性问题）。"""
        y = cp.Variable(self.b.shape[0], neg=True) # 对偶变量 y <= 0
        alpha = cp.Variable() # 对偶变量 alpha

        dual_obj = cp.Maximize(y@self.b + alpha) # 对偶目标函数

        dual_constraints = []
        dual_constraints.append(A_V.T @ y + alpha <= 0) # 对偶约束
        # 以下约束可能是为了界定 alpha 或确保问题有界
        dual_constraints.append(alpha <= 1)
        dual_constraints.append(-alpha <= 1)

        dual_prob = cp.Problem(dual_obj, dual_constraints)
        dual_prob.solve()

        return y.value, alpha.value
    
    def solve_dual(self, A_V, c_V):
        """求解Dantzig-Wolfe主问题的对偶问题。
        原主问题 (RMP): min c_V^T λ s.t. A_V λ <= b, sum(λ) = 1, λ >= 0
        其对偶问题是: max y^T b + alpha s.t. y^T A_V_k + alpha <= c_V_k (for each k), y <= 0
        """
    
        y = cp.Variable(self.b.shape[0], neg=True) # 对偶变量 y (对应 A_V λ <= b)，由于原约束是 <=，对偶变量 y <= 0
        alpha = cp.Variable() # 对偶变量 alpha (对应 sum(λ) = 1)

        dual_obj = cp.Maximize(y@self.b + alpha) # 对偶目标函数
        dual_constraints = [A_V.T @ y + alpha <= c_V] # 对偶约束 (A_V_k^T y + alpha <= c_V_k)
        dual_prob = cp.Problem(dual_obj, dual_constraints)
        dual_prob.solve()

        # dual_constraints[0].dual_value 是对偶约束的对偶变量，即原RMP的 primal variables λ

        return y.value, alpha.value, dual_constraints[0].dual_value
