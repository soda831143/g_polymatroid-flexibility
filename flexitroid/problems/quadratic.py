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
使用Frank-Wolfe算法来求解二次规划问题，其每一步也需要求解一个线性子问题 feasible_set.solve_linear_program(g)。
'''
class QuadraticProgram():
    def __init__(self, feasible_set: Flexitroid, Q, c, max_iters = 1000):
        """Initialize the linear program with Dantzig-Wolfe decomposition.
        
        Args:
            X         : Flexitroid object representing the feasible set
            Q         : (n x n) numpy array (assumed symmetric positive semidefinite)
            c         : (n,) numpy array
            x0        : Initial feasible point in the simplex (n-dimensional numpy array)
            tol       : Tolerance for the duality gap stopping criterion.
            max_iter  : Maximum number of iterations.
        """
        """初始化二次规划问题。
        
        Args:
            feasible_set: Flexitroid对象，代表可行集 Q(p,b)。
            Q: 二次项的对称正半定矩阵。
            c: 线性项的成本向量。
            max_iters: 最大迭代次数。
        """
        self.feasible_set = feasible_set
        self.Q = Q
        self.c = c
        self.max_iter = max_iters  # Maximum iterations
        self.epsilon = 1e-6  # Convergence tolerance

        self.solution = None

    def solve(self):
        """使用Frank-Wolfe算法求解二次规划。"""
        """Solve the linear program using Dantzig-Wolfe decomposition.
        
        Returns:
            Optimal solution vector
        """
        if self.solution == None:
            self.solution, self.history = self.frank_wolfe() # 执行Frank-Wolfe算法
    
    def frank_wolfe(self):
        """Frank-Wolfe算法的主体。"""
        # 步骤 0: 选择初始可行点 x_0。
        # 这里选择 feasible_set 上 c^T x 的最小点作为初始点。
        x = self.feasible_set.solve_linear_program(self.c)

        history = {'obj': [], 'gap': []} # 用于记录目标值和gap的历史

        for k in range(self.max_iter):
            # Compute gradient: g = Qx + c # 步骤 1: 计算当前点 x 的梯度 g = Qx + c
            g = self.Q @ x + self.c

            # 步骤 2: 求解线性子问题 s_k = argmin_{s in Q(p,b)} g^T s
            # s 是使 g^T s最小化的 feasible_set 中的一个顶点
            s = self.feasible_set.solve_linear_program(g)

            # 计算 Frank-Wolfe gap (收敛判据)
            # gap = g^T (x - s)
            gap = g.dot(x - s)
            
            # Record objective and gap
            obj = 0.5 * np.dot(x,self.Q @ x) + np.dot(self.c,x) # 当前目标函数值
            history['obj'].append(obj)
            history['gap'].append(gap)

            if gap < self.epsilon: # 如果gap小于容忍度
                print('converged') # 达到收敛，退出
                break

            # 步骤 3: 计算步长 gamma
            # d_k = s_k - x_k
            d = s - x
            
            # 对于二次目标函数，最优步长有闭式解:
            # gamma = - (Qx + c)^T d / (d^T Q d) = - g^T d / (d^T Q d)
            # 由于 d = s - x, g^T d = g^T (s - x) = -gap
            # 所以 gamma = gap / (d^T Q d)
            denom = np.dot(d, self.Q @ d) # d^T Q d
            if denom > 0: # 确保 denom 不是0或负（Q是PSD，所以 d^T Q d >= 0）
                gamma = - np.dot(d, self.Q @ x + self.c) / denom
                gamma = np.clip(gamma, 0, 1)  # 确保步长在 [0, 1] 区间内 (Frank-Wolfe的标准步长约束)
            else:
                # 如果 d^T Q d <= 0 (例如，如果 d 在 Q 的零空间中，或者数值误差)
                # 论文和标准实践中，若分母为0或目标函数沿d方向是线性的，
                # 不同的步长规则会被使用，这里简化为 gamma = 2.0 / (k + 2.0) 或其他递减步长
                # 或者，如果目标函数是线性的 (Q=0)，则gamma=1。
                # 此处代码选择 gamma = 1.0，这意味着如果 d^T Q d <=0，则直接跳到顶点 s。
                # 注释中提到 `gamma = 1.0` if denom is zero. This seems to be a specific choice.
                # A common alternative is gamma_k = 2 / (k+2) for general convex functions.
                # For QPs, the line search gamma = gap / (d_k.T @ Q @ d_k) clipped to [0,1] is common.
                # The code uses `gamma = 1.0` if denom <= 0, which is a simplification.

                # If the quadratic term is zero, we fall back to a step size of 1.
                gamma = 1.0
            
            # Update x
            # 步骤 4: 更新 x_{k+1} = x_k + gamma * d_k
            x = x + gamma * d
        print(k) # 打印实际迭代次数
        return x, history
