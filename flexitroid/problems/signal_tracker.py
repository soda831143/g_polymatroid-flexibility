"""
Signal Tracker - Dantzig-Wolfe 分解求解器
用于顶点分解(Vertex Disaggregation)

参考: flexitroid-benchmark/flexitroid/problems/signal_tracker.py
原理: 使用Frank-Wolfe迭代找到目标信号的凸组合表示

应用场景:
1. 将聚合信号u_agg分解为个体信号u_i，使得Σu_i = u_agg
2. 支持异构TCL（不同的a_i, δ_i参数）
3. 保证每个u_i在各自的可行域内
"""
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("警告: Gurobi未安装，顶点分解功能将不可用")


class SignalTracker:
    """
    信号跟踪器 - 使用Dantzig-Wolfe分解
    
    求解问题:
        min  ||u - signal||²
        s.t. u ∈ F (可行集的凸包)
        
    方法:
        使用Frank-Wolfe迭代 + Dantzig-Wolfe列生成
        u = Σ λ_j · v_j, 其中v_j是贪心算法找到的顶点
    """
    
    def __init__(self, feasible_set, signal, max_iters=10000, epsilon=None):
        """
        Args:
            feasible_set: Flexitroid对象（g-polymatroid）
            signal: 目标信号 (T,) numpy数组
            max_iters: 最大迭代次数
            epsilon: 收敛容差（默认为signal范数的1e-6倍）
        """
        self.signal = signal
        self.T = signal.shape[0]
        self.feasible_set = feasible_set
        self.max_iter = max_iters
        
        # 二次优化问题矩阵
        self.Q = np.eye(self.T)  # 目标函数: ||u - signal||² 的Q矩阵
        self.c = -signal  # 线性项
        
        # 收敛容差
        if epsilon is None:
            self.epsilon = 1e-6 * np.linalg.norm(signal)
        else:
            self.epsilon = epsilon
        
        # 解和历史记录
        self.solution = None
        self.value = None
        self.history = None
        self.V = None  # 顶点矩阵
        self.PI = None  # 对应的梯度向量
        self.lmda = None  # 凸组合系数
    
    def solve(self):
        """求解信号跟踪问题"""
        if self.solution is None:
            self.solution, self.history = self.frank_wolfe()
            self.value = 0.5 * np.dot(self.solution, self.Q @ self.solution) + np.dot(
                self.c, self.solution
            )
        
        # 提取顶点和系数
        self.V = np.array(self.history["V"])
        self.PI = np.array(self.history["c"])
        
        # 求解凸组合系数
        _, lmda = self._check_convergence(self.V)
        self.lmda = lmda if lmda is not None else np.zeros(len(self.V))
        
        return self.solution
    
    def frank_wolfe(self):
        """Frank-Wolfe迭代算法"""
        history = {"obj": [], "gap": [], "V": [], "c": []}
        
        # 初始化：使用梯度方向找第一个顶点
        x = self.feasible_set.solve_linear_program(self.c)
        history["V"].append(x)
        history["c"].append(self.c.copy())
        
        for k in range(self.max_iter):
            # 计算梯度: g = Qx + c
            g = self.Q @ x + self.c
            
            # 在可行集上求解线性规划，找到新顶点
            s = self.feasible_set.solve_linear_program(g)
            
            # 对偶间隙
            gap = g.dot(x - s)
            
            # 记录目标值和间隙
            obj = 0.5 * np.dot(x, self.Q @ x) + np.dot(self.c, x)
            history["obj"].append(obj)
            history["gap"].append(gap)
            history["V"].append(s)
            history["c"].append(g.copy())
            
            # 检查收敛
            if gap < self.epsilon:
                print(f"  [Signal Tracker] Frank-Wolfe收敛 (迭代{k+1}次, gap={gap:.2e})")
                break
            
            # 计算步长
            d = s - x
            denom = np.dot(d, self.Q @ d)
            
            if denom > 0:
                gamma = -np.dot(d, self.Q @ x + self.c) / denom
                gamma = np.clip(gamma, 0, 1)
            else:
                # 如果二次项为0，使用固定步长
                gamma = 1.0
            
            # 更新解
            x = x + gamma * d
            
            # 尝试提前检测收敛（基于凸组合可行性）
            model, lmda = self._check_convergence(history["V"])
            if model is not None and lmda is not None:
                print(f"  [Signal Tracker] 凸组合可行 (迭代{k+1}次)")
                break
        
        if k == self.max_iter - 1:
            print(f"  [Signal Tracker] 达到最大迭代次数 {self.max_iter}")
        
        return x, history
    
    def _check_convergence(self, V):
        """
        检查是否可以用顶点的凸组合表示目标信号
        
        求解:
            min  1
            s.t. Σλ_j·v_j = signal
                 Σλ_j = 1
                 λ_j ≥ 0
        """
        if not GUROBI_AVAILABLE:
            return None, None
        
        V_array = np.array(V)
        n = V_array.shape[0]
        
        try:
            # 创建Gurobi模型
            model = gp.Model("convergence_check")
            model.setParam('OutputFlag', 0)  # 静默模式
            
            # 变量: λ_j ≥ 0
            lmda = model.addVars(n, lb=0.0, name="lambda")
            
            # 约束: Σλ_j = 1
            model.addConstr(gp.quicksum(lmda[j] for j in range(n)) == 1, "sum_to_one")
            
            # 约束: Σλ_j·v_j = signal
            for t in range(self.T):
                model.addConstr(
                    gp.quicksum(lmda[j] * V_array[j, t] for j in range(n)) == self.signal[t],
                    f"signal_{t}"
                )
            
            # 目标函数: min 1 (可行性问题)
            model.setObjective(1, GRB.MINIMIZE)
            
            # 求解
            model.optimize()
            
            if model.Status == GRB.OPTIMAL:
                lmda_val = np.array([lmda[j].X for j in range(n)])
                return model, lmda_val
            else:
                return None, None
                
        except Exception as e:
            # 如果求解失败，返回None
            return None, None
    
    def get_vertices_and_weights(self):
        """
        获取顶点和对应的凸组合系数
        
        Returns:
            vertices: (num_vertices, T) 顶点矩阵
            weights: (num_vertices,) 凸组合系数
        """
        if self.lmda is None:
            raise RuntimeError("必须先调用solve()方法")
        
        # 过滤掉权重为0的顶点
        nonzero_mask = self.lmda > 1e-10
        vertices = self.V[nonzero_mask]
        weights = self.lmda[nonzero_mask]
        
        # 重新归一化（防止数值误差）
        weights = weights / np.sum(weights)
        
        return vertices, weights


def test_signal_tracker():
    """测试信号跟踪器"""
    print("\n" + "="*60)
    print("信号跟踪器测试")
    print("="*60)
    
    from flexitroid.devices.general_der import GeneralDER, DERParameters
    
    T = 24
    
    # 创建简单的DER
    params = DERParameters(
        u_min=np.full(T, -1.0),
        u_max=np.full(T, 2.0),
        x_min=np.linspace(-10, -5, T),
        x_max=np.linspace(20, 40, T)
    )
    der = GeneralDER(params)
    
    # 目标信号
    signal = np.sin(np.linspace(0, 2*np.pi, T))
    
    # 创建跟踪器
    tracker = SignalTracker(der, signal, max_iters=100)
    
    # 求解
    solution = tracker.solve()
    
    # 获取顶点和权重
    vertices, weights = tracker.get_vertices_and_weights()
    
    print(f"\n找到 {len(weights)} 个有效顶点")
    print(f"权重和: {np.sum(weights):.6f}")
    print(f"顶点凸组合误差: {np.linalg.norm(vertices.T @ weights - signal):.2e}")
    print(f"最终解误差: {np.linalg.norm(solution - signal):.2e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    test_signal_tracker()
