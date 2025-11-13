# -*- coding: utf-8 -*-
"""
顶点分解算法 (Vertex Decomposition)
基于Dantzig-Wolfe列生成方法

理论基础:
    g-polymatroid的任意点都可以表示为顶点的凸组合:
        u = Σ λ_j * v_j
    其中 Σλ_j = 1, λ_j ≥ 0
    
    顶点v_j是g-polymatroid极点，可通过贪心算法获得:
        v_j = greedy(c_j)
    
disaggregation原理:
    如果聚合解 u_agg = Σ λ_j * v_j^agg
    那么设备i的解为: u_i = Σ λ_j * v_j^i
    其中 v_j^i = device_i.greedy(c_j)
    
    关键: 所有设备使用相同的权重λ和成本向量c
"""

import numpy as np
import cvxpy as cp
from typing import List, Tuple, Set, Callable, Optional
from flexitroid.utils.greedy_optimized import greedy_optimized


class VertexDecomposer:
    """
    顶点分解器 - 基于Dantzig-Wolfe分解
    
    功能:
    1. 将聚合信号分解为g-polymatroid顶点的凸组合
    2. 使用相同的凸组合权重分解个体设备控制
    
    算法:
    - 主问题: 找到凸组合权重λ
    - 子问题: 贪心算法寻找新顶点
    - 列生成: 迭代添加改进顶点直到收敛
    """
    
    def __init__(
        self,
        b_func: Callable[[Set[int]], float],
        p_func: Callable[[Set[int]], float],
        T: int,
        epsilon: float = 1e-6,
        max_iter: int = 1000
    ):
        """
        Args:
            b_func: 聚合的b函数
            p_func: 聚合的p函数
            T: 时间步数
            epsilon: 收敛容差
            max_iter: 最大迭代次数
        """
        self.b_func = b_func
        self.p_func = p_func
        self.T = T
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def decompose(
        self,
        u_target: np.ndarray,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        将目标信号分解为顶点的凸组合
        
        求解:
            min ||Σ λ_j * v_j - u_target||²
            s.t. Σ λ_j = 1
                 A @ (Σ λ_j * v_j) ≤ b  (如果提供)
                 λ_j ≥ 0
        
        Args:
            u_target: 目标聚合信号 (T,)
            A, b: 额外线性约束（可选）
        
        Returns:
            lmda: 凸组合权重 (num_vertices,)
            vertices: 顶点列表，每个顶点是(T,)数组
        """
        # 1. 初始化顶点集
        V_subset = self._form_initial_vertices()
        
        # 2. 列生成迭代
        for iteration in range(self.max_iter):
            # 2.1 求解限制主问题
            lmda, dual_vars = self._solve_master_problem(
                V_subset, u_target, A, b
            )
            
            # 2.2 定价子问题（贪心算法）
            new_vertex, reduced_cost = self._pricing_problem(
                u_target, dual_vars, A
            )
            
            # 2.3 检查收敛
            if reduced_cost > -self.epsilon:
                print(f"[顶点分解] 收敛于迭代 {iteration+1}, "
                      f"使用 {len(V_subset)} 个顶点")
                break
            
            # 2.4 添加新顶点
            V_subset.append(new_vertex)
        else:
            print(f"[顶点分解] 警告: 达到最大迭代次数 {self.max_iter}")
        
        return lmda, V_subset
    
    def _form_initial_vertices(self) -> List[np.ndarray]:
        """
        形成初始顶点集 (Box顶点)
        
        使用标准方向：e_t 和 -e_t
        """
        T = self.T
        V = []
        
        # 正方向
        for t in range(T):
            c = np.zeros(T)
            c[t] = 1.0
            v = greedy_optimized(c, self.b_func, self.p_func, T)
            V.append(v)
        
        # 负方向
        for t in range(T):
            c = np.zeros(T)
            c[t] = -1.0
            v = greedy_optimized(c, self.b_func, self.p_func, T)
            V.append(v)
        
        return V
    
    def _solve_master_problem(
        self,
        V_subset: List[np.ndarray],
        u_target: np.ndarray,
        A: Optional[np.ndarray],
        b: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, dict]:
        """
        求解限制主问题
        
        Returns:
            lmda: 凸组合权重
            dual_vars: 对偶变量（用于定价）
        """
        num_vertices = len(V_subset)
        V_matrix = np.array(V_subset).T  # (T, num_vertices)
        
        # 决策变量
        lmda = cp.Variable(num_vertices, nonneg=True)
        
        # 目标函数: min ||V @ lmda - u_target||²
        obj = cp.Minimize(cp.sum_squares(V_matrix @ lmda - u_target))
        
        # 约束
        constraints = [cp.sum(lmda) == 1]
        
        if A is not None and b is not None:
            # 额外约束: A @ V @ lmda ≤ b
            constraints.append(A @ V_matrix @ lmda <= b)
        
        # 求解
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CLARABEL)
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"主问题求解失败: {prob.status}")
        
        # 提取对偶变量
        dual_vars = {
            "convexity": constraints[0].dual_value,  # 凸性约束的对偶
        }
        
        if A is not None:
            dual_vars["extra"] = constraints[1].dual_value
        
        return lmda.value, dual_vars
    
    def _pricing_problem(
        self,
        u_target: np.ndarray,
        dual_vars: dict,
        A: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """
        定价子问题: 寻找改进方向
        
        求解:
            min c^T v
            s.t. v ∈ g-polymatroid
        
        其中 c = -u_target - A^T @ y（从对偶得到）
        
        Returns:
            new_vertex: 新顶点
            reduced_cost: 简化成本（负值表示有改进）
        """
        # 构造简化成本向量
        c = -u_target  # 基础成本
        
        if A is not None and "extra" in dual_vars:
            y = dual_vars["extra"]
            c = c - A.T @ y
        
        # 贪心求解
        new_vertex = greedy_optimized(c, self.b_func, self.p_func, self.T)
        
        # 计算简化成本
        reduced_cost = c @ new_vertex - dual_vars["convexity"]
        
        return new_vertex, reduced_cost
    
    def disaggregate(
        self,
        u_agg_target: np.ndarray,
        device_list: List,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        将聚合信号分解为个体设备控制
        
        关键思想:
            如果 u_agg = Σ λ_j * v_j^agg
            那么 u_i = Σ λ_j * v_j^i
            其中所有设备使用相同的λ和成本向量
        
        Args:
            u_agg_target: 目标聚合信号 (T,)
            device_list: 设备列表（每个有b和p函数）
            A, b: 额外约束（可选）
        
        Returns:
            u_individual: (N, T) 每个设备的控制
        """
        # 1. 求解顶点分解
        print("[顶点分解] 分解聚合信号...")
        lmda, cost_vectors = self.decompose(u_agg_target, A, b)
        
        # 注意: cost_vectors实际上存储的是顶点v_j，不是成本向量c_j
        # 我们需要从顶点反推成本向量（通过贪心算法的逆过程）
        # 但实际上，我们可以直接使用这些顶点
        
        # 2. 对每个设备分解
        print(f"[顶点分解] 分解到 {len(device_list)} 个设备...")
        u_individual = []
        
        for i, device in enumerate(device_list):
            u_i = np.zeros(self.T)
            
            # 对每个顶点，计算设备i的贡献
            for j, (weight, vertex_agg) in enumerate(zip(lmda, cost_vectors)):
                # 问题: 如何从聚合顶点得到个体顶点？
                # 解决: 需要知道生成vertex_agg的成本向量c_j
                # 但我们可以用近似: 假设个体设备也在相同方向
                
                # TODO: 这里需要改进 - 需要存储生成顶点的成本向量
                # 临时方案: 使用比例分配
                u_i += weight * vertex_agg / len(device_list)
            
            u_individual.append(u_i)
        
        return np.array(u_individual)


class VertexDecomposerWithDevices:
    """
    改进的顶点分解器 - 直接处理设备列表
    
    这个版本在分解时同时跟踪个体设备的顶点，避免近似
    """
    
    def __init__(
        self,
        device_list: List,
        T: int,
        epsilon: float = 1e-6,
        max_iter: int = 1000
    ):
        """
        Args:
            device_list: 设备列表（每个有b和p函数）
            T: 时间步数
            epsilon: 收敛容差
            max_iter: 最大迭代次数
        """
        self.device_list = device_list
        self.T = T
        self.epsilon = epsilon
        self.max_iter = max_iter
        
        # 聚合的b和p函数
        self.b_agg = lambda A: sum(device.b(A) for device in device_list)
        self.p_agg = lambda A: sum(device.p(A) for device in device_list)
    
    def disaggregate(
        self,
        u_agg_target: np.ndarray,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        完整的disaggregation流程
        
        Args:
            u_agg_target: 目标聚合信号 (T,)
            A, b: 额外约束
        
        Returns:
            u_individual: (N, T) 每个设备的控制
        """
        # 1. 初始化顶点集（存储成本向量）
        cost_vectors = self._form_initial_cost_vectors()
        
        # 2. 列生成
        for iteration in range(self.max_iter):
            # 2.1 计算所有设备在当前顶点集的解
            V_agg, V_devices = self._compute_vertices(cost_vectors)
            
            # 2.2 求解主问题
            lmda, dual_vars = self._solve_master(
                V_agg, u_agg_target, A, b
            )
            
            # 2.3 定价
            new_cost, reduced_cost = self._pricing(
                u_agg_target, dual_vars, A
            )
            
            # 2.4 检查收敛
            if reduced_cost > -self.epsilon:
                print(f"[Disaggregation] 收敛于迭代 {iteration+1}, "
                      f"使用 {len(cost_vectors)} 个顶点")
                break
            
            # 2.5 添加新方向
            cost_vectors.append(new_cost)
        else:
            print(f"[Disaggregation] 警告: 达到最大迭代次数")
        
        # 3. 最终计算个体设备的解
        V_agg_final, V_devices_final = self._compute_vertices(cost_vectors)
        lmda_final, _ = self._solve_master(
            V_agg_final, u_agg_target, A, b
        )
        
        # 4. 组合个体解
        N = len(self.device_list)
        u_individual = np.zeros((N, self.T))
        
        for i in range(N):
            for j, weight in enumerate(lmda_final):
                u_individual[i] += weight * V_devices_final[i][j]
        
        # 验证
        u_sum = np.sum(u_individual, axis=0)
        error = np.linalg.norm(u_sum - u_agg_target)
        print(f"[Disaggregation] 重构误差: {error:.6f}")
        
        return u_individual
    
    def _form_initial_cost_vectors(self) -> List[np.ndarray]:
        """形成初始成本向量（标准方向）"""
        cost_vectors = []
        
        # 正负标准方向
        for t in range(self.T):
            c = np.zeros(self.T)
            c[t] = 1.0
            cost_vectors.append(c)
            cost_vectors.append(-c)
        
        return cost_vectors
    
    def _compute_vertices(
        self,
        cost_vectors: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        """
        对所有成本向量，计算聚合和个体的顶点
        
        Returns:
            V_agg: 聚合顶点列表
            V_devices: V_devices[i][j] = 设备i在方向j的顶点
        """
        V_agg = []
        V_devices = [[] for _ in self.device_list]
        
        for c in cost_vectors:
            # 聚合顶点
            v_agg = greedy_optimized(c, self.b_agg, self.p_agg, self.T)
            V_agg.append(v_agg)
            
            # 个体顶点
            for i, device in enumerate(self.device_list):
                v_i = greedy_optimized(c, device.b, device.p, self.T)
                V_devices[i].append(v_i)
        
        return V_agg, V_devices
    
    def _solve_master(
        self,
        V_agg: List[np.ndarray],
        u_target: np.ndarray,
        A: Optional[np.ndarray],
        b: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, dict]:
        """求解限制主问题"""
        num_vertices = len(V_agg)
        V_matrix = np.array(V_agg).T
        
        lmda = cp.Variable(num_vertices, nonneg=True)
        obj = cp.Minimize(cp.sum_squares(V_matrix @ lmda - u_target))
        
        constraints = [cp.sum(lmda) == 1]
        if A is not None and b is not None:
            constraints.append(A @ V_matrix @ lmda <= b)
        
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CLARABEL)
        
        dual_vars = {"convexity": constraints[0].dual_value}
        if A is not None:
            dual_vars["extra"] = constraints[1].dual_value
        
        return lmda.value, dual_vars
    
    def _pricing(
        self,
        u_target: np.ndarray,
        dual_vars: dict,
        A: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """定价子问题"""
        c = -u_target
        if A is not None and "extra" in dual_vars:
            c = c - A.T @ dual_vars["extra"]
        
        new_vertex = greedy_optimized(c, self.b_agg, self.p_agg, self.T)
        reduced_cost = c @ new_vertex - dual_vars["convexity"]
        
        return c, reduced_cost


# ============ 测试 ============

def test_vertex_decomposition():
    """测试顶点分解"""
    print("=" * 60)
    print("顶点分解测试")
    print("=" * 60)
    
    T = 8
    
    # 模拟简单设备
    def b_simple(A: Set[int]) -> float:
        return len(A) * 5.0 if A else 0
    
    def p_simple(A: Set[int]) -> float:
        return len(A) * (-1.0) if A else 0
    
    # 创建分解器
    decomposer = VertexDecomposer(b_simple, p_simple, T, epsilon=1e-4)
    
    # 目标信号
    u_target = np.array([2, 3, 4, 3, 2, 1, 0, -1])
    
    # 分解
    lmda, vertices = decomposer.decompose(u_target)
    
    # 重构
    u_recon = sum(l * v for l, v in zip(lmda, vertices))
    error = np.linalg.norm(u_recon - u_target)
    
    print(f"\n使用顶点数: {len(vertices)}")
    print(f"重构误差: {error:.6f}")
    print(f"λ权重范围: [{lmda.min():.4f}, {lmda.max():.4f}]")
    print(f"非零权重数: {np.sum(lmda > 1e-6)}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_vertex_decomposition()
