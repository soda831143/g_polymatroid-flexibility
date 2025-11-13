"""Aggregator module for DER flexibility sets.

This module implements the aggregation framework for DER flexibility sets,
including the Minkowski sum of individual flexibility sets.
定义了一个通用的聚合器基类 Aggregator，它也继承自 Flexitroid。
它将聚合灵活性集合实现为个体灵活性集合的闵可夫斯基和。
__init__ 方法接收一个设备列表 fleet 进行聚合。
其 b(A) 和 p(A) 方法简单地将fleet中所有设备的相应函数值求和，这完全符合g-polymatroid闵可夫斯基和的计算规则（Mukhi第一篇论文Theorem 3）。
"""

from typing import List, Set, TypeVar, Generic
import numpy as np
from flexitroid.flexitroid import Flexitroid
from flexitroid.problems.signal_tracker import SignalTracker

D = TypeVar("D", bound=Flexitroid)
# 定义一个类型变量 D。
# "bound=Flexitroid" 表示类型 D 必须是 Flexitroid 类或其任何子类。
# 这使得 Aggregator 类可以聚合任何符合 Flexitroid 接口的设备。

class Aggregator(Flexitroid, Generic[D]):
    # 定义 Aggregator 类。
# 它继承自 Flexitroid，意味着 Aggregator 本身也是一个灵活性实体，拥有 b(A), p(A) 方法和 T 属性。
# 它也继承自 Generic[D]，表示这是一个泛型类，可以针对特定类型的 Flexitroid 设备进行参数化。
# 例如，可以创建一个 Aggregator[V1G] 或 Aggregator[GeneralDER]。

    """Generic aggregator for device flexibility sets.

    This class implements the aggregate flexibility set F(Ξₙ) as the Minkowski
    sum of individual flexibility sets, represented as a g-polymatroid.
    """
    # 各个体灵活性集合的闵可夫斯基和，并且这个聚合集合本身也是一个g-polymatroid。


    def __init__(self, fleet: List[D]):
        """Initialize the aggregate flexibility set.

        Args:
            fleet: List of fleet to aggregate.
        """
        # Aggregator 类的构造函数。
    # 参数:
    #   fleet: List[D]，一个包含待聚合设备的列表。列表中的每个设备都必须是 Flexitroid 类型或其子类型。

        if not fleet:
            raise ValueError("Must provide at least one device")
             # 如果列表为空，则抛出 ValueError，因为聚合至少需要一个设备。

        self.fleet = fleet
        # 将传入的设备列表存储为实例属性 self.fleet。
        self._T = fleet[0].T
        # 假设所有设备具有相同的时间范围，所以用第一个设备的时间范围 T 来设置聚合器的时间范围 _T。


        # Validate all fleet have same time horizon
        # 验证设备列表中的所有设备是否都具有相同的时间范围。
        for device in fleet[1:]:
            if device.T != self.T:
                # 如果发现某个设备的时间范围与第一个设备的不同。
                raise ValueError("All fleet must have same time horizon")
                # 抛出 ValueError，表示所有设备必须具有相同的时间范围。

    @property
     # @property 装饰器将一个方法变成一个属性，可以通过 .T 的方式访问。
    def T(self) -> int:
        return self._T
        # 返回聚合器的时间范围，即其所聚合的设备的时间范围。

    def b(self, A: Set[int]) -> float:
        # 实现聚合后的子模函数 b(A)。
    # 这是 Aggregator 类作为 Flexitroid 子类必须实现的方法。
        """Compute aggregate submodular function b.

        Args:
            A: Subset of the ground set T.

        Returns:
            Sum of individual b functions over all fleet.
        """
        return sum(device.b(A) for device in self.fleet)
        # 文档字符串说明该方法计算聚合子模函数 b，其值为群体中所有设备各自 b(A) 函数值之和。
        # 这精确对应论文中的 Theorem 3: $b(A) = \sum_{i \in \mathcal{N}} b_i^T(A)$。
        # 使用一个生成器表达式遍历 self.fleet 中的每个设备 device。
        # 对每个设备调用其自身的 .b(A) 方法。
        # sum() 函数将所有这些单独的 b(A) 值加起来，得到聚合的 b(A) 值。

    def p(self, A: Set[int]) -> float:
         # 实现聚合后的超模函数 p(A)。
    # 这也是 Aggregator 类作为 Flexitroid 子类必须实现的方法。
        """Compute aggregate supermodular function p.

        Args:
            A: Subset of the ground set T.

        Returns:
            Sum of individual p functions over all fleet.
        """
        return sum(device.p(A) for device in self.fleet)
        # 逻辑与 b(A) 方法类似：
        # 遍历 self.fleet 中的每个设备 device。
        # 对每个设备调用其自身的 .p(A) 方法。
        # sum() 函数将所有这些单独的 p(A) 值加起来，得到聚合的 p(A) 值。
    
    def disaggregate(self, signal: np.ndarray) -> np.ndarray:
        """
        将聚合信号分解为个体信号（顶点分解）
        
        使用Dantzig-Wolfe分解方法：
        1. 使用SignalTracker找到signal的凸组合表示: signal = Σλ_j·v_j
        2. 对每个设备i，计算: u_i = Σλ_j·v_ij，其中v_ij = device_i.greedy(c_j)
        3. 保证: Σu_i = signal 且每个u_i在各自的可行域内
        
        应用场景:
        - 支持异构TCL（不同的a_i, δ_i参数）
        - JCC-SRO/Re-SRO算法的逆变换前需要分解
        
        Args:
            signal: 聚合信号 (T,) numpy数组
        
        Returns:
            disaggregation: (N, T) 个体信号矩阵，其中N为设备数量
        """
        print(f"\n  [Disaggregation] 开始分解聚合信号...")
        
        # 1. 使用SignalTracker找到凸组合表示
        tracker = SignalTracker(self, signal, max_iters=1000)
        tracker.solve()
        
        # 2. 获取顶点和权重
        vertices, weights = tracker.get_vertices_and_weights()
        pi = tracker.PI[tracker.lmda > 1e-10]  # 对应的梯度向量
        
        print(f"  [Disaggregation] 找到 {len(weights)} 个有效顶点")
        print(f"  [Disaggregation] 凸组合误差: {np.linalg.norm(vertices.T @ weights - signal):.2e}")
        
        # 3. 为每个设备分配信号
        disaggregation = []
        for i, device in enumerate(self.fleet):
            u_i = np.zeros(self.T)
            
            # 使用相同的λ和c为每个设备构造信号
            for lmda, c in zip(weights, pi):
                # 在设备的可行域上求解线性规划
                vertex_i = device.solve_linear_program(c)
                u_i += lmda * vertex_i
            
            disaggregation.append(u_i)
        
        disaggregation = np.array(disaggregation)
        
        # 验证分解正确性
        u_sum = np.sum(disaggregation, axis=0)
        error = np.linalg.norm(u_sum - signal)
        print(f"  [Disaggregation] 重构误差: {error:.2e}")
        
        if error > 1e-3:
            print(f"  警告: 分解误差较大 ({error:.2e}), 可能需要更多迭代")
        
        return disaggregation
