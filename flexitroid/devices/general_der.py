"""Core aggregation module for DER flexibility sets.

This module implements the DER flexibility model and aggregation framework,
including individual flexibility sets and their Minkowski sums.
定义了一个通用的DER参数数据类 DERParameters，包含功率上下限 (u_min, u_max) 和状态上下限 (x_min, x_max)。
实现了 GeneralDER 类，它继承自 Flexitroid，代表单个通用DER的灵活性集合，由功率和能量约束定义。
__init__ 方法使用 DERParameters 初始化。
A_b 属性用于生成描述DER灵活性的H表示（约束矩阵A和向量b）。
b(A) 和 p(A) 方法实现了计算g-polymatroid的子模和超模函数的具体递归公式，
这与Mukhi第一篇论文中Lemma 1和Corollary 1描述的计算方法相符。
example() 类方法用于创建一个具有典型功率和能量约束的示例DER。
"""

from dataclasses import dataclass  # 导入 dataclass，用于方便地创建主要用于存储数据的类。
from typing import Set # 导入 Set 类型提示，用于指示参数 A 是一个集合。
import numpy as np # 导入 NumPy 库，用于高效的数值计算，特别是数组操作。
from . import parameter_sampling as sample # 从当前包（devices）导入 parameter_sampling.py 文件，并赋予别名 sample。
from flexitroid.flexitroid import Flexitroid # 从 flexitroid 包的 flexitroid.py 文件导入 Flexitroid 抽象基类。

# 尝试导入 Cython 加速版本
try:
    from flexitroid.cython.b_fast import b_fast
    from flexitroid.cython.p_fast import p_fast
    USE_CYTHON = True
    print("[Cython] 成功加载 b_fast 和 p_fast，使用加速版本")
except ImportError:
    USE_CYTHON = False
    print("[Cython] 未找到编译的 Cython 模块，使用纯 Python 版本")


@dataclass
class DERParameters:
    """Parameters defining a DER's flexibility.

    Args:
        u_min: Lower bound on power consumption for each timestep.
        u_max: Upper bound on power consumption for each timestep.
        x_min: Lower bound on state of charge for each timestep.
        x_max: Upper bound on state of charge for each timestep.
    """
# DERParameters 类的文档字符串，描述了其用途和参数。
# 定义了四个属性：u_min、u_max、x_min 和 x_max，它们都是 NumPy 数组。
# 这些属性分别表示DER在每个时间步的功率下限、功率上限、状态下限和状态上限。
    u_min: np.ndarray
    u_max: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray

# 定义了两个方法：__str__ 和 __post_init__。
# __str__ 方法返回一个字符串，表示DER的参数。
# __post_init__ 方法在对象初始化后执行，用于验证参数的维度约束和约束条件。
    def __str__(self):
        return 'sss'# 一个简单的 __str__ 方法，目前返回固定字符串 'sss'。在实际应用中，它通常会返回更有意义的设备参数摘要。

    def __post_init__(self):
        """Validate parameter dimensions and constraints."""
        # __post_init__ 方法在 __init__ 方法被 dataclass 自动生成并调用后执行。
        # 用于进行参数的验证。
        # 首先获取时间步数 T，即 u_min 数组的长度。
        T = len(self.u_min)
        # 然后检查 u_max、x_min 和 x_max 数组的长度是否与 T 一致。
        # 如果不一致，则抛出 AssertionError 异常，提示功率或状态约束的维度不正确。
        assert len(self.u_max) == T, "Power bounds must have same length"
        assert len(self.x_min) == T, "SoC bounds must have same length"
        assert len(self.x_max) == T, "SoC bounds must have same length"
        assert np.all(self.u_min <= self.u_max), "Invalid power bounds"
        assert np.all(self.x_min <= self.x_max), "Invalid SoC bounds"


class GeneralDER(Flexitroid):
    """General DER flexibility set representation.

    This class implements the individual flexibility set F(ξᵢ) for a single DER,
    defined by power and energy constraints.
    """
# GeneralDER 类的文档字符串，说明此类代表由功率和能量约束定义的单个DER的个体灵活性集合 F(ξᵢ)。
    def __init__(self, params: DERParameters):
        """Initialize the flexibility set.

        Args:
            params: DER parameters defining power and energy constraints.
        """
        # 将传入的 DERParameters 对象赋值给类的 params 属性。
        self.params = params
        # 获取时间步数 T，即 u_min 数组的长度。
        self._T = len(params.u_min)
        # 将所有时间步设置为活动状态。
        # 使用 set(range(self.T)) 创建一个包含所有时间步的集合，并赋值给 active 属性。
        self.active = set(range(self.T))

    # 定义了一个属性 T，用于获取时间步数。
    # 使用 @property 装饰器将 T 定义为一个属性，而不是方法。
    # 当访问 T 属性时，会调用 get_T 方法，并返回 _T 属性的值。
    @property
    def T(self) -> int:
        return self._T

    @property
    def A_b(self) -> np.ndarray:
        # 这个属性返回DER灵活性的H表示法，即不等式约束 Ax <= b。
        # -np.eye(self.T) 对应 -u(t) <= -u_min(t)  => u(t) >= u_min(t)
        # np.eye(self.T) 对应 u(t) <= u_max(t)
        # -np.tri(self.T) 对应 -x(t) <= -x_min(t) => sum_{k=0 to t} u(k) >= x_min(t) (因为 x(t) = sum u(k) 假设 x(0)=0, δ=1)
        # np.tri(self.T) 对应 x(t) <= x_max(t) => sum_{k=0 to t} u(k) <= x_max(t)
        # 使用 np.vstack 将这些约束堆叠成一个矩阵 A，每一行对应一个约束。
        # 使用 np.concatenate 将这些约束的右侧值 b 连接成一个向量。
        # 最后，只保留有定义的约束（即 b 中不为无穷大），并返回 A 和 b。
        A = np.vstack(
            [-np.eye(self.T), np.eye(self.T), -np.tri(self.T), np.tri(self.T)]
        )
        b = np.concatenate(
            [
                -self.params.u_min,
                self.params.u_max,
                -self.params.x_min,
                self.params.x_max,
            ]
        )
        A = A[np.isfinite(b)]
        b = b[np.isfinite(b)]
        return A, b


    def b(self, A: Set[int]) -> float:
        # 计算子模函数 b(A)。这个函数对应论文中定义的 b^T(A)，
        # 它是在考虑了所有时间步的功率和SoC约束后，集合 A 内的总消耗量的上界。
        # 这个实现通过迭代地加入每个时间步的SoC约束来计算最终的 b(A)。
        # 它基于论文 Lemma 1 [cite: 142] 和 Corollary 1 [cite: 158] 的递归思想，
        # 特别是利用 Theorem 1 (Intersection Theorem) [cite: 129] 将 SoC 约束 (plank K^t [cite: 138])
        # 逐个与当前的多面体求交集来更新 b 函数。
        # 详细的推导和对应关系见于论文的附录A和D [cite: 284, 313]。

        # 【Cython加速】如果可用，使用编译的C版本（10-100倍加速）
        if USE_CYTHON:
            return b_fast(A, self.T, self.active, 
                         self.params.u_min, self.params.u_max,
                         self.params.x_min, self.params.x_max)

        # 首先计算 A 的补集 A_c，即所有时间步的集合减去 A。即 T \ A。
        A_c = self.active - A
        # 初始化 b 为 A 内的总消耗量的上界，即 u_max(A)。
        # 初始化 b_val 为在集合 A 内只考虑功率上限时的最大消耗量。
        # 这相当于递归定义中的 b^0(A) = sum_{t in A} u_max(t)。
        b = np.sum(self.params.u_max[list(A)])
        # 初始化 p_c 为 A_c 内的总消耗量的下界，即 u_min(A_c)。
        # 初始化 p_val_Ac 为在集合 A_c 内只考虑功率下限时的最小消耗量 (或最大发电量)。
        # 这相当于递归定义中的 p^0(A_c) = sum_{t in A_c} u_min(t)。
        p_c = np.sum(self.params.u_min[list(A_c)])
        # 初始化一个空集合 t_set，用于跟踪当前考虑的时间步。用于累积已经处理过SoC约束的时间步。
        t_set = set()
        for t in range(self.T):# 迭代所有时间步 t_loop_idx (0 到 T-1)。
            t_set.add(t) # 将当前时间步加入 t_set。
            # 更新 b_val：
            # b_val 保留上一轮迭代的值（或初始值）。
            # 第二项是根据 Theorem 1 [cite: 129] (Intersection Theorem) 和论文附录D的推导 [cite: 313]，
            # 当加入 t_loop_idx 处的SoC上限约束 x_max[t_loop_idx] 时，对 b(A) 的更新。
            # self.params.x_max[t_loop_idx] 是 x̄(s)
            # p_val_Ac 是 p^{s-1}(A')
            # np.sum(self.params.u_min[list(A_c - t_set)]) 是 u(A' \ [s]) (校正项)
            # np.sum(self.params.u_max[list(A - t_set)]) 是 u(A \ [s]) (校正项)
            b = np.min(
                [
                    b, # 上一轮的 b(A)
                    self.params.x_max[t] # 当前时刻 t 的 SoC 上限 x̄(t)
                    - p_c  # A的补集 A_c 在上一轮的 p 值: p_{s-1}(A')
                    # 以下两项是校正因子，因为 p_val_Ac 和 b_val 是基于整个 A 和 A_c 计算的，
                      # 而SoC约束是关于到时刻 t 的累积量。
                    + np.sum(self.params.u_min[list(A_c - t_set)])# (T \ A) \ {0,...,t} 上的 u_min 之和
                    + np.sum(self.params.u_max[list(A - t_set)]),# A \ {0,...,t} 上的 u_max 之和
                ]
            )
            # 更新 p_val_Ac（A的补集的p值）：
            # 当加入 t_loop_idx 处的SoC下限约束 x_min[t_loop_idx] 时，对 p(A_c) 的更新。
            # 注意，这里用的是更新后的 b_val。
            p_c = np.max(
                [
                    p_c,
                    self.params.x_min[t]
                    - b
                    + np.sum(self.params.u_max[list(A - t_set)])# A \ {0,...,t} 上的 u_max 之和
                    + np.sum(self.params.u_min[list(A_c - t_set)]),# (T \ A) \ {0,...,t} 上的 u_min 之和
                ]
            )
        return b


    def p(self, A: Set[int]) -> float:
        # 计算超模函数 p(A)。这个函数对应论文中定义的 p^T(A)，
        # 它是在考虑了所有时间步的功率和SoC约束后，集合 A 内的总消耗量的下界。
        # 实现逻辑与 b(A) 非常相似，只是 min/max 和约束上下限的角色互换。
        # 同样基于论文 Lemma 1[cite: 142], Corollary 1 [cite: 158] 及附录的推导 [cite: 284, 313]。
        
        # 【Cython加速】如果可用，使用编译的C版本（10-100倍加速）
        if USE_CYTHON:
            return p_fast(A, self.T, self.active,
                         self.params.u_min, self.params.u_max,
                         self.params.x_min, self.params.x_max)
        
        A_c = self.active - A
        # 初始化 p_val 为在集合 A 内只考虑功率下限时的最小消耗量。
        # 这相当于递归定义中的 p^0(A) = sum_{t in A} u_min(t)。
        p = np.sum(self.params.u_min[list(A)])
        # 初始化 b_val_Ac 为在集合 A_c 内只考虑功率上限时的最大消耗量。
        # 这相当于递归定义中的 b^0(A_c) = sum_{t in A_c} u_max(t)。
        b_c = np.sum(self.params.u_max[list(A_c)])
        # 初始化一个空集合 t_set，用于跟踪当前考虑的时间步。用于累积已经处理过SoC约束的时间步。
        t_set = set()
        for t in range(self.T):# 迭代所有时间步 t_loop_idx (0 到 T-1)。
            t_set.add(t)
            # 更新 p_val：
            # p_val 保留上一轮迭代的值（或初始值）。
            # 第二项是根据 Theorem 1 [cite: 129] (Intersection Theorem) 和论文附录D的推导 [cite: 313]，
            # 当加入 t_loop_idx 处的SoC下限约束 x_min[t_loop_idx] 时，对 p(A) 的更新。
            # self.params.x_min[t_loop_idx] 是 x̲(t)
            # b_val_Ac 是 b^{s-1}(A')
            # np.sum(self.params.u_max[list(A_c - t_set)]) 是 u(A' \ [s]) (校正项)
            p = np.max(
                [
                    p,
                    self.params.x_min[t] # 当前时刻 t 的 SoC 下限 x̲(t)
                    - b_c # A的补集 A_c 在上一轮的 b 值: b_{s-1}(A')
                    + np.sum(self.params.u_max[list(A_c - t_set)])
                    + np.sum(self.params.u_min[list(A - t_set)]),
                ]
            )
            # 更新 b_val_Ac（A的补集的b值）：
            # 当加入 t_loop_idx 处的SoC上限约束 x_max[t_loop_idx] 时，对 b(A_c) 的更新。
            # 注意，这里用的是更新后的 p_val。
            b_c = np.min(
                [
                    b_c,
                    self.params.x_max[t]
                    - p
                    + np.sum(self.params.u_min[list(A - t_set)])
                    + np.sum(self.params.u_max[list(A_c - t_set)]),
                ]
            )
        return p

    # def b(self, A: Set[int]) -> float:
    #     """Compute submodular function b for the g-polymatroid representation.

    #     Args:
    #         A: Subset of the ground set T.

    #     Returns:
    #         Value of b(A) as defined by the recursive formula.
    #     """
    #     if not A:
    #         return 0.0

    #     t_max = max(A)
    #     T_t = set(range(t_max + 1))
    #     A_c = T_t - A
    #     b = np.sum(self.params.u_max[list(A)])
    #     p_c = np.sum(self.params.u_min[list(A_c)])
    #     t_set = set()
    #     for t in range(t_max):
    #         t_set.add(t)
    #         b = np.min(
    #             [
    #                 b,
    #                 self.params.x_max[t]
    #                 - p_c
    #                 + np.sum(self.params.u_min[list(A_c - t_set)])
    #                 + np.sum(self.params.u_max[list(A - t_set)]),
    #             ]
    #         )
    #         p_c = np.max(
    #             [
    #                 p_c,
    #                 self.params.x_min[t]
    #                 - b
    #                 + np.sum(self.params.u_max[list(A - t_set)])
    #                 + np.sum(self.params.u_min[list(A_c - t_set)]),
    #             ]
    #         )
    #     return b

    # def p(self, A: Set[int]) -> float:
    #     """Compute supermodular function p for the g-polymatroid representation.

    #     Args:
    #         A: Subset of the ground set T.

    #     Returns:
    #         Value of p(A) as defined by the recursive formula.
    #     """
    #     if not A:
    #         return 0.0

    #     t_max = max(A)
    #     T_t = set(range(t_max + 1))
    #     A_c = T_t - A

    #     p = np.sum(self.params.u_min[list(A)])
    #     b_c = np.sum(self.params.u_max[list(A_c)])
    #     t_set = set()

    #     for t in range(t_max):
    #         t_set.add(t)
    #         p = np.max(
    #             [
    #                 p,
    #                 self.params.x_min[t]
    #                 - b_c
    #                 + np.sum(self.params.u_max[list(A_c - t_set)])
    #                 + np.sum(self.params.u_min[list(A - t_set)]),
    #             ]
    #         )
    #         b_c = np.min(
    #             [
    #                 b_c,
    #                 self.params.x_max[t]
    #                 - p
    #                 + np.sum(self.params.u_min[list(A - t_set)])
    #                 + np.sum(self.params.u_max[list(A_c - t_set)]),
    #             ]
    #         )
    #     return p

    @classmethod# @classmethod 装饰器表示这是一个类方法，可以通过类名直接调用，例如 GeneralDER.example()。
    def example(cls, T: int = 24) -> "GeneralDER":
        """Create an example DER with typical power and energy constraints.

        Creates a DER with:
        - Bidirectional power flow (-2kW to 2kW)
        - Energy storage capacity of 10kWh
        - Must maintain state of charge between 20% and 80%

        Args:
            T: Number of timesteps (default 24 for hourly resolution)

        Returns:
            GeneralDER instance with example parameters
        """
        # example 类方法的文档字符串，描述了其功能和创建的DER实例的大致特性。
        # 调用 parameter_sampling.py 中的 der 函数来生成随机的参数。
        #
        u_min, u_max, x_min, x_max = sample.der(T)
# 使用生成的随机参数创建 DERParameters 对象。
        params = DERParameters(u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)
        # 使用创建的参数对象实例化并返回一个 GeneralDER 对象。
        # cls 指代当前的类 GeneralDER。
        return cls(params)
