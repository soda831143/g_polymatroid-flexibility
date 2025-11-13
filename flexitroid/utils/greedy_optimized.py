# -*- coding: utf-8 -*-
"""
优化的贪心算法实现
使用lifted base polyhedron技术，提升性能并支持负值
"""

import numpy as np
from typing import Set, Callable, Optional


def greedy_optimized(
    c: np.ndarray,
    b_func: Callable[[Set[int]], float],
    p_func: Callable[[Set[int]], float],
    T: int
) -> np.ndarray:
    """
    优化的贪心算法 - 使用lifted base polyhedron技术
    
    关键改进：
    1. 通过添加虚拟时间步t*处理负值
    2. 每次迭代只需调用一次扩展函数b*(S)
    3. 增量构造：v[k] = b*(S_k) - b*(S_{k-1})
    
    Args:
        c: 成本向量 (T,)
        b_func: 子模函数 b(A) - 上界
        p_func: 超模函数 p(A) - 下界  
        T: 时间步数
    
    Returns:
        u: 最优解 (T,)
    
    复杂度: O(T log T + T * Cost(b*))
    
    理论基础:
        使用lifted base polyhedron扩展:
        对于集合A，定义 b*(A):
          - 如果 t* ∉ A: b*(A) = b(A)
          - 如果 t* ∈ A: b*(A) = -p(T \ A)
        
        这样可以处理负值，因为：
        Σu(t) = b*({0,...,T-1}) - b*({0,...,T-1,t*}) 
              = b(T) - (-p(∅))
              = b(T) + p(∅)  # 可以是负数
    """
    def b_star(A: Set[int]) -> float:
        """扩展集合函数 b*(A)"""
        if T in A:  # t* ∈ A
            T_set = set(range(T))
            A_without_tstar = A - {T}
            A_complement = T_set - A_without_tstar
            return -p_func(A_complement)
        else:
            return b_func(A)
    
    # 1. 扩展成本向量（添加虚拟维度t*，成本为0）
    c_star = np.append(c, 0)
    
    # 2. 按成本非递减排序
    pi = np.argsort(c_star)
    
    # 3. 增量构造解
    v = np.zeros(T + 1)
    S_k = set()
    b_prev = 0.0
    
    for k in pi:
        S_k.add(int(k))
        b_curr = b_star(S_k)  # 只需调用一次！
        v[k] = b_curr - b_prev
        b_prev = b_curr
    
    # 4. 投影回原空间（去掉虚拟维度）
    return v[:-1]


def greedy_with_constraints(
    c: np.ndarray,
    b_func: Callable[[Set[int]], float],
    p_func: Callable[[Set[int]], float],
    T: int,
    A_extra: Optional[np.ndarray] = None,
    b_extra: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    带额外约束的贪心算法
    
    求解:
        min c^T u
        s.t. p(A) ≤ Σ_{t∈A} u(t) ≤ b(A), ∀A⊆T  (g-polymatroid约束)
             A_extra @ u ≤ b_extra               (额外线性约束)
    
    当有额外约束时，贪心算法不再是最优的，需要使用Dantzig-Wolfe分解
    这里先提供基础版本（忽略额外约束），后续可扩展
    
    Args:
        c: 成本向量
        b_func, p_func: g-polymatroid函数
        T: 时间步数
        A_extra, b_extra: 额外约束（可选）
    
    Returns:
        u: 近似最优解
    """
    if A_extra is None or b_extra is None:
        # 无额外约束，使用标准贪心
        return greedy_optimized(c, b_func, p_func, T)
    else:
        # 有额外约束，需要更复杂的方法
        # TODO: 实现Dantzig-Wolfe分解
        raise NotImplementedError(
            "带额外约束的优化需要Dantzig-Wolfe分解，请使用vertex_decomposition模块"
        )


def split_into_consecutive_ranges(input_set: Set[int]) -> list:
    """
    将离散集合拆分为连续区间
    
    用于b/p函数的高效计算，避免重复遍历
    
    Example:
        {0,1,2,7,8,9,15} -> [(0,2), (7,9), (15,15)]
        
        时间轴:  0---1---2---3---4---5---6---7---8---9---10
        集合:    [======]                   [====]   ↑
                 区间1         间隙          区间2   单点
    
    Args:
        input_set: 时间步索引集合
    
    Returns:
        List of (start, end) tuples 表示连续区间
    """
    if len(input_set) == 0:
        return []
    
    sorted_list = sorted(input_set)
    result = []
    start = sorted_list[0]
    
    for i in range(1, len(sorted_list)):
        # 检查是否连续
        if sorted_list[i] != sorted_list[i - 1] + 1:
            # 发现间隙，保存当前区间
            if start == sorted_list[i - 1]:
                result.append((start, start))  # 单点
            else:
                result.append((start, sorted_list[i - 1]))  # 区间
            start = sorted_list[i]
    
    # 添加最后一个区间
    if start == sorted_list[-1]:
        result.append((start, start))
    else:
        result.append((start, sorted_list[-1]))
    
    return result


# ============ 测试和验证 ============

def test_greedy_optimization():
    """测试优化的贪心算法"""
    import time
    
    T = 24
    
    # 模拟简单的b和p函数
    def b_simple(A: Set[int]) -> float:
        if not A:
            return 0
        return len(A) * 10.0
    
    def p_simple(A: Set[int]) -> float:
        if not A:
            return 0
        return len(A) * (-2.0)
    
    c = np.random.randn(T)
    
    # 测试优化版本
    start = time.perf_counter()
    u_opt = greedy_optimized(c, b_simple, p_simple, T)
    time_opt = time.perf_counter() - start
    
    print(f"优化版本耗时: {time_opt*1000:.3f} ms")
    print(f"解的范围: [{u_opt.min():.2f}, {u_opt.max():.2f}]")
    print(f"目标值: {c @ u_opt:.2f}")
    
    return u_opt


def test_interval_splitting():
    """测试区间分割"""
    test_cases = [
        ({0,1,2,7,8,9,15}, [(0,2), (7,9), (15,15)]),
        ({0,5,10}, [(0,0), (5,5), (10,10)]),
        ({0,1,2,3}, [(0,3)]),
        (set(), []),
    ]
    
    print("\n区间分割测试:")
    for input_set, expected in test_cases:
        result = split_into_consecutive_ranges(input_set)
        status = "✓" if result == expected else "✗"
        print(f"{status} {input_set} -> {result}")


if __name__ == "__main__":
    print("=" * 60)
    print("优化贪心算法测试")
    print("=" * 60)
    
    test_greedy_optimization()
    test_interval_splitting()
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
