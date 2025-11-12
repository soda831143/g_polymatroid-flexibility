from abc import ABC, abstractmethod
from typing import List, Set, Optional, TypeVar, Generic
import numpy as np
from itertools import permutations

'''
定义了 Flexitroid 抽象基类，它是一个抽象的基类，定义了所有灵活性集合必须实现的方法。
b(A) 和 p(A) 方法分别计算子模和超模函数，T 属性表示时间步长。
_b_star 方法计算扩展集合函数 b*，用于构建凸包。
solve_linear_program 方法使用贪心算法求解线性规划问题。
form_box 方法构建凸包。
get_all_vertices 方法计算所有顶点。
定义了一个抽象基类 Flexitroid。
这个基类规定了所有可灵活实体（单个DER或聚合体）必须实现的通用接口，
特别是用于g-polymatroid表示的子模函数 b(A) 和超模函数 p(A)，以及时间跨度 T 属性。
还包含了一个 _b_star(A) 方法，用于计算提升后的基础多面体的扩展集函数，
以及一个 solve_linear_program(c) 方法，
该方法使用贪心算法在g-polymatroid上求解线性规划问题。这与Mukhi第一篇论文中描述的优化方法一致。
form_box() 和 get_all_vertices() 可能是用于获取多胞体边界或顶点的方法。
'''
class Flexitroid(ABC):
    """Abstract base class for flexiblity of DERs and aggregations of DERS.

    This class defines the common interface that flexibile entities must implement
    for flexibility set representation and computation.
    """

    @abstractmethod
    def b(self, A: Set[int]) -> float:
        """Compute submodular function b for the g-polymatroid representation."""
        pass

    @abstractmethod
    def p(self, A: Set[int]) -> float:
        """Compute supermodular function p for the g-polymatroid representation."""
        pass

    @property
    @abstractmethod
    def T(self) -> int:
        """Get the time horizon."""
        pass

    def _b_star(self, A: Set) -> float:
        """Extended set function b* for the lifted base polyhedron.

        Args:
            A: A subset of the extended ground set T*.

        Returns:
            Value of b*(A) as defined in the paper.
        """
        if not isinstance(A, set):
            A = set(A)

        T_set = set(range(self.T))
        if self.T in A:  # t* is in A
            return -self.p(T_set - A)
        return self.b(A)

    def solve_linear_program(self, c: np.ndarray) -> np.ndarray:
        """Solve a linear program over the g-polymatroid using the greedy algorithm.

        Args:
            c: Cost vector of length T.

        Returns:
            Optimal solution vector.
        """
        # Extend cost vector with c*(t*) = 0
        c_star = np.append(c, 0)

        # Sort indices by non-decreasing cost
        pi = np.argsort(c_star)

        # Initialize solution vector
        v = np.zeros(self.T + 1)

        # Apply greedy algorithm
        S_k = set()
        b_star_prev = 0
        for k in pi:
            S_k.add(int(k))
            b_star = self._b_star(S_k)
            v[k] = b_star - b_star_prev
            b_star_prev = b_star

        # Project solution by removing t* component
        return v[:-1]
    
    def form_box(self):
        C = np.vstack([np.eye(self.T) + 1, -np.eye(self.T) - 1])
        box = np.array([self.solve_linear_program(c) for c in  C])
        return box


    def get_all_vertices(self):
        perms = []
        for t in range(self.T+1):
            perms.append(list(permutations(np.arange(self.T) + 1 - t))) 

        perms = np.array(perms).reshape(-1,self.T)
        V = np.array([self.solve_linear_program(c) for c in perms])
        return V