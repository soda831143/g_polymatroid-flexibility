from .general_der import GeneralDER, DERParameters
from . import parameter_sampling as sample
import numpy as np
from typing import Set
'''
定义了光伏系统 PV 类，它继承自 GeneralDER。
PV 系统的特点是没有能量存储，因此其能量状态上下限被设置为无穷大。
b(A) 和 p(A) 函数被简化为仅基于功率上下限的求和，
这对应Mukhi第一篇论文中针对无能量约束的PV的简化情况（Section III-D1）。
'''

class PV(GeneralDER):
    """Photovoltaic system flexibility set representation.

    This class implements the flexibility set for a PV system with
    curtailment capabilities but no energy storage.
    """

    def __init__(self, u_min: np.ndarray, u_max: np.ndarray):
        """Initialize PV flexibility set with power bounds only.

        Args:
            u_min: Lower bound on power consumption for each timestep.
            u_max: Upper bound on power consumption for each timestep.
        """
        T = len(u_min)
        assert len(u_max) == T, "Power bounds must have same length"
        assert np.all(u_min <= u_max), "Invalid power bounds"

        # Create DER parameters with infinite energy bounds
        params = DERParameters(
            u_min=u_min,
            u_max=u_max,
            x_min=np.full(T, -np.inf),
            x_max=np.full(T, np.inf),
        )
        super().__init__(params)

    def b(self, A: Set[int]) -> float:
        """Compute submodular function b for the g-polymatroid representation.

        Args:
            A: Subset of the ground set T.

        Returns:
            Value of b(A) as defined in Section II-D of the paper
        """
        return np.sum(self.params.u_max[list(A)])

    def p(self, A: Set[int]) -> float:
        """Compute supermodular function p for the g-polymatroid representation.

        Args:
            A: Subset of the ground set T.

        Returns:
            Value of p(A) as defined in Section II-D of the paper
        """
        return np.sum(self.params.u_min[list(A)])

    @classmethod
    def example(cls, T: int = 24) -> "PV":
        """Create an example PV system with realistic power bounds.

        Creates a PV system with a sinusoidal generation profile peaking at midday.
        Negative power indicates generation (export).

        Args:
            T: Number of timesteps (default 24 for hourly resolution)

        Returns:
            PV instance with example parameters
        """
        u_min, u_max = sample.generation(T)
        return cls(u_min=u_min, u_max=u_max)
