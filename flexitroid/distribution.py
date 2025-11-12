# In flexitroid/distribution.py
import random
from typing import List, Tuple

class EmpiricalDistribution:
    def __init__(self, samples: List[Tuple]):
        # samples: List of xi_tuples, e.g., [(e_l1,e_u1,t_a1,t_d1,m1), ...]
        if not samples:
            raise ValueError("Samples cannot be empty for empirical distribution.")
        self.samples = samples
        self.num_samples = len(samples)

    def get_support_points(self) -> List[Tuple]:
        # These are the \Xi_P = {xi_1, ..., xi_M} from the paper (Section 5)
        return self.samples

    def get_probabilities(self) -> List[float]:
        # For empirical distribution, each sample has prob 1/M
        return [1.0 / self.num_samples] * self.num_samples

    def sample_n(self, n_evs: int) -> List[Tuple]:
        """Draws N i.i.d. samples from this empirical distribution."""
        return random.choices(self.samples, k=n_evs)