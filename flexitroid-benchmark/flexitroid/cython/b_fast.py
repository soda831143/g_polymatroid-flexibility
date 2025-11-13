"""Pure Python implementation of b_fast (fallback when Cython extension is not available)."""
import numpy as np

def b_fast(A, T, active, u_min, u_max, x_min, x_max):
    """Fast implementation of b function (Python version)."""
    A_c = active - A
    b = np.sum(u_max[list(A)])
    p_c = np.sum(u_min[list(A_c)])
    t_set = set()
    
    for t in range(T):
        t_set.add(t)
        A_c_minus_t = A_c - t_set
        A_minus_t = A - t_set
        
        # Update b
        b = min(b, x_max[t] - p_c +
                np.sum(u_min[list(A_c_minus_t)]) +
                np.sum(u_max[list(A_minus_t)]))
        
        # Update p_c
        p_c = max(p_c, x_min[t] - b +
                  np.sum(u_max[list(A_minus_t)]) +
                  np.sum(u_min[list(A_c_minus_t)]))
    
    return b

