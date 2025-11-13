"""Pure Python implementation of p_fast (fallback when Cython extension is not available)."""
import numpy as np

def p_fast(A, T, active, u_min, u_max, x_min, x_max):
    """Fast implementation of p function (Python version)."""
    A_c = active - A
    p = np.sum(u_min[list(A)])
    b_c = np.sum(u_max[list(A_c)])
    t_set = set()
    
    for t in range(T):
        t_set.add(t)
        A_c_minus_t = A_c - t_set
        A_minus_t = A - t_set
        
        # Update p
        p = max(p, x_min[t] - b_c + 
                np.sum(u_min[list(A_minus_t)]) +
                np.sum(u_max[list(A_c_minus_t)]))
        
        # Update b_c
        b_c = min(b_c, x_max[t] - p +
                  np.sum(u_min[list(A_minus_t)]) +
                  np.sum(u_max[list(A_c_minus_t)]))
    
    return p

