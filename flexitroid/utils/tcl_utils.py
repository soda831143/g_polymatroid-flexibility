# flexitroid/utils/tcl_utils.py (清理后 - 仅保留必要功能)

import numpy as np
from scipy.stats import binom

def get_true_tcl_polytope_H_representation(
    T: int,
    a: float,
    x0: float,
    u_min: np.ndarray,
    u_max: np.ndarray,
    x_min_phys: float,
    x_max_phys: float,
    delta: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    构建TCL真实灵活性多面体的H表示 (A*u <= b)。
    
    物理约束:
    - 功率约束: u_min <= u(k) <= u_max
    - 状态约束: x_min_phys <= x(k) <= x_max_phys
    - 状态演化: x(k) = a*x(k-1) + δ*u(k)
    
    Args:
        T: 时间步数
        a: 热耗散系数
        x0: 初始状态
        u_min: 功率下限向量 (T,)
        u_max: 功率上限向量 (T,)
        x_min_phys: 状态下限
        x_max_phys: 状态上限
        delta: 状态演化系数
        
    Returns:
        (A, b): H表示矩阵和向量
    """
    # 1. 功率约束: u_min <= u <= u_max
    A_power_upper = np.eye(T)
    b_power_upper = u_max
    A_power_lower = -np.eye(T)
    b_power_lower = -u_min

    # 2. 状态约束: x_min_phys <= x(k) <= x_max_phys
    A_state_upper = np.zeros((T, T))
    A_state_lower = np.zeros((T, T))
    b_state_upper = np.zeros(T)
    b_state_lower = np.zeros(T)
    
    for k in range(1, T + 1):
        # x(k) = a^k * x0 + δ * sum_{s=1}^{k} a^(k-s) * u(s)
        coeffs = np.zeros(T)
        for s in range(k):
            coeffs[s] = delta * (a ** (k - 1 - s))
        
        # 上界: δ * sum a^(k-s) * u(s) <= x_max - a^k * x0
        A_state_upper[k-1, :] = coeffs
        b_state_upper[k-1] = x_max_phys - (a**k) * x0
        
        # 下界: -δ * sum a^(k-s) * u(s) <= -(x_min - a^k * x0)
        A_state_lower[k-1, :] = -coeffs
        b_state_lower[k-1] = -(x_min_phys - (a**k) * x0)
    
    # 3. 组合所有约束
    A = np.vstack([A_power_upper, A_power_lower, A_state_upper, A_state_lower])
    b = np.concatenate([b_power_upper, b_power_lower, b_state_upper, b_state_lower])

    return A, b


def build_uncertainty_set_ellipsoidal(
    shape_set: np.ndarray, 
    calibration_set: np.ndarray,
    epsilon: float, 
    delta: float,
    use_diag_cov: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    构建椭球不确定性集 (用于SRO方法)。
    
    基于论文: "Sample-Adaptive Robust Economic Dispatch..." (Lu et al., 2024)
    
    Args:
        shape_set: 形状学习数据集 D1 (n1_samples, T)
        calibration_set: 尺寸校准数据集 D2 (n2_samples, T)
        epsilon: 约束违反概率
        delta: 置信水平
        use_diag_cov: 是否只用对角协方差
        
    Returns:
        (mu, cov, s_star): 椭球中心、形状矩阵、尺寸参数
    """
    print("--- 构建椭球不确定性集 (SRO方法) ---")
    n1 = shape_set.shape[0]
    n2 = calibration_set.shape[0]
    print(f"形状学习集: {n1} 样本, 尺寸校准集: {n2} 样本")
    
    # 1. 形状学习 (从D1计算均值和协方差)
    mu = np.mean(shape_set, axis=0)
    cov = np.cov(shape_set, rowvar=False)
    
    if use_diag_cov:
        cov = np.diag(np.diag(cov))
    
    # 处理奇异协方差矩阵
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print("警告: 协方差矩阵奇异,使用伪逆")
        inv_cov = np.linalg.pinv(cov)
    
    # 2. 尺寸校准 (使用D2)
    diff = calibration_set - mu
    t_values = np.sum((diff @ inv_cov) * diff, axis=1)
    t_values_sorted = np.sort(t_values)
    
    # 3. 计算临界索引 k*
    k_star = int(np.floor(binom.ppf(q=1-delta, n=n2, p=1-epsilon))) if n2 > 0 else 0
    if np.isnan(k_star):
        k_star = n2 - 1
    k_star = max(0, min(k_star, len(t_values_sorted) - 1))
    s_star = t_values_sorted[k_star]
    
    print(f"置信度 1-δ = {1-delta:.2f}, 违反概率 ε = {epsilon:.2f}")
    print(f"临界索引 k* = {k_star}, 尺寸参数 s* = {s_star:.4f}")
    
    return mu, cov, s_star
