# flexitroid/aggregations/robust_aggregator.py (新增文件)

import numpy as np
import math
from .aggregator import Aggregator
from scipy.linalg import sqrtm

class RobustAggregator(Aggregator):
    """
    鲁棒聚合器，实现了“先（确定性）聚合，再（在聚合层面）鲁棒化”的逻辑。
    """
    def __init__(self, devices: list, uncertainty_params: tuple):
        """
        初始化鲁棒聚合器。

        Args:
            devices (list): TCL设备对象的列表。
            uncertainty_params (tuple): 包含(mu, cov, s_star)的元组，描述共同的不确定性来源。
        """
        print("\n--- 初始化鲁棒聚合器 (RobustAggregator) ---")
        
        # --- 步骤2: 确定性聚合 ---
        # 调用父类的构造函数，它会自动完成对所有设备确定性(p,b)函数的求和
        super().__init__(devices)
        print("确定性聚合完成。")
        
        self.mu, self.cov, self.s_star = uncertainty_params
        
        # 预计算用于鲁棒边界的协方差矩阵平方根
        try:
            self.cov_sqrt = sqrtm(self.cov).real
        except Exception:
            self.cov_sqrt = np.linalg.pinv(sqrtm(self.cov.T @ self.cov)).T @ self.cov.T

        # --- 步骤3: 聚合层面鲁棒化 ---
        self.num_devices = len(devices)
        if self.num_devices > 0:
            # 假设所有设备的物理参数相同，用于计算总的不确定性影响系数
            tcl_params = devices[0].tcl_params
            a = tcl_params['a']; b = tcl_params['b']; C_th = tcl_params['C_th']
            eta = tcl_params['eta']; delta = tcl_params['delta']
            # 根据物理模型 w(k) = -δ_ω * ω(k), 定义单个设备的扰动系数
            self.w_coeff_single = -(1 - a) * delta * (C_th / (eta * b))
        else:
            self.w_coeff_single = 0
            
        print("鲁棒聚合器已准备就绪。")

    def _get_robust_terms(self, A: frozenset) -> tuple[float, float]:
        """
        为给定的时间集合A，计算聚合后的均值影响和鲁棒裕度。
        """
        if self.num_devices == 0:
            return 0.0, 0.0
        
        # 聚合扰动系数: w_agg(t) = N * w_single(t)
        w_coeff_agg = self.num_devices * self.w_coeff_single
        
        # 构造系数向量 c_A，用于计算 Σ_{t in A} w_agg(t)
        c_A = np.zeros(self.T)
        for t in A:
            c_A[t] = w_coeff_agg
            
        # 计算均值影响和鲁棒裕度
        mean_impact = c_A @ self.mu
        robust_margin = math.sqrt(self.s_star) * np.linalg.norm(c_A @ self.cov_sqrt, 2)
        
        return mean_impact, robust_margin

    def b(self, A: frozenset) -> float:
        """
        返回鲁棒化的上界子模函数 b_rob(A)。
        b_rob(A) = b_det(A) - max(Σ w_agg(t))
        """
        b_deterministic = super().b(A)
        mean_impact, robust_margin = self._get_robust_terms(A)
        
        # 上界收缩，减去扰动的最大值 (mean + margin)
        return b_deterministic - (mean_impact + robust_margin)

    def p(self, A: frozenset) -> float:
        """
        返回鲁棒化的下界超模函数 p_rob(A)。
        p_rob(A) = p_det(A) - min(Σ w_agg(t))
        """
        p_deterministic = super().p(A)
        mean_impact, robust_margin = self._get_robust_terms(A)

        # 下界收缩（即增大），减去扰动的最小值 (mean - margin)
        return p_deterministic - (mean_impact - robust_margin)