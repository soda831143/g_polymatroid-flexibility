"""TCL device wrapper for flexitroid.

Minimal, well-formed implementation restoring importability. This file may be
extended later with more features but currently focuses on fixing syntax and
providing a safe internal g-polymatroid construction used by comparison code.
"""

from typing import Optional, Dict, Any

import numpy as np

from flexitroid.flexitroid import Flexitroid
from flexitroid.devices.general_der import GeneralDER, DERParameters
try:
    from flexitroid.utils.tcl_utils import (
        solve_deterministic_maximal_inner_approximation,
        solve_provably_inner_approximation,
    )
except Exception:
    # tcl_utils may not expose these helpers in all branches; fall back to None
    solve_deterministic_maximal_inner_approximation = None
    solve_provably_inner_approximation = None


class TCL(Flexitroid):
    """A single TCL device wrapper.

    Modes:
    - physical simulation (build_g_poly=False): store physical params only.
    - deterministic g-polymatroid (build_g_poly=True): build a simple
      deterministic internal GeneralDER representation for comparison algorithms.
    """

    def __init__(
        self,
        tcl_params: Dict[str, Any],
        build_g_poly: bool = True,
        theta_a_forecast: Optional[np.ndarray] = None,
        use_provable_inner: bool = True,
    ) -> None:
        self.tcl_params = dict(tcl_params)
        self.a = self.tcl_params.get("a")
        self.delta = self.tcl_params.get("delta")
        self.C_th = self.tcl_params.get("C_th")
        self.eta = self.tcl_params.get("eta")
        self.b_coef = self.tcl_params.get("b")
        self.P_m = self.tcl_params.get("P_m")
        self.theta_r = self.tcl_params.get("theta_r")
        self.x0 = self.tcl_params.get("x0", 0.0)
        self.delta_val = self.tcl_params.get("delta_val", 2.0)

        self.theta_a_forecast = theta_a_forecast
        self._internal_g_poly: Optional[GeneralDER] = None
        
        # 物理约束矩阵 (用于JCC鲁棒边界计算)
        self.A_phys = None
        self.b_phys_nom = None

        if build_g_poly:
            if theta_a_forecast is None:
                raise ValueError("build_g_poly=True 时必须提供 theta_a_forecast")

            T = int(self.tcl_params["T"])
            P0_unconstrained = (np.asarray(theta_a_forecast) - self.theta_r) / self.b_coef
            P0_forecast = np.maximum(0.0, P0_unconstrained)

            u_min_det = -P0_forecast
            u_max_det = self.P_m - P0_forecast

            # 计算g-polymatroid的累积能量边界
            # 注意: 这里的边界是累积和 sum_{s=0}^{t} u(s) 的范围,不是物理状态x(t)
            # g-polymatroid的p()和b()函数会自动考虑物理状态约束的影响
            y_lower_cumsum = np.cumsum(u_min_det)[:T]
            y_upper_cumsum = np.cumsum(u_max_det)[:T]

            # 使用累积和边界构建g-polymatroid
            params_g_poly = DERParameters(u_min=u_min_det, u_max=u_max_det, x_min=y_lower_cumsum, x_max=y_upper_cumsum)
            self._internal_g_poly = GeneralDER(params_g_poly)
            
            # 构建物理约束矩阵 (H-representation: A*[u; x] <= b)
            # 约束形式:
            # 1. u_min <= u <= u_max  (功率约束,每个时刻)
            # 2. x_min_phys <= x <= x_max_phys  (物理状态约束,每个时刻)
            # 其中 x(k) = a*x(k-1) + delta*u(k)
            
            # 计算物理状态边界(用于JCC鲁棒优化)
            x_plus = (self.C_th * self.delta_val) / self.eta if self.eta > 0 else 100.0
            x_min_phys = -x_plus
            x_max_phys = x_plus
            
            # 注意: 温度预测误差仅影响 u 的边界,不影响 x 的边界
            # P0(θ_a + ξ) = (θ_a + ξ - θ_r)/b
            # u_min = -P0 → 受ξ影响: ∂u_min/∂ξ = -1/b
            # u_max = P_m - P0 → 受ξ影响: ∂u_max/∂ξ = -1/b
            
            # 构建 A 矩阵和名义 b 向量
            num_constraints = 2 * T + 2 * T  # u上下界(2T) + x上下界(2T)
            num_vars = 2 * T  # [u_0,...,u_{T-1}, x_0,...,x_{T-1}]
            
            A = np.zeros((num_constraints, num_vars))
            b_nom = np.zeros(num_constraints)
            
            row_idx = 0
            
            # u的上界约束: u_t <= u_max_t → u_t <= P_m - P0_t
            for t in range(T):
                A[row_idx, t] = 1.0  # u_t系数
                b_nom[row_idx] = u_max_det[t]  # 名义RHS = P_m - P0_forecast
                row_idx += 1
            
            # u的下界约束: u_t >= u_min_t → -u_t <= -u_min_t
            for t in range(T):
                A[row_idx, t] = -1.0
                b_nom[row_idx] = -u_min_det[t]  # RHS = P0_forecast
                row_idx += 1
            
            # x的上界约束: x_t <= x_max_phys (使用物理边界)
            for t in range(T):
                A[row_idx, T + t] = 1.0
                b_nom[row_idx] = x_max_phys  # 物理上界(常数)
                row_idx += 1
            
            # x的下界约束: x_t >= x_min_phys → -x_t <= -x_min_phys (使用物理边界)
            for t in range(T):
                A[row_idx, T + t] = -1.0
                b_nom[row_idx] = -x_min_phys  # 物理下界(常数)
                row_idx += 1
            
            self.A_phys = A
            self.b_phys_nom = b_nom

            # 预计算常用子集的 b 与 p（前缀集合与单时刻集合），以便上层算法快速访问
            b_dict: Dict[frozenset, float] = {}
            p_dict: Dict[frozenset, float] = {}

            # 前缀集合 {0}, {0,1}, ..., {0..t-1}
            for t in range(1, T + 1):
                A = frozenset(range(t))
                try:
                    b_dict[A] = self._internal_g_poly.b(A)
                    p_dict[A] = self._internal_g_poly.p(A)
                except Exception:
                    # 若内部计算失败，保守赋值为 0
                    b_dict[A] = 0.0
                    p_dict[A] = 0.0

            # 单时刻集合 {0}, {1}, ..., {T-1}
            for t in range(T):
                A = frozenset({t})
                try:
                    b_dict[A] = self._internal_g_poly.b(A)
                    p_dict[A] = self._internal_g_poly.p(A)
                except Exception:
                    b_dict[A] = 0.0
                    p_dict[A] = 0.0

            # 将字典注入内部对象，以便 GeneralDER 的调用可以使用缓存（若它支持）
            try:
                self._internal_g_poly.b_dict = b_dict
                self._internal_g_poly.p_dict = p_dict
            except Exception:
                # 如果 GeneralDER 没有这些属性，忽略
                pass

    def b(self, A: frozenset) -> float:
        """返回子模上界函数 b(A)。代理到内部 GeneralDER。"""
        if self._internal_g_poly is None:
            raise RuntimeError("g-polymatroid未初始化, 无法调用 b(A)。请设置 build_g_poly=True")
        return self._internal_g_poly.b(A)

    def p(self, A: frozenset) -> float:
        """返回超模下界函数 p(A)。代理到内部 GeneralDER。"""
        if self._internal_g_poly is None:
            raise RuntimeError("g-polymatroid未初始化, 无法调用 p(A)。请设置 build_g_poly=True")
        return self._internal_g_poly.p(A)

    @property
    def T(self) -> int:
        """返回时间跨度 T。"""
        return int(self.tcl_params["T"])
    
    def get_u_only_polytope(self):
        """构建仅包含u变量的polytope投影,用于传统算法。
        
        利用动力学方程 x(k) = a·x(k-1) + delta·u(k) 和 x(0)=x0,
        将 x(k) = sum_{j=0}^{k-1} a^{k-1-j}·delta·u(j) + a^k·x0
        代入 x 的约束,得到关于 u 的约束。
        
        【关键修正】: 使用物理状态边界,而不是g-polymatroid的操作边界。
        物理边界基于TCL的热容量和温度范围,比操作边界更严格,
        能够防止优化器选择u=-P0的平凡解。
        
        Returns:
            tuple: (A_u, b_u) 其中 A_u @ u <= b_u
        """
        if self.A_phys is None or self.b_phys_nom is None:
            raise RuntimeError("物理约束矩阵未构建")
        
        T = self.T
        a = self.tcl_params['a']
        delta = self.tcl_params['delta']
        x0 = self.tcl_params['x0']
        
        # 使用物理状态边界 (与Exact算法一致)
        x_plus = (self.tcl_params['C_th'] * self.tcl_params['delta_val']) / self.tcl_params['eta'] if self.tcl_params['eta'] > 0 else 100.0
        x_min_phys = -x_plus
        x_max_phys = x_plus
        
        # 【关键修正】: 使用物理边界代替g-polymatroid的y_lower/y_upper
        # 物理边界对所有时刻是常数,更严格
        y_lower_phys = np.full(T, x_min_phys)
        y_upper_phys = np.full(T, x_max_phys)
        
        # A_phys 结构: [u约束(2T行); x约束(2T行)], 变量顺序 [u_0...u_{T-1}, x_0...x_{T-1}]
        # 我们需要提取:
        # 1. 直接的 u 约束 (前2T行)
        # 2. x 约束转换为 u 约束 (后2T行,但使用物理边界)
        
        # 直接u约束: A_u @ u <= b_u (前2T行,仅涉及前T列)
        A_u_direct = self.A_phys[:2*T, :T]  # shape (2T, T)
        b_u_direct = self.b_phys_nom[:2*T]
        
        # x约束转换: 使用物理边界而不是名义边界
        # x(k) = a·x(k-1) + delta·u(k), x(0) = x0
        # 递推展开: x(k) = a^k·x0 + sum_{j=0}^{k} a^{k-j}·delta·u(j)
        # 构建转换矩阵 M: x = M @ u + x0_offset
        M = np.zeros((T, T))  # x = M @ u + offset
        x0_offset = np.zeros(T)
        
        for k in range(T):
            x0_offset[k] = (a ** k) * x0
            for j in range(k + 1):  # 包括当前时刻 j=k
                M[k, j] = (a ** (k - j)) * delta
        
        # 构建物理状态约束: x_min_phys <= x(k) <= x_max_phys
        # 转换为: x_min_phys <= M @ u + offset <= x_max_phys
        # 即: M @ u <= x_max_phys - offset  (上界)
        #    -M @ u <= -x_min_phys + offset (下界)
        
        A_x_upper = M  # shape (T, T)
        b_x_upper = y_upper_phys - x0_offset
        
        A_x_lower = -M  # shape (T, T)
        b_x_lower = -y_lower_phys + x0_offset
        
        # 合并所有u约束
        A_u_combined = np.vstack([A_u_direct, A_x_upper, A_x_lower])  # shape (4T, T)
        b_u_combined = np.concatenate([b_u_direct, b_x_upper, b_x_lower])
        
        return A_u_combined, b_u_combined
