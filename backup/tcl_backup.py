# flexitroid/devices/tcl.py (修改后)

import numpy as np
from flexitroid.flexitroid import Flexitroid
from flexitroid.devices.general_der import GeneralDER, DERParameters
# 【修改点1】: 导入新旧两个函数
from flexitroid.utils.tcl_utils import solve_deterministic_maximal_inner_approximation, solve_provably_inner_approximation
from itertools import combinations

class TCL(Flexitroid):
    """
    表示单个TCL的类。
    只递推和存储所有前缀子集（累计能量约束）的g-polymatroid边界，
    以便高效聚合和贪心算法调用。
    """
    def __init__(self, tcl_params: dict, build_g_poly: bool = True, theta_a_forecast: np.ndarray = None, use_provable_inner: bool = True):
        """
        初始化TCL设备。
        
        Args:
            tcl_params: TCL参数字典
            build_g_poly: 是否构建g-polymatroid
            theta_a_forecast: 预测温度序列
            use_provable_inner: 是否使用可证明的内近似方法（默认True）
        """
        # 存储参数以备后用
        self.tcl_params = tcl_params
        self.a = tcl_params['a']
        self.delta = tcl_params['delta']
        self.C_th = tcl_params['C_th']
        self.eta = tcl_params['eta']
        self.b_coef = tcl_params['b']
        self.P_m = tcl_params['P_m']
        self.theta_r = tcl_params['theta_r']
        self.x0 = tcl_params.get('x0', 0.0)
        self.theta_a_forecast = theta_a_forecast  # 只在需要时传递
        self.delta_val = tcl_params.get('delta_val', 2.0)
        self._internal_g_poly = None
        if build_g_poly:
            if theta_a_forecast is None:
                raise ValueError("build_g_poly=True 时必须提供 theta_a_forecast")
            
            if use_provable_inner:
                # 【修改点2】: 调用新的、理论保证的内近似函数
                print(f"INFO: 为TCL设备进行可证明的内近似...")
                y_lower, y_upper = solve_provably_inner_approximation(
                    tcl_params={**self.tcl_params, 'theta_a_forecast': theta_a_forecast}
                )
            else:
                # 使用原有的方法（为了向后兼容或特殊需求）
                print(f"INFO: 为TCL设备进行传统内近似...")
                y_lower, y_upper = solve_deterministic_maximal_inner_approximation(
                    tcl_params={**self.tcl_params, 'theta_a_forecast': theta_a_forecast}
                )

            if y_lower is None or np.any(np.isinf(y_lower)) or y_upper is None or np.any(np.isinf(y_upper)):
                raise ValueError("内近似边界计算失败。")

            # 自动修正：如果 y_lower > y_upper，调整为相等
            # if np.any(y_lower > y_upper):
            #     print("警告: 检测到 y_lower > y_upper，自动修正为相等！")
            #     print("y_lower:", y_lower)
            #     print("y_upper:", y_upper)
            #     y_lower = np.minimum(y_lower, y_upper)
            #     y_upper = np.maximum(y_lower, y_upper)
            # 使用计算出的(y, ȳ)边界来构建内部的g-polymatroid表示
            # 这是通过一个通用的DER模型来实现的，该模型由p和b函数定义
            # 使用统一的基线功率公式：P0 = (θa_forecast - θr) / b
            # 当 θa > θr 时，P0 > 0 (需要制冷)
            # 当 θa < θr 时，P0 < 0 (需要制热)
            # P0_forecast = (theta_a_forecast - self.theta_r) / self.b_coef
            P0_unconstrained = (theta_a_forecast - self.theta_r) / self.b_coef
            P0_forecast = np.maximum(0, P0_unconstrained)
            u_min_det = -P0_forecast
            u_max_det = self.P_m - P0_forecast
            params_g_poly = DERParameters(
                u_min=u_min_det,
                u_max=u_max_det,
                x_min=y_lower,
                x_max=y_upper
            )
            self._internal_g_poly = GeneralDER(params_g_poly)
            # 只递推所有前缀子集的p_dict和b_dict
            T = tcl_params['T']
            p_dict = {}
            b_dict = {}
            # 1. 递推所有前缀子集 {0}, {0,1}, ..., {0,1,...,T-1}
            for t in range(1, T+1):
                A = frozenset(range(t))
                b_dict[A] = self.b(A)
                p_dict[A] = self.p(A)
            # 2. 补充所有单时刻子集 {0}, {1}, ..., {T-1}（贪心算法需要）
            for t in range(T):
                A = frozenset({t})
                b_dict[A] = self.b(A)
                p_dict[A] = self.p(A)
            self._internal_g_poly.b_dict = b_dict
            self._internal_g_poly.p_dict = p_dict
    def b(self, A: frozenset) -> float:
        """返回定义该TCL确定性g-polymatroid的子模上界函数 b(A)"""
        if self._internal_g_poly is None:
            raise RuntimeError("g-polymatroid未初始化，无法调用b(A)。")
        return self._internal_g_poly.b(A)
    def p(self, A: frozenset) -> float:
        """返回定义该TCL确定性g-polymatroid的超模下界函数 p(A)"""
        if self._internal_g_poly is None:
            raise RuntimeError("g-polymatroid未初始化，无法调用p(A)。")
        return self._internal_g_poly.p(A)
    @property
    def T(self) -> int:
        """返回时间跨度 T"""
        return self.tcl_params['T']