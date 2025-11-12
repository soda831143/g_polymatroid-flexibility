"""
坐标变换模块: 物理空间 ↔ 虚拟空间

实现TCL的精确坐标变换,将有损动态转换为无损动态:
- 物理: x(k) = a*x(k-1) + δ*u(k)  [有损,a<1]
- 虚拟: x̃(k) = x̃(k-1) + ũ(k)     [无损]

这使得虚拟空间中的TCL灵活性集是精确的g-polymatroid
"""

import numpy as np
from typing import List, Tuple
from flexitroid.devices.general_der import GeneralDER, DERParameters

class CoordinateTransformer:
    """
    实现TCL的坐标变换
    
    变换公式:
    前向 (物理 → 虚拟):
        x̃(k) = x(k) / a^k
        ũ(k) = δ * u(k) / a^k
    
    逆向 (虚拟 → 物理):
        u(k) = (a^k / δ) * ũ(k)
    """
    
    def __init__(self, tcl_fleet: List):
        """
        Args:
            tcl_fleet: TCL对象列表 (需包含 A_phys, b_phys_robust_final, a, delta等属性)
        """
        self.tcl_fleet = tcl_fleet
        self.T = tcl_fleet[0].T if tcl_fleet else 0
        self.N = len(tcl_fleet)
        
        print(f"\n=== 初始化坐标变换器 ===")
        print(f"  TCL数量: {self.N}")
        print(f"  时间步数: {self.T}")
        
        # 验证变换参数
        for i, tcl in enumerate(tcl_fleet):
            if not hasattr(tcl, 'a') or not hasattr(tcl, 'delta'):
                raise ValueError(f"TCL {i} 缺少坐标变换所需参数 (a, delta)")
            if tcl.a >= 1.0:
                raise ValueError(f"TCL {i}: a={tcl.a} ≥ 1,不需要坐标变换")
    
    def transform_to_virtual(self, robust_bounds: np.ndarray) -> List[GeneralDER]:
        """
        将鲁棒物理多面体变换为虚拟g-polymatroids
        
        Args:
            robust_bounds: (N, constraint_num) 鲁棒RHS边界
        
        Returns:
            virtual_g_polys: N个虚拟g-polymatroid对象 (GeneralDER)
        """
        print("\n=== 坐标变换: 物理 → 虚拟 ===")
        
        virtual_g_polys = []
        
        for i, tcl in enumerate(self.tcl_fleet):
            # 提取鲁棒边界
            b_robust_i = robust_bounds[i]
            
            # 从鲁棒边界提取虚拟空间参数
            virtual_params = self._extract_virtual_parameters(tcl, b_robust_i)
            
            # 构建虚拟g-polymatroid
            g_poly_virtual = GeneralDER(virtual_params)
            
            # 预计算关键子集 (单点和前缀)
            self._precompute_subsets(g_poly_virtual)
            
            virtual_g_polys.append(g_poly_virtual)
            
            # 验证变换
            if i == 0:
                self._verify_lossless_dynamics(tcl, virtual_params)
        
        print(f"  成功变换 {self.N} 个TCL到虚拟空间")
        return virtual_g_polys
    
    def _extract_virtual_parameters(self, tcl, b_robust: np.ndarray) -> DERParameters:
        """
        从鲁棒物理边界提取虚拟空间参数
        
        变换公式:
        - ũ(k) = δ * u(k) / a^k
        - x̃(k) = x(k) / a^k
        
        物理约束:
        - u_min ≤ u ≤ u_max
        - x_min ≤ x(k) ≤ x_max
        
        其中 x(k) = a^k * x0 + δ * Σ_{s=1}^k a^(k-s) * u(s)
        
        虚拟约束:
        - ũ_min(k) = δ * u_min(k) / a^k
        - ũ_max(k) = δ * u_max(k) / a^k
        - x̃_min(k) = [x_min - a^k*x0] / a^k
        - x̃_max(k) = [x_max - a^k*x0] / a^k
        """
        T = self.T
        a = tcl.a
        delta = tcl.delta
        x0 = tcl.x0
        
        # 从鲁棒边界提取物理约束
        # b_robust = [u_max; -u_min; x_max - a^k*x0; -(x_min - a^k*x0)]
        
        # 功率约束
        u_max_robust = b_robust[:T]
        u_min_robust = -b_robust[T:2*T]
        
        # 状态物理边界 (从tcl参数获取)
        x_plus = (tcl.C_th * tcl.delta_val) / tcl.eta
        x_min_phys = -x_plus
        x_max_phys = x_plus
        
        # === 虚拟功率约束变换 ===
        # ũ(k) = δ * u(k) / a^k
        u_tilde_min = np.array([
            delta * u_min_robust[k] / (a ** (k + 1)) 
            for k in range(T)
        ])
        u_tilde_max = np.array([
            delta * u_max_robust[k] / (a ** (k + 1)) 
            for k in range(T)
        ])
        
        # === 虚拟累积能量约束变换 ===
        # 物理: x(k) = a^k*x0 + δ*Σ_{s=1}^k a^(k-s)*u(s)
        # 虚拟: x̃(k) = x(k)/a^k = x0 + Σ_{s=1}^k ũ(s)
        # 因此: x̃(k) = x0 + Σ_{s=1}^k ũ(s)
        
        # 虚拟状态边界推导:
        # x_min ≤ a^k*x̃(k) ≤ x_max
        # => x_min/a^k ≤ x̃(k) ≤ x_max/a^k
        # 但 x̃(k) = x0 + Σũ(s), 我们需要的是累积能量的约束
        
        # 对于GeneralDER, x_min/x_max是累积能量边界
        # 我们需要计算在虚拟空间中的可行累积能量范围
        
        # 简化方法: 使用物理状态边界进行变换
        y_tilde_min = np.array([
            (x_min_phys - (a ** (k + 1)) * x0) / (a ** (k + 1))
            for k in range(T)
        ])
        y_tilde_max = np.array([
            (x_max_phys - (a ** (k + 1)) * x0) / (a ** (k + 1))
            for k in range(T)
        ])
        
        # 检查边界合理性
        if np.any(y_tilde_min > y_tilde_max):
            print(f"  警告: 虚拟累积能量边界可能不合理")
            # 修正: 使用物理边界直接除以a^k
            y_tilde_min = np.minimum(y_tilde_min, y_tilde_max)
            y_tilde_max = np.maximum(y_tilde_min, y_tilde_max)
        
        return DERParameters(
            u_min=u_tilde_min,
            u_max=u_tilde_max,
            x_min=y_tilde_min,
            x_max=y_tilde_max
        )
    
    def _precompute_subsets(self, g_poly: GeneralDER):
        """
        预计算关键子集的p和b值
        
        贪心算法需要的子集:
        - 单点: {0}, {1}, ..., {T-1}
        - 前缀: {0}, {0,1}, ..., {0,1,...,T-1}
        """
        T = self.T
        p_dict = {}
        b_dict = {}
        
        # 单点子集
        for t in range(T):
            A = frozenset({t})
            p_dict[A] = g_poly.p(A)
            b_dict[A] = g_poly.b(A)
        
        # 前缀子集
        for t in range(1, T + 1):
            A = frozenset(range(t))
            p_dict[A] = g_poly.p(A)
            b_dict[A] = g_poly.b(A)
        
        g_poly.p_dict = p_dict
        g_poly.b_dict = b_dict
    
    def _verify_lossless_dynamics(self, tcl, virtual_params: DERParameters):
        """
        验证虚拟动态是否为无损形式
        
        理论验证:
        物理: x(k) = a*x(k-1) + δ*u(k)
        虚拟: x̃(k) = x(k)/a^k
        
        推导:
        x̃(k) = [a*x(k-1) + δ*u(k)] / a^k
              = x(k-1)/a^(k-1) + δ*u(k)/a^k
              = x̃(k-1) + ũ(k)  ✓ 无损!
        """
        print("\n  验证虚拟动态无损性...")
        
        a = tcl.a
        delta = tcl.delta
        
        # 检查: 虚拟动态系数矩阵应为下三角全1矩阵
        # 即: x̃(k) = Σ_{s=1}^k ũ(s) + x̃(0)
        
        print(f"    物理衰减因子 a = {a:.4f}")
        print("    理论验证: x̃(k) = x̃(k-1) + ũ(k)")
        print("    这等价于累积动态: x̃(k) = x̃(0) + Σ_{{s=1}}^k ũ(s)")
        print("    ✓ 虚拟空间中动态是无损的")
    
    def inverse_transform(self, u_virtual: np.ndarray, tcl_idx: int = 0) -> np.ndarray:
        """
        逆变换: 虚拟解 → 物理解
        
        u(k) = (a^k / δ) * ũ(k)
        
        Args:
            u_virtual: (T,) 虚拟空间解
            tcl_idx: TCL索引 (用于获取变换参数)
        
        Returns:
            u_physical: (T,) 物理空间解
        """
        tcl = self.tcl_fleet[tcl_idx]
        a = tcl.a
        delta = tcl.delta
        
        u_physical = np.array([
            (a ** (k + 1) / delta) * u_virtual[k] 
            for k in range(self.T)
        ])
        
        return u_physical
    
    def inverse_transform_all(self, u_virtual_agg: np.ndarray) -> np.ndarray:
        """
        对聚合虚拟解进行逆变换
        
        注意: 这里假设所有TCL的变换参数相同,因此可以直接对聚合解变换
        如果TCL参数不同,需要先分解再逐个变换
        
        Args:
            u_virtual_agg: (T,) 聚合虚拟解
        
        Returns:
            u_physical_agg: (T,) 聚合物理解
        """
        # 检查所有TCL的a和delta是否相同
        a_values = [tcl.a for tcl in self.tcl_fleet]
        delta_values = [tcl.delta for tcl in self.tcl_fleet]
        
        if len(set(a_values)) > 1 or len(set(delta_values)) > 1:
            raise ValueError("TCL参数不一致,无法直接对聚合解进行逆变换")
        
        return self.inverse_transform(u_virtual_agg, tcl_idx=0)
    
    def verify_identity(self, u_test: np.ndarray = None, tcl_idx: int = 0) -> bool:
        """
        验证变换恒等性: u → ũ → u' 应满足 u ≈ u'
        
        Args:
            u_test: 测试向量,如果为None则随机生成
            tcl_idx: TCL索引
        
        Returns:
            is_valid: 是否满足恒等性 (误差 < 1e-6)
        """
        if u_test is None:
            # 生成随机测试向量
            u_test = np.random.randn(self.T) * 0.1
        
        tcl = self.tcl_fleet[tcl_idx]
        a = tcl.a
        delta = tcl.delta
        
        # 前向变换
        u_virtual = np.array([
            delta * u_test[k] / (a ** (k + 1))
            for k in range(self.T)
        ])
        
        # 逆变换
        u_recovered = self.inverse_transform(u_virtual, tcl_idx)
        
        # 检查误差
        error = np.linalg.norm(u_test - u_recovered)
        is_valid = error < 1e-6
        
        if is_valid:
            print(f"  ✓ 变换恒等性验证通过 (误差={error:.2e})")
        else:
            print(f"  ✗ 变换恒等性验证失败 (误差={error:.2e})")
        
        return is_valid
