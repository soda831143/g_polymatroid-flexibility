"""
Joint Chance Constraint (JCC) 鲁棒边界计算模块
使用 Statistical Feasibility (SRO/Re-SRO) 方法

参考MATLAB实现:
- get_joint_uncertainty_set.m (SRO阶段)
- get_reconstruct_joint.m (Re-SRO阶段)
- RO_reserve_joint.m (鲁棒优化求解)
- Recon_RO_joint.m (重构鲁棒优化)
"""

import numpy as np
try:
    import cvxpy as cp
except ImportError:
    cp = None
from scipy.stats import binom
from typing import List, Dict, Tuple, Optional
import warnings

class JCCRobustCalculator:
    """
    为多个TCL计算满足联合机会约束的鲁棒边界
    
    实现思路参考MATLAB代码:
    1. SRO: 椭球不确定性集 + 初始鲁棒边界
    2. Re-SRO: 基于初始解重构多面体不确定性集
    """
    
    def __init__(self, tcl_fleet: List, uncertainty_data: Dict):
        """
        Args:
            tcl_fleet: TCL对象列表 (已构建物理约束 A_phys, b_phys_nom)
            uncertainty_data: {
                'D_shape': 形状集数据 (n1 × T) - 温度预测误差 (用于SRO)
                'D_calib': 校准集数据 (n2 × T) - 用于SRO大小校准
                'D_resro_calib': Re-SRO独立校准集 (n3 × T) - 可选,用于Re-SRO
                'epsilon': JCC违反概率上限 (对应MATLAB的rho)
                'delta': 统计置信度 (对应MATLAB的eta)
                'use_full_cov': 是否使用完整协方差矩阵 (默认True)
            }
        """
        self.tcl_fleet = tcl_fleet
        self.N = len(tcl_fleet)
        self.T = tcl_fleet[0].T if tcl_fleet else 0
        
        # 不确定性数据
        self.D_shape = uncertainty_data['D_shape']
        self.D_calib = uncertainty_data['D_calib']
        self.D_resro_calib = uncertainty_data.get('D_resro_calib', None)  # Re-SRO独立校准集
        self.epsilon = uncertainty_data.get('epsilon', 0.05)  # 对应MATLAB的rho
        self.delta = uncertainty_data.get('delta', 0.05)      # 对应MATLAB的eta
        self.use_full_cov = uncertainty_data.get('use_full_cov', True)
        
        self.n1 = len(self.D_shape)
        self.n2 = len(self.D_calib)
        self.n3 = len(self.D_resro_calib) if self.D_resro_calib is not None else 0
        
        # 存储中间结果
        self.b_robust_initial = None  # SRO阶段鲁棒边界
        self.b_robust_final = None    # Re-SRO阶段鲁棒边界
        self.U_initial = None         # 椭球不确定性集
        self.U_final = None           # 多面体不确定性集
        self.s_unified = None         # 统一校准参数
        
        print(f"\n=== 初始化JCC鲁棒计算器 ===")
        print(f"  TCL数量: {self.N}")
        print(f"  时间步数: {self.T}")
        print(f"  SRO形状集样本数: {self.n1}")
        print(f"  SRO校准集样本数: {self.n2}")
        if self.D_resro_calib is not None:
            print(f"  Re-SRO独立校准集样本数: {self.n3}")
        print(f"  JCC违反概率上限 ε: {self.epsilon}")
        print(f"  统计置信度 δ: {self.delta}")
    
    def compute_sro_bounds(self) -> Tuple[np.ndarray, Dict]:
        """
        SRO阶段: 计算椭球不确定性集和初始鲁棒边界
        
        参考: get_joint_uncertainty_set.m 中的形状学习和尺寸校准
        
        Returns:
            b_robust_initial: (N, constraint_num) 所有TCL的初始鲁棒RHS
            U_initial: 初始不确定性集参数 {mu, M, M_inv, s_star}
        """
        print("\n=== SRO阶段: 构建椭球不确定性集 ===")
        
        # 1. 形状估计 (MATLAB: mu_joint, M_joint)
        mu = np.mean(self.D_shape, axis=0)  # (T,)
        
        if self.use_full_cov:
            # 使用完整协方差矩阵
            cov = np.cov(self.D_shape.T)  # (T, T)
            # 添加正则化防止奇异
            cov = cov + 1e-8 * np.eye(self.T)
        else:
            # 仅使用对角协方差
            cov = np.diag(np.var(self.D_shape, axis=0))
        
        try:
            M_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            warnings.warn("协方差矩阵奇异,使用伪逆")
            M_inv = np.linalg.pinv(cov)
        
        # 2. 大小校准 (MATLAB: index = binoinv(eta, R, rho))
        k_star = self._calculate_k_star(self.n2, self.epsilon, self.delta)
        
        # 计算校准集样本的马氏距离
        mahalanobis_dists = []
        for xi in self.D_calib:
            diff = xi - mu
            dist_sq = diff @ M_inv @ diff
            mahalanobis_dists.append(dist_sq)
        
        # 排序并选择第k*个 (MATLAB: s_flow = sort(s_vals_flow); s_flow = s_flow(index))
        sorted_dists = np.sort(mahalanobis_dists)
        s_star = sorted_dists[k_star - 1] if k_star > 0 else 0
        
        self.U_initial = {
            'mu': mu,
            'M': cov,
            'M_inv': M_inv,
            's_star': s_star,
            'type': 'ellipsoid'
        }
        
        print(f"  均值 μ 范围: [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"  协方差矩阵条件数: {np.linalg.cond(cov):.2e}")
        print(f"  校准参数 k*: {k_star}")
        print(f"  椭球半径平方 s*: {s_star:.4f}")
        
        # 3. 计算每个TCL的鲁棒边界
        print("\n  计算各TCL的鲁棒边界...")
        b_robust_initial = []
        for i, tcl in enumerate(self.tcl_fleet):
            b_i_robust = self._compute_robust_rhs_ellipsoid(
                tcl.A_phys, tcl.b_phys_nom, self.U_initial, tcl
            )
            b_robust_initial.append(b_i_robust)
            
            # 显示收缩幅度
            shrinkage = np.mean(tcl.b_phys_nom - b_i_robust)
            print(f"    TCL {i}: 平均收缩 = {shrinkage:.4f}")
        
        self.b_robust_initial = np.array(b_robust_initial)
        return self.b_robust_initial, self.U_initial
    
    def compute_resro_bounds(self, u0_phys_individual: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Re-SRO阶段: 基于初始解u0重构不确定性集并计算最终鲁棒边界
        
        参考: get_reconstruct_joint.m 的统一校准方法
        
        Args:
            u0_phys_individual: 初始物理个体解 (N, T) 或 聚合解 (T,)
        
        Returns:
            b_robust_final: (N, constraint_num) 最终鲁棒RHS
            U_final: 最终不确定性集 (包含s_unified)
        """
        print("\n=== Re-SRO阶段: 重构多面体不确定性集 ===")
        
        # 处理输入维度
        if u0_phys_individual.ndim == 1:
            print("  警告: Re-SRO 接收到聚合解, 假设TCL同质并平均分配")
            u0_phys_individual = np.tile(u0_phys_individual / self.N, (self.N, 1))
        
        # 检查是否提供了独立的Re-SRO校准集
        if self.D_resro_calib is None:
            print("  警告: 未提供独立的Re-SRO校准集,使用SRO校准集 (不符合理论设计!)")
            resro_calib_data = self.D_calib
            n_resro = self.n2
        else:
            print(f"  使用独立的Re-SRO校准集: {self.n3} 样本")
            resro_calib_data = self.D_resro_calib
            n_resro = self.n3
        
        # 1. 计算初始解的裕度
        print("  步骤1: 计算初始解的鲁棒裕度...")
        margins = self._compute_initial_solution_margins(u0_phys_individual)
        
        # 2. 统一的归一化变换与尺寸校准 (使用Re-SRO独立校准集)
        print(f"  步骤2: 对所有约束进行统一归一化校准 (使用{n_resro}个Re-SRO样本)...")
        scores_unified = []
        
        for xi in resro_calib_data:  # ✓ 使用Re-SRO独立校准集
            score = self._compute_unified_violation_score(u0_phys_individual, xi, margins)
            scores_unified.append(score)
        
        # 3. 统计学校准 (基于Re-SRO样本数)
        k_star = self._calculate_k_star(n_resro, self.epsilon, self.delta)
        scores_unified_sorted = np.sort(scores_unified)
        s_unified = scores_unified_sorted[k_star - 1] if k_star > 0 else 0
        
        # 确保 s_unified 非负
        if s_unified < 0:
            s_unified = 0
            
        self.s_unified = s_unified
        self.U_final = {
            'u0_individual': u0_phys_individual,
            's_unified': s_unified,
            'margins': margins,
            'n_samples': n_resro,  # 记录使用的样本数
            'type': 'polyhedron_unified'
        }
        
        print(f"  统一校准参数 s_unified (s_r*): {s_unified:.4f}")
        print(f"  基于 {n_resro} 个独立Re-SRO样本校准")
        
        # 4. 计算最终鲁棒边界 (使用LP求解器)
        print("\n  步骤3: 计算最终鲁棒边界 (求解 LPs)...")
        b_robust_final = []
        for i, tcl in enumerate(self.tcl_fleet):
            b_i_robust = self._compute_robust_rhs_polyhedron(tcl, self.U_final, i)
            b_robust_final.append(b_i_robust)
            
            # 对比SRO和Re-SRO
            if self.b_robust_initial is not None:
                improvement = np.mean(
                    b_i_robust[:2*self.T] - self.b_robust_initial[i, :2*self.T]
                )
                print(f"    TCL {i}: Re-SRO 相比 SRO 功率裕度放松 = {improvement:.4f}")
        
        self.b_robust_final = np.array(b_robust_final)
        return self.b_robust_final, self.U_final
    
    def _calculate_k_star(self, n: int, epsilon: float, delta: float) -> int:
        """
        计算置信度参数 k* (统计可行性保证的核心)
        
        参考: Sample-Adaptive Robust (Eq. 20) 和 Statistically Feasible (Eq. 47)
        k* = min{r: Σ_{k=0}^{r-1} C(n,k)(1-ε)^k ε^(n-k) ≥ 1-δ}
        
        等价于: k* = binom.ppf(1-δ, n, 1-ε) + 1
        
        这确保了在升序排列的校准分数中,第k*个分数满足统计可行性条件
        """
        k_star_ppf = binom.ppf(1 - delta, n, 1 - epsilon)
        k_star = int(k_star_ppf) + 1
        
        # 边界处理
        if k_star > n:
            k_star = n
        if k_star <= 0:
            k_star = 1
            
        return k_star
    
    def _compute_uncertainty_coefficient(self, tcl) -> float:
        """
        计算不确定性系数 w = ∂b/∂ξ
        
        对于温度预测不确定性:
        P0(k) = (θ_a(k) - θ_r) / b
        θ_a(k) = θ̂_a(k) + ξ(k)
        
        因此: ∂P0/∂ξ = 1/b
        功率约束: u ≤ P_max - P0
        鲁棒边界收缩: w = -1/b (负号因为是上界收缩)
        """
        return -1.0 / tcl.b_coef
    
    def _compute_robust_rhs_ellipsoid(self, A: np.ndarray, b_nom: np.ndarray,
                                      U: Dict, tcl) -> np.ndarray:
        """
        在椭球不确定性集上计算鲁棒RHS
        
        参考MATLAB: get_joint_uncertainty_set.m 中的 alpha 计算
        
        对于约束 a_i^T u ≤ b_i(ξ) = b_nom[i] + w_i^T ξ
        鲁棒边界: b_robust[i] = b_nom[i] + w_i^T μ - √(s* · w_i^T M w_i)
        """
        mu = U['mu']
        M = U['M']
        s_star = U['s_star']
        T = self.T
        
        b_robust = b_nom.copy()
        
        # 不确定性系数
        w_coeff = self._compute_uncertainty_coefficient(tcl)
        
        # 对功率约束应用鲁棒化 (前2T行)
        # 功率上界约束: u(k) ≤ u_max(k) - w*ξ(k)
        # 功率下界约束: -u(k) ≤ -u_min(k) - w*ξ(k)
        for t in range(T):
            # 单个时刻的不确定性向量 (只有第t个分量非零)
            w_t = np.zeros(T)
            w_t[t] = w_coeff
            
            # 计算鲁棒裕度: √(s* · w^T M w)
            robust_margin = np.sqrt(s_star * (w_t @ M @ w_t))
            mean_shift = w_t @ mu
            
            # 上界约束 (第t行)
            b_robust[t] = b_nom[t] + mean_shift - robust_margin
            
            # 下界约束 (第T+t行)
            b_robust[T + t] = b_nom[T + t] + mean_shift - robust_margin
        
        # 状态约束 (2T:4T行) 不受温度不确定性直接影响,保持名义值
        # (这里假设状态约束仅依赖于物理参数,不依赖于温度预测)
        
        return b_robust
    
    def _compute_initial_solution_margins(self, u0_individual: np.ndarray) -> Dict:
        """
        计算初始解的裕度
        
        Args:
            u0_individual: (N, T) 个体物理功率调度解
        
        Returns:
            margins: 字典, 包含 (N, 4T) 的所有裕度
        """
        margins = {}
        all_margins_list = []
        eps_margin = 1e-6  # 防止除零
        
        for i, tcl in enumerate(self.tcl_fleet):
            # 从功率调度 u 计算状态轨迹 x
            u_i = u0_individual[i]  # (T,)
            x_i = np.zeros(self.T)
            x_prev = tcl.x0
            
            for t in range(self.T):
                x_i[t] = tcl.a * x_prev + tcl.delta * u_i[t]
                x_prev = x_i[t]
            
            # 构建完整变量向量 [u; x]
            full_vars = np.concatenate([u_i, x_i])  # (2T,)
            
            # 计算约束值 A*[u; x]
            constraint_vals = tcl.A_phys @ full_vars
            
            # 裕度 = b_nom - A*[u; x]
            tcl_margins = tcl.b_phys_nom - constraint_vals
            
            # 防止裕度为负或为零
            tcl_margins = np.maximum(tcl_margins, eps_margin)
            all_margins_list.append(tcl_margins)
        
        # (N, 4T)
        margins['all'] = np.array(all_margins_list)
        return margins
    
    def _compute_unified_violation_score(self, u0_individual: np.ndarray, xi: np.ndarray,
                                         margins: Dict) -> float:
        """
        计算 u0 在不确定性 xi 下的统一归一化违反分数 (g_r)
        
        g_r(ξ, u0) = max_{i,l} { Violation_il(ξ) / Margin_il }
        
        Violation_il(ξ) = (A_i u0_i)_l - b_i,l(ξ)
                        = (A_i u0_i)_l - (b_nom_il + w_il^T ξ)
                        = [ (A_i u0_i)_l - b_nom_il ] - w_il^T ξ
                        = -Margin_il - w_il^T ξ
        
        参考MATLAB: get_reconstruct_joint.m 中的 viol_up/viol_down 计算
        """
        max_normalized_violation = -np.inf
        all_margins = margins['all']  # (N, 4T)
        
        for i, tcl in enumerate(self.tcl_fleet):
            w_coeff = self._compute_uncertainty_coefficient(tcl)
            T = tcl.T
            
            # 遍历所有 4T 个约束
            for l in range(4 * T):
                margin_il = all_margins[i, l]
                
                # 计算 w_l^T * ξ
                if l < 2 * T:
                    # 功率约束
                    t = l % T
                    w_l_T_xi = w_coeff * xi[t]
                else:
                    # 状态约束 (假定不受温度不确定性直接影响)
                    w_l_T_xi = 0.0
                
                # Violation = -Margin - w^T*ξ
                violation_il = -margin_il - w_l_T_xi
                
                # 归一化违反分数
                score = violation_il / margin_il
                
                if score > max_normalized_violation:
                    max_normalized_violation = score
        
        # g_r 只关心正的违反
        return max(0, max_normalized_violation)
    
    def _compute_robust_rhs_polyhedron(self, tcl, U: Dict, tcl_idx: int) -> np.ndarray:
        """
        *** 核心修改：使用LP求解最终鲁棒边界 ***
        
        替换启发式方法,严格按照Re-SRO理论
        
        求解 4T 个 LP:
        h_i,l^robust = b_nom_il + max_{ξ ∈ U_final} { w_il^T * ξ }
        
        其中:
        U_final = { ξ | g_r(ξ, u0) ≤ s_r_star }
        g_r(ξ, u0) = max_{j,k} { [ -Margin_jk - w_jk^T * ξ ] / Margin_jk }
        
        参考MATLAB: Recon_RO_joint.m
        """
        T = tcl.T
        b_nom = tcl.b_phys_nom.copy()
        b_robust_final = b_nom.copy()
        
        s_r_star = U['s_unified']
        margins = U['margins']
        u0_individual = U['u0_individual']
        
        # 定义优化变量
        xi_var = cp.Variable(T)
        
        # === 构建 U_final 的约束 ===
        constraints_U_final = []
        
        for j, tcl_j in enumerate(self.tcl_fleet):
            w_coeff_j = self._compute_uncertainty_coefficient(tcl_j)
            
            for l_constr in range(4 * T):
                margin_jl = margins['all'][j, l_constr]
                
                # 计算 w_jl^T * ξ
                if l_constr < 2 * T:
                    t = l_constr % T
                    w_jl_T_xi = w_coeff_j * xi_var[t]
                else:
                    w_jl_T_xi = cp.Constant(0.0)  # 使用CVXPy常量而非标量
                
                # Violation = -Margin - w^T*ξ
                # 约束: Violation / Margin <= s_r_star
                # => (-Margin - w^T*ξ) / Margin <= s_r_star
                # => -Margin - w^T*ξ <= s_r_star * Margin
                # => -w^T*ξ <= s_r_star * Margin + Margin
                # => -w^T*ξ <= (1 + s_r_star) * Margin
                constraints_U_final.append(
                    -margin_jl - w_jl_T_xi <= s_r_star * margin_jl
                )
        
        # === 求解每个约束的LP ===
        w_coeff_i = self._compute_uncertainty_coefficient(tcl)
        
        for l in range(2 * T):  # 只优化功率约束
            t = l % T
            w_il_T_xi = w_coeff_i * xi_var[t]
            
            # 目标: max w_il^T * ξ
            objective = cp.Maximize(w_il_T_xi)
            
            prob = cp.Problem(objective, constraints_U_final)
            
            # 求解LP (使用Gurobi)
            try:
                prob.solve(solver=cp.GUROBI, verbose=False)
            except Exception as e:
                warnings.warn(f"LP求解失败 (TCL {tcl_idx}, Constr {l}): {e}")
                b_robust_final[l] = b_nom[l]  # 保守回退
                continue
            
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # 最终鲁棒边界 = b_nom + max(w^T * ξ)
                b_robust_final[l] = b_nom[l] + prob.value
            else:
                warnings.warn(f"LP求解器状态异常: {prob.status} (TCL {tcl_idx}, Constr {l})")
                b_robust_final[l] = b_nom[l]
        
        # 状态约束保持名义值 (不受温度不确定性直接影响)
        # b_robust_final[2*T:4*T] 已经是 b_nom[2*T:4*T]
        
        return b_robust_final


def ledoit_wolf_shrinkage(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf 协方差矩阵收缩估计
    
    参考: Ledoit & Wolf (2004) "A well-conditioned estimator for large-dimensional covariance matrices"
    
    Args:
        X: (n_samples, n_features) 数据矩阵
    
    Returns:
        shrunk_cov: 收缩后的协方差矩阵
        shrinkage: 收缩强度 λ ∈ [0, 1]
    """
    n_samples, n_features = X.shape
    
    # 样本协方差矩阵
    S = np.cov(X.T, bias=False)
    
    # 目标矩阵 F (对角阵,使用样本方差的平均值)
    mu = np.trace(S) / n_features
    F = mu * np.eye(n_features)
    
    # 计算最优收缩强度
    # (这里使用简化版本,完整版本需要更复杂的计算)
    delta = np.linalg.norm(S - F, 'fro') ** 2 / n_features
    
    # 估计 β
    X_centered = X - X.mean(axis=0)
    beta = 0
    for i in range(n_samples):
        dev = np.outer(X_centered[i], X_centered[i]) - S
        beta += np.linalg.norm(dev, 'fro') ** 2
    beta = beta / (n_samples ** 2)
    
    # 收缩强度
    shrinkage = min(beta / delta, 1.0) if delta > 0 else 0.0
    
    # 收缩估计
    shrunk_cov = (1 - shrinkage) * S + shrinkage * F
    
    return shrunk_cov, shrinkage
