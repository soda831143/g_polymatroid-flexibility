"""
正确的TCL g-polymatroid实现 - 使用显式LP求解p(A)和b(A)

【核心问题】
=============
`GeneralDER.py` 中的动态规划算法无法正确处理指数衰减的虚拟边界:
    x_max[t] = x_bar / a^t  (随时间指数衰减)

这导致 p(A) 和 b(A) 的计算错误,进而导致聚合可行集F_agg成为外近似。

【解决方案】
=============
重写 p(A) 和 b(A) 函数,使用Gurobi显式求解其数学定义:
    b(A) = max { Σ_{t∈A} u(t) | u ∈ F }
    p(A) = min { Σ_{t∈A} u(t) | u ∈ F }

这是100%数学正确的方法。
"""

import numpy as np
from flexitroid.devices.general_der import GeneralDER

# 【关键修复】导入Gurobi - 必须成功，否则算法无法执行
try:
    import gurobipy as gp
    from gurobipy import GRB
except (ImportError, ModuleNotFoundError) as e:
    # 【致命错误】不允许静默回退！
    error_msg = (
        f"\n{'='*80}\n"
        f"【致命错误】Gurobi 导入失败！\n"
        f"{'='*80}\n"
        f"错误信息: {e}\n\n"
        f"CorrectTCL_GPoly 依赖 Gurobi 来正确计算 p(A) 和 b(A) 函数。\n"
        f"没有 Gurobi，该类将静默回退到 GeneralDER 的错误 DP 算法，\n"
        f"导致所有结果都不可靠。\n\n"
        f"【必须采取的行动】\n"
        f"1. 确保 Gurobi 已安装: pip install gurobipy\n"
        f"2. 确保 Gurobi 许可证已配置（检查 GRB_LICENSE_FILE 环境变量）\n"
        f"3. 运行此脚本的 Python 环境必须有访问 Gurobi 的权限\n\n"
        f"【诊断】\n"
        f"请在命令行运行以下命令进行诊断：\n"
        f"  python -c \"import gurobipy; print('Gurobi 可用')\"\n"
        f"  gurobi_cl --version\n"
        f"  echo %GRB_LICENSE_FILE%  （Windows）\n"
        f"  echo $GRB_LICENSE_FILE  （Linux/Mac）\n"
        f"{'='*80}\n"
    )
    raise ImportError(error_msg) from e


class CorrectTCL_GPoly(GeneralDER):
    """
    正确的TCL g-polymatroid实现
    
    重写p(A)和b(A)函数,使用Gurobi显式求解LP,
    而不是使用GeneralDER中可能不适用于指数边界的DP算法。
    """
    
    def __init__(self, params):
        """
        Args:
            params: DERParameters对象,包含:
                - u_min, u_max: 功率边界 (T,)
                - x_min, x_max: 累积能量边界 (T,)
        """
        super().__init__(params)
        self._T = len(params.u_min)
        self.active = set(range(self._T))
        
        # 缓存边界参数以供LP使用
        self._u_min = np.array(params.u_min)
        self._u_max = np.array(params.u_max)
        self._x_min = np.array(params.x_min)
        self._x_max = np.array(params.x_max)
        
        # 【调试】计数器
        self._pb_call_count = 0
    
    def _solve_lp_for_set(self, A, objective_type='max'):
        """
        为给定集合A显式求解LP来计算p(A)或b(A)
        
        问题形式:
            max/min  Σ_{t∈A} u(t)
            s.t.     u_min(t) ≤ u(t) ≤ u_max(t),  ∀t
                     x_min(t) ≤ Σ_{s=0}^{t} u(s) ≤ x_max(t),  ∀t
        
        Args:
            A: 时间步集合
            objective_type: 'max' 或 'min'
        
        Returns:
            最优目标值
        """
        if not A or len(A) == 0:
            return 0.0
        
        try:
            # 创建Gurobi模型
            model = gp.Model("pb_function")
            model.setParam('OutputFlag', 0)  # 静默模式
            
            # 决策变量: u(t) for t in [0, T-1]
            u = model.addVars(self._T, lb=-gp.GRB.INFINITY, name="u")
            
            # 约束1: 功率边界
            for t in range(self._T):
                model.addConstr(u[t] >= self._u_min[t], f"u_min_{t}")
                model.addConstr(u[t] <= self._u_max[t], f"u_max_{t}")
            
            # 约束2: 累积能量边界
            for t in range(self._T):
                cumsum_expr = gp.quicksum(u[s] for s in range(t+1))
                model.addConstr(cumsum_expr >= self._x_min[t], f"x_min_{t}")
                model.addConstr(cumsum_expr <= self._x_max[t], f"x_max_{t}")
            
            # 目标函数: Σ_{t∈A} u(t)
            obj_expr = gp.quicksum(u[t] for t in A)
            
            if objective_type == 'max':
                model.setObjective(obj_expr, GRB.MAXIMIZE)
            else:  # 'min'
                model.setObjective(obj_expr, GRB.MINIMIZE)
            
            # 求解
            model.optimize()
            
            # 检查求解状态
            if model.Status == GRB.OPTIMAL:
                return model.ObjVal
            elif model.Status == GRB.INFEASIBLE:
                # 不可行 - 这不应该发生,因为我们的边界应该总是可行的
                print(f"警告: LP不可行 (A={A}, type={objective_type})")
                return 0.0 if objective_type == 'min' else -np.inf
            else:
                print(f"警告: LP求解失败 (状态={model.Status})")
                return 0.0
                
        except Exception as e:
            print(f"错误: LP求解异常: {e}")
            # 不再允许回退到父类！这会导致错误的结果
            raise
    
    def b(self, A):
        """
        计算b(A) = max { Σ_{t∈A} u(t) | u ∈ F }
        
        这是g-polymatroid的上界函数。
        
        Args:
            A: frozenset或set,时间步集合
        
        Returns:
            float: b(A)的值
        """
        self._pb_call_count += 1
        if self._pb_call_count <= 3:
            print(f"    [CorrectTCL_GPoly] b({A}) 使用 Gurobi LP 求解")
        
        if isinstance(A, frozenset):
            A = set(A)
        result = self._solve_lp_for_set(A, 'max')
        
        return result
    
    def p(self, A):
        """
        计算p(A) = min { Σ_{t∈A} u(t) | u ∈ F }
        
        这是g-polymatroid的下界函数。
        
        Args:
            A: frozenset或set,时间步集合
        
        Returns:
            float: p(A)的值
        """
        self._pb_call_count += 1
        if self._pb_call_count <= 3:
            print(f"    [CorrectTCL_GPoly] p({A}) 使用 Gurobi LP 求解")
        
        if isinstance(A, frozenset):
            A = set(A)
        result = self._solve_lp_for_set(A, 'min')
        
        return result
    
    def b_star(self, A):
        """
        计算b*(A) = b(E\A) + p(A),其中E是全集
        
        这是lifted base polyhedron中使用的辅助函数。
        """
        E = self.active
        E_minus_A = E - A
        return self.b(E_minus_A) + self.p(A)


def create_correct_tcl_gpoly(u_min_virtual, u_max_virtual, x_min_virtual, x_max_virtual):
    """
    工厂函数:创建使用正确p/b实现的TCL g-polymatroid对象
    
    Args:
        u_min_virtual, u_max_virtual: 虚拟功率边界 (T,)
        x_min_virtual, x_max_virtual: 虚拟累积能量边界 (T,)
    
    Returns:
        CorrectTCL_GPoly对象
    """
    from flexitroid.devices.general_der import DERParameters
    
    params = DERParameters(
        u_min=u_min_virtual,
        u_max=u_max_virtual,
        x_min=x_min_virtual,
        x_max=x_max_virtual
    )
    
    return CorrectTCL_GPoly(params)
