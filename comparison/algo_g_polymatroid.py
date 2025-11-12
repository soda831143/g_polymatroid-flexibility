# -*- coding: utf-8 -*-
"""
G-Polymatroid Aggregation Helper Functions

This module provides common functions for the g-polymatroid based aggregation
methods, including TCL fleet creation and optimization problem solving.
"""

# PACKAGES:
from . import tools
import time
import numpy as np
import gurobipy as gp
from typing import Dict, Any, List
import sys
import os
from scipy.optimize import linprog

# Import necessary classes from the flexitroid library
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from flexitroid.devices.tcl import TCL
    from flexitroid.devices.general_der import GeneralDER, DERParameters
    from flexitroid.flexitroid import Flexitroid # 导入 Flexitroid 基类
    print("Successfully imported TCL, GeneralDER and Flexitroid modules")
except ImportError as e:
    print(f"Warning: failed to import flexitroid modules: {e}")
    # Define a dummy class if import fails, to avoid crashing the framework
    class Flexitroid:
        def solve_linear_program(self, c): pass
    class TCL(Flexitroid):
        def __init__(self, *args, **kwargs): pass
        def p(self, A): return 0
        def b(self, A): return 0
    class GeneralDER(Flexitroid):
        def __init__(self, *args, **kwargs): pass
    class DERParameters:
        def __init__(self, *args, **kwargs): pass


def create_tcl_fleet_from_data(data: Dict, build_g_poly: bool = True) -> list:
    """
    如果data中已包含tcl_objs，则直接返回。
    否则，保持原有逻辑。
    """
    if 'tcl_objs' in data:
        return data['tcl_objs']
    tcl_fleet = []
    if 'tcls' not in data:
        print("警告: 数据字典中未找到 'tcls' 键。")
        return tcl_fleet
        
    for i, tcl_params in enumerate(data['tcls']):
        try:
            # TCL类在初始化时需要显式传递theta_a_forecast
            tcl = TCL(
                tcl_params, 
                build_g_poly=build_g_poly, 
                theta_a_forecast=tcl_params['theta_a_forecast']
            )
            tcl_fleet.append(tcl)
        except Exception as e:
            print(f"警告：创建 TCL {i} 失败: {e}，跳过该设备。")
            continue
    
    return tcl_fleet

def aggregate_g_polymatroids(tcl_fleet):
    """
    以“确定的子集集合”来聚合每台设备的 p/b：
    - 单点集合 {t}
    - 前缀集合 {0,1,...,t}
    直接调用设备的 p(A)/b(A) 来避免内部字典缺项造成的失真。
    """
    if len(tcl_fleet) == 0:
        return {}, {}
    # 推断 T
    example_tcl = tcl_fleet[0]
    T = getattr(example_tcl, 'T', None) or len(getattr(example_tcl, 'theta_a_forecast', []))
    if T is None:
        raise ValueError("Cannot infer time horizon T from TCL objects")

    # 构造我们关心的子集键
    subset_keys = []
    # 单点
    for t in range(T):
        subset_keys.append(frozenset({t}))
    # 前缀
    for t in range(1, T + 1):
        subset_keys.append(frozenset(range(t)))

    p_agg: dict = {}
    b_agg: dict = {}
    for A in subset_keys:
        p_agg[A] = 0.0
        b_agg[A] = 0.0

    # 累加每台设备的精确 p/b
    for tcl in tcl_fleet:
        for A in subset_keys:
            try:
                p_val = float(tcl.p(A))
            except Exception:
                # 回退到内部字典（若存在）
                p_val = float(getattr(getattr(tcl, '_internal_g_poly', None), 'p_dict', {}).get(A, 0.0))
            try:
                b_val = float(tcl.b(A))
            except Exception:
                b_val = float(getattr(getattr(tcl, '_internal_g_poly', None), 'b_dict', {}).get(A, 0.0))
            p_agg[A] += p_val
            b_agg[A] += b_val

    return p_agg, b_agg

def solve_optimization_with_g_polymatroid(g_poly_result: Any, data: Dict, obj_type: str, 
                                        use_greedy: bool = True) -> tuple:
    """
    使用g-polymatroid聚合结果求解优化问题。
    返回优化结果以及用于解聚的信息。
    
    Args:
        g_poly_result: 聚合的p,b字典
        data: 数据字典
        obj_type: 目标类型 ('cost' 或 'peak')
        use_greedy: 是否使用贪心算法 (对成本和削峰优化都有效)
    
    Returns:
        (obj_value, computation_time, solution, success, ordering_or_info)
        其中 ordering_or_info 对于贪心算法是排序，对于Gurobi是布尔值
    """
    # 兼容tuple或dict输入以获取聚合的p和b字典
    if isinstance(g_poly_result, tuple):
        p_dict, b_dict = g_poly_result
    else:
        p_dict = g_poly_result.get('p_agg_nom', {})
        b_dict = g_poly_result.get('b_agg_nom', {})

    # 对于启用贪心算法的情况
    if use_greedy:
        try:
            # 若可用每台设备对象，按论文直接对每台设备用相同价格向量求贪心顶点并求和
            if 'tcl_objs' in data and isinstance(data['tcl_objs'], list) and len(data['tcl_objs']) > 0:
                t0 = time.time()
                if obj_type == 'cost':
                    prices = data['prices']
                    demands = data['demands']
                    demand_aggr = np.sum(demands, axis=1)
                    T = data['periods']
                    u_agg = np.zeros(T)
                    for tcl in data['tcl_objs']:
                        u_agg += tcl.solve_linear_program(prices)
                    obj = float(np.dot(prices, demand_aggr + u_agg)) * data.get('dt', 1.0)
                    ordering_full = np.argsort(np.append(prices, 0.0))
                else:  # obj_type == 'peak'
                    demands = data['demands']
                    demand_aggr = np.sum(demands, axis=1)
                    T = data['periods']
                    # 构造削峰成本向量
                    peak_cost_vector = demand_aggr + 1.0
                    u_agg = np.zeros(T)
                    for tcl in data['tcl_objs']:
                        u_agg += tcl.solve_linear_program(peak_cost_vector)
                    obj = float(np.max(demand_aggr + u_agg))
                    ordering_full = np.argsort(np.append(peak_cost_vector, 0.0))
                
                return obj, time.time() - t0, u_agg, True, ordering_full
            # 否则退化为使用聚合字典的贪心近似
            if obj_type == 'cost':
                obj_value, computation_time, solution, ordering = solve_cost_optimization_greedy(p_dict, b_dict, data)
            else:  # obj_type == 'peak'
                obj_value, computation_time, solution, ordering = solve_peak_optimization_greedy(p_dict, b_dict, data)
            return obj_value, computation_time, solution, True, ordering
        except Exception as e:
            print(f"贪心算法失败，回退到Gurobi求解: {e}")
            # 回退到Gurobi求解
            return solve_optimization_gurobi(p_dict, b_dict, data, obj_type)
    else:
        # 使用Gurobi求解
        return solve_optimization_gurobi(p_dict, b_dict, data, obj_type)


def solve_optimization_gurobi(p_dict: dict, b_dict: dict, data: dict, obj_type: str) -> tuple:
    """
    使用Gurobi求解优化问题（原有方法）
    """
    t0 = time.time()
    T = data['periods']
    prices = data['prices']
    demands = data['demands']
    demand_aggr = np.sum(demands, axis=1)

    try:
        model = gp.Model(f"G-Polymatroid_{obj_type}")
        model.Params.OutputFlag = 0
        
        u_min_agg = np.array([p_dict.get(frozenset({t}), -gp.GRB.INFINITY) for t in range(T)])
        u_max_agg = np.array([b_dict.get(frozenset({t}),  gp.GRB.INFINITY) for t in range(T)])
        u_agg = model.addMVar(shape=T, lb=u_min_agg, ub=u_max_agg, name="u_agg")
        
        for t in range(1, T + 1):
            A = frozenset(range(t))
            p_val = p_dict.get(A, -gp.GRB.INFINITY)
            b_val = b_dict.get(A, gp.GRB.INFINITY)
            model.addConstr(gp.quicksum(u_agg[i] for i in range(t)) >= p_val, name=f"p_prefix_{t}")
            model.addConstr(gp.quicksum(u_agg[i] for i in range(t)) <= b_val, name=f"b_prefix_{t}")
            
        total_power = demand_aggr + u_agg
        if obj_type == 'cost':
            objective = prices @ total_power
            model.setObjective(objective, gp.GRB.MINIMIZE)
        elif obj_type == 'peak':
            peak_var = model.addVar(lb=0.0, name="peak")
            model.addConstr(total_power <= peak_var)
            model.setObjective(peak_var, gp.GRB.MINIMIZE)
        
        model.optimize()
        computation_time = time.time() - t0

        if model.status == gp.GRB.OPTIMAL:
            obj_value = model.ObjVal
            if obj_type == 'cost':
                 obj_value *= data.get('dt', 1.0)
            # 返回：目标值，计算时间，最优解，成功标志，无排序信息
            return obj_value, computation_time, u_agg.X, True, None
        else:
            print(f"警告：G-Polymatroid {obj_type} 模型求解失败，状态: {model.status}")
            return np.nan, computation_time, None, False, None

    except Exception as e:
        print(f"G-Polymatroid {obj_type} 优化求解异常: {e}")
        return np.nan, time.time() - t0, None, False, None

def fast_support_func_value(A, U_initial):
    """
    计算集合A在椭球U_initial下的支撑函数二次型值：
    s(A) = (1_A^T) M_inv (1_A)
    其中1_A是A对应的单位向量，M_inv是椭球协方差逆矩阵。
    """
    phi = U_initial['phi']
    M_inv = U_initial['M_inv']
    # periods长度由phi长度推断
    T = len(phi)
    v = np.zeros(T)
    if len(A) > 0:
        v[list(A)] = 1.0
    return float(np.dot(np.dot(v, M_inv), v))

def get_subset_vector(A, T):
    """
    返回长度为T的0-1向量，A中元素为1，其余为0。
    """
    v = np.zeros(T)
    if len(A) > 0:
        v[list(A)] = 1.0
    return v 


def compute_vertex_from_ordering(tcl_obj: Flexitroid, ordering_full_T_star: np.ndarray) -> np.ndarray:
    """
    根据论文Theorem 5的贪心算法，基于给定排序计算g-polymatroid的顶点。
    
    Args:
        tcl_obj: 单个TCL的Flexitroid对象
        ordering_full_T_star: 排序数组，长度为 T+1，取值为 0..T，其中 T 表示 t*
    
    Returns:
        该排序下的顶点向量 v
    """
    T = tcl_obj.T
    # 确保包含 t* 的排序
    if len(ordering_full_T_star) != T + 1:
        # 如果传入的排序不含 t*，则将 t* 放在成本最小处（c*(t*)=0）
        ordering_full_T_star = np.argsort(np.append(np.zeros(T), 0.0))

    v_star = np.zeros(T + 1)
    b_star_prev = 0.0
    S_k: set = set()
    for k in ordering_full_T_star:
        S_k.add(int(k))
        b_star_k = tcl_obj._b_star(S_k)
        v_star[k] = b_star_k - b_star_prev
        b_star_prev = b_star_k

    # 投影，移除 t* 分量
    return v_star[:-1]


def solve_cost_optimization_greedy(p_dict: dict, b_dict: dict, data: dict) -> tuple:
    """
    使用改进的贪心算法和Dantzig-Wolfe分解求解成本最小化问题。
    
    参考flexitroid/problems/linear.py中的Dantzig-Wolfe分解思想，
    但针对聚合G-Polymatroid进行优化。
    """
    print("  使用改进贪心算法求解成本最小化...")
    t0 = time.time()
    
    T = data['periods']
    prices = data['prices']
    demands = data['demands']
    demand_aggr = np.sum(demands, axis=1)
    
    # 创建改进的聚合Flexitroid对象
    class ImprovedAggregatedFlexitroid:
        def __init__(self, p_dict, b_dict, T):
            self.p_dict = p_dict
            self.b_dict = b_dict
            self._T = T
        
        @property
        def T(self):
            return self._T
            
        def p(self, A):
            if not isinstance(A, set):
                A = set(A)
            return self.p_dict.get(frozenset(A), 0.0)
            
        def b(self, A):
            if not isinstance(A, set):
                A = set(A)
            return self.b_dict.get(frozenset(A), 0.0)
            
        def _b_star(self, A):
            if not isinstance(A, set):
                A = set(A)
            T_set = set(range(self.T))
            if self.T in A:  # t* is in A
                return -self.p(T_set - A)
            return self.b(A)
        
        def solve_linear_program(self, c):
            """
            改进的贪心算法，参考flexitroid/flexitroid.py的实现
            但针对聚合G-Polymatroid进行优化
            """
            # 扩展成本向量，t*的成本为0
            c_star = np.append(c, 0.0)
            
            # 按非递减成本排序
            pi = np.argsort(c_star)
            
            # 初始化解向量
            v = np.zeros(self.T + 1)
            
            # 应用贪心算法
            S_k = set()
            b_star_prev = 0.0
            
            for k in pi:
                S_k.add(int(k))
                b_star = self._b_star(S_k)
                v[k] = b_star - b_star_prev
                b_star_prev = b_star
            
            # 投影解，移除t*分量
            return v[:-1]
        
        def form_box(self):
            """
            生成边界框顶点，用于Dantzig-Wolfe分解的初始顶点集
            参考flexitroid/flexitroid.py的form_box方法
            """
            C = np.vstack([np.eye(self.T), -np.eye(self.T)])
            box_vertices = []
            for c in C:
                vertex = self.solve_linear_program(c)
                box_vertices.append(vertex)
            return np.array(box_vertices)
    
    # 创建改进的聚合对象
    agg_flexitroid = ImprovedAggregatedFlexitroid(p_dict, b_dict, T)
    
    # 尝试直接贪心求解
    try:
        u_agg_optimal = agg_flexitroid.solve_linear_program(prices)
        
        # 验证解的可行性
        is_feasible = True
        for t in range(1, T + 1):
            A = frozenset(range(t))
            prefix_sum = np.sum(u_agg_optimal[:t])
            if prefix_sum < p_dict.get(A, -np.inf) or prefix_sum > b_dict.get(A, np.inf):
                is_feasible = False
                break
        
        if not is_feasible:
            print("  贪心解不可行，尝试Dantzig-Wolfe分解...")
            # 回退到Dantzig-Wolfe分解
            u_agg_optimal = _solve_with_dantzig_wolfe(agg_flexitroid, prices, T)
            
    except Exception as e:
        print(f"  贪心算法失败: {e}，使用Dantzig-Wolfe分解...")
        u_agg_optimal = _solve_with_dantzig_wolfe(agg_flexitroid, prices, T)
    
    # 计算目标值（使用原始电价）
    total_power = demand_aggr + u_agg_optimal
    obj_value = float(np.dot(prices, total_power)) * data.get('dt', 1.0)
    
    # 获取包含 t* 的排序信息（用于顶点法解聚）
    ordering = np.argsort(np.append(prices, 0.0))
    
    computation_time = time.time() - t0
    
    # 返回：目标值，计算时间，最优解，排序（用于解聚）
    return obj_value, computation_time, u_agg_optimal, ordering


def _solve_with_dantzig_wolfe(agg_flexitroid, c, T, max_iter=100, epsilon=1e-6):
    """
    使用Dantzig-Wolfe分解求解聚合G-Polymatroid上的线性规划
    参考flexitroid/problems/linear.py的实现
    """
    print("  使用Dantzig-Wolfe分解求解...")
    
    # 初始化顶点集
    V_subset = agg_flexitroid.form_box()
    
    for i in range(max_iter):
        # 求解限制主问题 (RMP)
        try:
            # 简化版RMP：直接求解凸组合
            num_vertices = V_subset.shape[0]
            if num_vertices == 0:
                break
                
            # 计算每个顶点的成本
            c_V = np.array([np.dot(c, v) for v in V_subset])
            
            # 找到成本最小的顶点
            min_idx = np.argmin(c_V)
            u_agg_optimal = V_subset[min_idx]
            
            # 检查是否需要添加新顶点
            if i == 0:  # 第一次迭代，直接返回
                break
                
            # 计算reduced cost
            d = c - np.dot(c_V[min_idx], np.ones_like(c))
            new_vertex = agg_flexitroid.solve_linear_program(d)
            
            # 如果新顶点不能改善解，则停止
            if np.dot(d, new_vertex) >= -epsilon:
                break
                
            # 添加新顶点
            V_subset = np.vstack([V_subset, new_vertex])
            
        except Exception as e:
            print(f"  Dantzig-Wolfe迭代 {i} 失败: {e}")
            break
    
    return u_agg_optimal


def disaggregate_vertex_based(u_agg_optimal: np.ndarray, tcl_fleet: List[Flexitroid], 
                             ordering_full_T_star: np.ndarray, T: int) -> Dict[int, np.ndarray]:
    """
    基于论文Theorem 7的顶点法解聚。
    
    对于成本最小化等线性问题，最优解是一个顶点，可以表示为：
    u_N = Σ_i u_i，其中每个 u_i 是设备i在给定排序下的顶点。
    
    改进版本：支持Dantzig-Wolfe分解结果的顶点凸组合解聚。
    
    Args:
        u_agg_optimal: 聚合最优解
        tcl_fleet: TCL设备列表
        ordering_full_T_star: 产生最优解的排序（长度 T+1，含 t*）
        T: 时间周期数
    
    Returns:
        解聚后的设备功率曲线字典
    """
    print("  开始执行顶点法解聚...")
    t_start = time.time()
    
    try:
        # 方法1：直接顶点法（适用于纯贪心算法）
        disaggregated_profiles: Dict[int, np.ndarray] = {}
        
        # 为每个设备计算在给定排序下的顶点
        for i, tcl in enumerate(tcl_fleet):
            u_i = compute_vertex_from_ordering(tcl, ordering_full_T_star)
            disaggregated_profiles[i] = u_i
        
        # 验证解聚结果
        validation_sum = np.sum(list(disaggregated_profiles.values()), axis=0)
        if np.allclose(validation_sum, u_agg_optimal, atol=1e-5):
            t_end = time.time()
            print(f"  顶点法解聚成功，耗时 {t_end - t_start:.4f} 秒。")
            return disaggregated_profiles
        
        print("  直接顶点法验证失败，尝试改进的顶点法...")
        
        # 方法2：改进的顶点法（适用于Dantzig-Wolfe分解结果）
        # 如果直接顶点法失败，说明最优解可能是多个顶点的凸组合
        # 我们需要找到每个设备对应的顶点权重
        
        # 使用Dantzig-Wolfe思想：为每个设备找到最优的顶点组合
        improved_profiles = _improved_vertex_disaggregation(u_agg_optimal, tcl_fleet, T, ordering_full_T_star)
        
        if improved_profiles is not None:
            t_end = time.time()
            print(f"  改进顶点法解聚成功，耗时 {t_end - t_start:.4f} 秒。")
            return improved_profiles
        
        # 如果所有顶点法都失败，回退到LP解聚
        print("  所有顶点法都失败，回退到可行性LP解聚...")
        return disaggregate_profile_lp_fallback(u_agg_optimal, tcl_fleet, T)
        
    except Exception as e:
        print(f"顶点法解聚异常: {e}")
        # 回退到可行性LP解聚
        return disaggregate_profile_lp_fallback(u_agg_optimal, tcl_fleet, T)


def _improved_vertex_disaggregation(u_agg_optimal: np.ndarray, tcl_fleet: List[Flexitroid], 
                                   T: int, ordering_full_T_star: np.ndarray) -> Dict[int, np.ndarray]:
    """
    改进的顶点法解聚，使用Dantzig-Wolfe分解思想。
    
    对于复杂的聚合g-polymatroid，最优解可能是多个顶点的凸组合。
    我们为每个设备找到最优的顶点权重。
    """
    try:
        num_devices = len(tcl_fleet)
        
        # 为每个设备生成候选顶点集
        device_vertices = {}
        for i, tcl in enumerate(tcl_fleet):
            # 生成多个候选顶点
            vertices = []
            
            # 1. 基于给定排序的顶点
            v1 = compute_vertex_from_ordering(tcl, ordering_full_T_star)
            vertices.append(v1)
            
            # 2. 基于反向排序的顶点
            reverse_ordering = np.flip(ordering_full_T_star)
            v2 = compute_vertex_from_ordering(tcl, reverse_ordering)
            vertices.append(v2)
            
            # 3. 基于成本排序的顶点（如果排序包含成本信息）
            if len(ordering_full_T_star) == T + 1:
                # 构造成本向量
                cost_vector = np.zeros(T)
                for t in range(T):
                    if t in ordering_full_T_star:
                        cost_vector[t] = np.where(ordering_full_T_star == t)[0][0]
                    else:
                        cost_vector[t] = T  # 高成本
                v3 = tcl.solve_linear_program(cost_vector)
                vertices.append(v3)
            
            device_vertices[i] = np.array(vertices)
        
        # 构建顶点组合优化问题
        # 目标：找到每个设备的顶点权重，使得总和最接近u_agg_optimal
        try:
            import gurobipy as gp
            
            model = gp.Model("ImprovedVertexDisaggregation")
            model.Params.OutputFlag = 0
            
            # 变量：每个设备在每个顶点上的权重
            lambda_vars = {}
            for i in range(num_devices):
                num_vertices_i = device_vertices[i].shape[0]
                lambda_vars[i] = model.addMVar(shape=num_vertices_i, lb=0.0, ub=1.0, name=f"lambda_{i}")
                # 权重和为1约束
                model.addConstr(gp.quicksum(lambda_vars[i]) == 1.0, name=f"convex_{i}")
            
            # 目标：最小化与u_agg_optimal的差异
            deviation = model.addMVar(shape=T, lb=-gp.GRB.INFINITY, name="deviation")
            
            for t in range(T):
                # 计算每个设备在时间t的功率
                device_power_t = gp.quicksum(
                    lambda_vars[i] @ device_vertices[i][:, t]
                    for i in range(num_devices)
                )
                # 与聚合解的差异
                model.addConstr(device_power_t - u_agg_optimal[t] == deviation[t], name=f"dev_t{t}")
            
            # 最小化总偏差
            model.setObjective(gp.quicksum(gp.abs_(deviation)), gp.GRB.MINIMIZE)
            model.optimize()
            
            if model.status == gp.GRB.OPTIMAL:
                # 提取解聚结果
                disaggregated_profiles = {}
                for i in range(num_devices):
                    lambda_opt = lambda_vars[i].X
                    # 计算加权顶点组合
                    u_i = np.zeros(T)
                    for j, vertex in enumerate(device_vertices[i]):
                        u_i += lambda_opt[j] * vertex
                    disaggregated_profiles[i] = u_i
                
                return disaggregated_profiles
                
        except ImportError:
            print("  Gurobi不可用，使用简化方法...")
            pass
        
        # 简化方法：直接选择最接近的顶点
        disaggregated_profiles = {}
        for i in range(num_devices):
            vertices = device_vertices[i]
            # 选择与u_agg_optimal最接近的顶点
            best_vertex_idx = 0
            min_error = np.inf
            for j, vertex in enumerate(vertices):
                error = np.linalg.norm(vertex - u_agg_optimal / num_devices)
                if error < min_error:
                    min_error = error
                    best_vertex_idx = j
            disaggregated_profiles[i] = vertices[best_vertex_idx]
        
        return disaggregated_profiles
        
    except Exception as e:
        print(f"改进顶点法解聚失败: {e}")
        return None


def disaggregate_profile_lp_fallback(u_agg_optimal: np.ndarray, tcl_fleet: List[Flexitroid], T: int) -> Dict[int, np.ndarray]:
    """
    可行性LP解聚方法（作为顶点法的回退）
    """
    print("  使用可行性LP解聚作为回退方法...")
    t_start = time.time()

    try:
        model = gp.Model("DisaggregationLP")
        model.Params.OutputFlag = 0

        num_devices = len(tcl_fleet)

        # 变量 u[i,t]
        u_vars = []  # list of gp.MVar per device
        for i, tcl in enumerate(tcl_fleet):
            p_i = getattr(getattr(tcl, "_internal_g_poly", None), "p_dict", {}) or {}
            b_i = getattr(getattr(tcl, "_internal_g_poly", None), "b_dict", {}) or {}

            # 单点上下界
            lb = np.array([p_i.get(frozenset({t}), -gp.GRB.INFINITY) for t in range(T)])
            ub = np.array([b_i.get(frozenset({t}),  gp.GRB.INFINITY) for t in range(T)])
            u_i = model.addMVar(shape=T, lb=lb, ub=ub, name=f"u_{i}")
            u_vars.append(u_i)

            # 前缀约束
            for t in range(1, T + 1):
                A = frozenset(range(t))
                p_pref = p_i.get(A, -gp.GRB.INFINITY)
                b_pref = b_i.get(A,  gp.GRB.INFINITY)
                model.addConstr(gp.quicksum(u_i[k] for k in range(t)) >= p_pref, name=f"p_i{i}_t{t}")
                model.addConstr(gp.quicksum(u_i[k] for k in range(t)) <= b_pref, name=f"b_i{i}_t{t}")

        # 聚合匹配约束（允许极小松弛，以抵抗数值问题）
        s_pos = model.addMVar(shape=T, lb=0.0, name="s_pos")
        s_neg = model.addMVar(shape=T, lb=0.0, name="s_neg")
        for t in range(T):
            model.addConstr(
                gp.quicksum(u_vars[i][t] for i in range(num_devices)) - float(u_agg_optimal[t]) == s_pos[t] - s_neg[t],
                name=f"agg_t{t}"
            )

        # 目标函数: 最小化松弛之和
        model.setObjective(gp.quicksum(s_pos) + gp.quicksum(s_neg), gp.GRB.MINIMIZE)
        model.optimize()

        if model.status != gp.GRB.OPTIMAL:
            print(f"警告: 解聚可行性LP未能获得最优解，状态: {model.status}")
            return None

        disaggregated_profiles: Dict[int, np.ndarray] = {}
        for i, u_i in enumerate(u_vars):
            disaggregated_profiles[i] = u_i.X.copy()

        t_end = time.time()
        print(f"  可行性LP解聚完成，耗时 {t_end - t_start:.4f} 秒。")

        return disaggregated_profiles

    except Exception as e:
        print(f"可行性LP解聚异常: {e}")
        return None


def disaggregate_profile(u_agg_optimal: np.ndarray, tcl_fleet: List[Flexitroid], T: int, 
                        ordering: np.ndarray = None, method: str = "vertex") -> Dict[int, np.ndarray]:
    """
    统一的解聚接口，支持顶点法和可行性LP两种方法。
    
    Args:
        u_agg_optimal: 聚合最优解
        tcl_fleet: TCL设备列表 
        T: 时间周期数
        ordering: 顶点排序（仅顶点法需要）
        method: 解聚方法，"vertex" 或 "lp"
    
    Returns:
        解聚后的设备功率曲线字典
    """
    if method == "vertex" and ordering is not None:
        # 确保传入的是 T+1 长度的排序（含 t*）
        if len(ordering) == T:
            # 如果传入的是T维排序，构造T+1维排序，t*放在成本最小处
            ordering_full = np.argsort(np.append(ordering, 0.0))
        elif len(ordering) == T + 1:
            # 如果已经是T+1维，直接使用
            ordering_full = ordering
        else:
            # 其他情况，构造标准排序
            ordering_full = np.argsort(np.append(np.arange(T), 0.0))
        
        return disaggregate_vertex_based(u_agg_optimal, tcl_fleet, ordering_full, T)
    else:
        return disaggregate_profile_lp_fallback(u_agg_optimal, tcl_fleet, T)


def solve_peak_optimization_greedy(p_dict: dict, b_dict: dict, data: dict) -> tuple:
    """
    使用贪心算法求解削峰优化问题。
    
    削峰优化等价于：min max(total_power) = min max(demand + u)
    这可以通过贪心算法求解，类似于成本优化。
    """
    print("  使用贪心算法求解削峰优化...")
    t0 = time.time()
    
    T = data['periods']
    demands = data['demands']
    demand_aggr = np.sum(demands, axis=1)
    
    # 创建聚合Flexitroid对象
    class AggregatedFlexitroid:
        def __init__(self, p_dict, b_dict, T):
            self.p_dict = p_dict
            self.b_dict = b_dict
            self._T = T
        
        @property
        def T(self):
            return self._T
            
        def p(self, A):
            if not isinstance(A, set):
                A = set(A)
            return self.p_dict.get(frozenset(A), 0.0)
            
        def b(self, A):
            if not isinstance(A, set):
                A = set(A)
            return self.b_dict.get(frozenset(A), 0.0)
            
        def _b_star(self, A):
            if not isinstance(A, set):
                A = set(A)
            T_set = set(range(self.T))
            if self.T in A:  # t* is in A
                return -self.p(T_set - A)
            return self.b(A)
        
        def solve_linear_program(self, c):
            # 使用与Flexitroid基类相同的贪心算法
            c_star = np.append(c, 0)
            pi = np.argsort(c_star)
            v = np.zeros(self.T + 1)
            S_k = set()
            b_star_prev = 0
            for k in pi:
                S_k.add(int(k))
                b_star = self._b_star(S_k)
                v[k] = b_star - b_star_prev
                b_star_prev = b_star
            return v[:-1]  # 投影到T维
    
    # 对于削峰优化，我们需要构造一个更智能的成本向量
    # 目标是让功率在时间上尽可能均匀分布，减少峰值
    
    # 方法1：基于当前需求的"反峰值"成本
    # 需求高的时段成本高，引导灵活性向低需求时段转移
    peak_cost_vector = np.zeros(T)
    max_demand = np.max(demand_aggr)
    min_demand = np.min(demand_aggr)
    demand_range = max_demand - min_demand
    
    if demand_range > 1e-6:
        # 标准化需求到[0,1]范围，然后反转（高需求→高成本）
        normalized_demand = (demand_aggr - min_demand) / demand_range
        peak_cost_vector = normalized_demand + 1.0  # 确保成本为正
    else:
        # 如果需求均匀，使用时间索引作为成本（引导灵活性向早期时段转移）
        peak_cost_vector = np.arange(T) + 1.0
    
    # 方法2：迭代优化（如果方法1效果不好）
    best_peak = np.inf
    best_solution = None
    best_ordering = None
    
    # 尝试不同的成本向量策略
    cost_strategies = [
        peak_cost_vector,  # 方法1：基于需求
        np.flip(peak_cost_vector),  # 方法2：反向策略
        np.ones(T),  # 方法3：均匀成本
    ]
    
    for i, cost_vector in enumerate(cost_strategies):
        try:
            # 创建聚合对象并求解
            agg_flexitroid = AggregatedFlexitroid(p_dict, b_dict, T)
            u_agg_optimal = agg_flexitroid.solve_linear_program(cost_vector)
            
            # 计算目标值：最大总功率
            total_power = demand_aggr + u_agg_optimal
            current_peak = np.max(total_power)
            
            # 更新最佳解
            if current_peak < best_peak:
                best_peak = current_peak
                best_solution = u_agg_optimal
                best_ordering = np.argsort(np.append(cost_vector, 0.0))
                
        except Exception as e:
            print(f"    削峰策略 {i+1} 失败: {e}")
            continue
    
    if best_solution is None:
        print("  所有削峰策略都失败，回退到Gurobi求解")
        return solve_optimization_gurobi(p_dict, b_dict, data, 'peak')
    
    computation_time = time.time() - t0
    
    # 返回：峰值，计算时间，最优解，排序（用于解聚）
    return best_peak, computation_time, best_solution, best_ordering