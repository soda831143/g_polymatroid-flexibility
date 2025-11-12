# -*- coding: utf-8 -*-
"""
G-Polymatroid Aggregation Helper Functions (Revised)

This module provides dedicated, theoretically-sound solvers for different stages 
of the aggregation process.
"""

# PACKAGES:
from . import tools
import time
import numpy as np
try:
    import gurobipy as gp
except ImportError:
    gp = None
except ImportError:
    pass

try:
    from gurobipy import GRB
except ImportError:
    # GRB constants defined as None
    GRB = None
from typing import Dict, Any, List, Optional
import sys
import os

# Import necessary classes from the flexitroid library
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from flexitroid.devices.tcl import TCL
    from flexitroid.flexitroid import Flexitroid
    print("✓ 成功导入 TCL 和 Flexitroid 模块")
except ImportError as e:
    print(f"警告：无法导入 flexitroid 模块: {e}")
    class Flexitroid:
        def solve_linear_program(self, c): return np.zeros(1)
        @property
        def T(self): return 1
    class TCL(Flexitroid):
        def __init__(self, *args, **kwargs): pass
        def p(self, A): return 0
        def b(self, A): return 0

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

def aggregate_g_polymatroids(tcl_fleet: List[TCL]) -> tuple[dict, dict]:
    """
    聚合TCL群体的名义p和b函数值。
    注意：这只聚合了TCL对象内部缓存的子集，主要用于后续的鲁棒计算。
    """
    p_agg, b_agg = {}, {}
    all_subsets = set()
    for tcl in tcl_fleet:
        if hasattr(tcl, '_internal_g_poly'):
            all_subsets.update(tcl._internal_g_poly.p_dict.keys())
            all_subsets.update(tcl._internal_g_poly.b_dict.keys())

    for A in all_subsets:
        p_sum = sum(tcl._internal_g_poly.p_dict.get(A, 0) for tcl in tcl_fleet if hasattr(tcl, '_internal_g_poly'))
        b_sum = sum(tcl._internal_g_poly.b_dict.get(A, 0) for tcl in tcl_fleet if hasattr(tcl, '_internal_g_poly'))
        p_agg[A] = p_sum
        b_agg[A] = b_sum
    return p_agg, b_agg

def solve_with_greedy(data: Dict, obj_type: str, p_final: Optional[Dict] = None, b_final: Optional[Dict] = None, final_aggregator: Optional[Flexitroid] = None) -> tuple:
    """
    【理论正确】使用贪心算法在g-polymatroid上求解优化问题。
    适用于确定性聚合（approximate）和最终鲁棒聚合（Re-SRA）。
    """
    t0 = time.time()
    tcl_fleet = data['tcl_objs']
    T = data['periods']
    demands = data['demands']
    demand_aggr = np.sum(demands, axis=1)

    try:
        # 确定成本向量
        if obj_type == 'cost':
            cost_vector = data['prices']
        elif obj_type == 'peak':
            # 为确保准确性，peak 不使用贪心启发式，转到LP精确求解
            cost_vector = None
        else:
            raise ValueError(f"未知的优化目标类型: {obj_type}")

        # --- 求解 ---
        if obj_type == 'peak':
            # 精确LP建模（确定性或鲁棒版本）
            model = gp.Model(f"PeakLP_{'final' if (p_final or b_final) else 'det'}")
            model.Params.OutputFlag = 0
            # 聚合前缀约束来源
            if final_aggregator is not None:
                def p_get(A):
                    return final_aggregator.p(A)
                def b_get(A):
                    return final_aggregator.b(A)
            else:
                if p_final is None and b_final is None:
                    p_agg, b_agg = aggregate_g_polymatroids(tcl_fleet)
                else:
                    p_agg, b_agg = p_final, b_final
                def p_get(A):
                    return p_agg.get(A, -GRB.INFINITY)
                def b_get(A):
                    return b_agg.get(A, GRB.INFINITY)
            u = model.addMVar(T, lb=[p_get(frozenset({t})) for t in range(T)],
                              ub=[b_get(frozenset({t})) for t in range(T)], name='u')
            for t in range(1, T+1):
                A = frozenset(range(t))
                p_val = p_get(A)
                b_val = b_get(A)
                if p_val > -GRB.INFINITY:
                    model.addConstr(gp.quicksum(u[i] for i in range(t)) >= p_val)
                if b_val < GRB.INFINITY:
                    model.addConstr(gp.quicksum(u[i] for i in range(t)) <= b_val)
            total_power = demand_aggr + u
            peak = model.addVar(lb=0.0, name='peak')
            model.addConstr(total_power <= peak)
            model.addConstr(total_power >= -peak)
            model.setObjective(peak, GRB.MINIMIZE)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                u_agg_optimal = u.X
            else:
                print(f"[PeakLP] 求解失败，状态: {model.status}")
                return np.nan, time.time() - t0, None
        else:
            u_agg_optimal = np.zeros(T)
            if final_aggregator is not None:
                # 对最终鲁棒集合，使用LP精确建模（避免 _b_star 缺失导致的错误）
                print(f"  正在为 '{obj_type}' 目标执行最终鲁棒LP求解...")
                model = gp.Model(f"CostLP_final")
                model.Params.OutputFlag = 0
                def p_get(A): return final_aggregator.p(A)
                def b_get(A): return final_aggregator.b(A)
                u = model.addMVar(T, lb=[p_get(frozenset({t})) for t in range(T)],
                                  ub=[b_get(frozenset({t})) for t in range(T)], name='u')
                for t in range(1, T+1):
                    A = frozenset(range(t))
                    p_val = p_get(A)
                    b_val = b_get(A)
                    if p_val > -GRB.INFINITY:
                        model.addConstr(gp.quicksum(u[i] for i in range(t)) >= p_val)
                    if b_val < GRB.INFINITY:
                        model.addConstr(gp.quicksum(u[i] for i in range(t)) <= b_val)
                total_power = demand_aggr + u
                # 物理约束：总功率不可为负
                # model.addConstr(total_power >= 0)
                objective = data['prices'] @ total_power
                model.setObjective(objective, GRB.MINIMIZE)
                model.optimize()
                if model.status == GRB.OPTIMAL:
                    u_agg_optimal = u.X
                else:
                    print(f"[CostLP_final] 求解失败，状态: {model.status}")
                    return np.nan, time.time() - t0, None
            elif p_final is None and b_final is None:
                print(f"  正在为 '{obj_type}' 目标执行确定性贪心求解...")
                for tcl in tcl_fleet:
                    u_agg_optimal += tcl.solve_linear_program(cost_vector)
            else:
                print(f"  正在为 '{obj_type}' 目标执行最终鲁棒贪心求解...")
                class FinalRobustAggregator(Flexitroid):
                    def __init__(self, p_dict, b_dict, T_val):
                        self.p_dict = p_dict; self.b_dict = b_dict; self._T = T_val
                    def p(self, A): return self.p_dict.get(frozenset(A), 0.0)
                    def b(self, A): return self.b_dict.get(frozenset(A), 0.0)
                    @property
                    def T(self): return self._T
                final_aggregator = FinalRobustAggregator(p_final, b_final, T)
                u_agg_optimal = final_aggregator.solve_linear_program(cost_vector)
        
        # 计算目标函数值
        total_power = demand_aggr + u_agg_optimal
        if obj_type == 'cost':
            obj_value = float(np.dot(data['prices'], total_power)) * data.get('dt', 1.0)
        else: # peak
            obj_value = float(np.max(total_power))

        return obj_value, time.time() - t0, u_agg_optimal
        
    except Exception as e:
        print(f"贪心算法求解失败: {e}")
        import traceback
        traceback.print_exc()
        return np.nan, time.time() - t0, None

def solve_sra_for_initial_solution(p_sro: Dict, b_sro: Dict, data: Dict, obj_type: str) -> tuple:
    """
    【理论正确】使用Gurobi求解SRA问题以获得初始解 u0。
    该模型只包含前缀和单点约束，用于在一个非g-polymatroid的凸集上找到一个可行解。
    """
    t0 = time.time()
    T = data['periods']
    demands = data['demands']
    demand_aggr = np.sum(demands, axis=1)

    try:
        model = gp.Model(f"SRA_for_u0_{obj_type}")
        model.Params.OutputFlag = 0
        
        u_min_agg = np.array([p_sro.get(frozenset({t}), -GRB.INFINITY) for t in range(T)])
        u_max_agg = np.array([b_sro.get(frozenset({t}),  GRB.INFINITY) for t in range(T)])
        u_agg = model.addMVar(shape=T, lb=u_min_agg, ub=u_max_agg, name="u_agg")
        
        for t in range(1, T + 1):
            A = frozenset(range(t))
            p_val = p_sro.get(A, -GRB.INFINITY)
            b_val = b_sro.get(A, GRB.INFINITY)
            if p_val > -GRB.INFINITY:
                model.addConstr(gp.quicksum(u_agg[i] for i in range(t)) >= p_val)
            if b_val < GRB.INFINITY:
                model.addConstr(gp.quicksum(u_agg[i] for i in range(t)) <= b_val)
        
        total_power = demand_aggr + u_agg
        if obj_type == 'cost':
            objective = data['prices'] @ total_power
            model.setObjective(objective, GRB.MINIMIZE)
        elif obj_type == 'peak':
            peak_var = model.addVar(lb=0.0, name="peak")
            model.addConstr(total_power <= peak_var)
            model.setObjective(peak_var, GRB.MINIMIZE)
        
        model.optimize()
        
        computation_time = time.time() - t0
        if model.status == GRB.OPTIMAL:
            obj_value = model.ObjVal
            if obj_type == 'cost':
                obj_value *= data.get('dt', 1.0)
            return u_agg.X, obj_value, computation_time
        else:
            return None, np.nan, computation_time

    except Exception as e:
        print(f"SRA Gurobi求解器异常: {e}")
        return None, np.nan, time.time() - t0

def get_subset_vector(A, T):
    """返回长度为T的0-1向量，A中元素为1，其余为0。"""
    v = np.zeros(T)
    if A: # 确保A非空
        v[list(A)] = 1.0
    return v

def fast_support_func_value(A, U_initial):
    """
    计算集合A在椭球U_initial下的支撑函数二次型值：
    s(A) = (1_A^T) M_inv (1_A)
    其中 1_A 是A对应的单位向量，M_inv 是椭球协方差逆矩阵。
    """
    M_inv = U_initial['M_inv']
    # periods长度由M_inv形状推断
    T = M_inv.shape[0]
    v = np.zeros(T)
    if A:
        v[list(A)] = 1.0
    return float(v @ M_inv @ v)