# -*- coding: utf-8 -*-
"""
调试广义多面体坐标变换算法
"""

import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from comparison.advanced_comparison_framework import generate_realistic_tcl_data
from flexitroid.devices.general_der import GeneralDER, DERParameters
from flexitroid.aggregations.aggregator import Aggregator

# 生成单个TCL进行测试
data = generate_realistic_tcl_data(num_households=1, periods=24, sample_seed=42)
tcl = data['tcl_objs'][0]
prices = data['prices']
P0 = data['demands'][:, 0]  # 第一个TCL的基线功率
T = 24

print("="*80)
print("调试广义多面体坐标变换")
print("="*80)

# 1. 显示TCL参数
print(f"\n1. TCL物理参数:")
print(f"  a = {tcl.a:.4f}")
print(f"  delta = {tcl.delta:.4f}")
print(f"  C_th = {tcl.C_th:.2f}")
print(f"  eta = {tcl.eta:.2f}")
print(f"  P_m = {tcl.P_m:.2f}")
print(f"  theta_r = {tcl.theta_r:.2f}")
print(f"  x0 = {tcl.x0:.2f}")

# 2. 计算物理边界
P0_i = (tcl.theta_a_forecast - tcl.theta_r) / tcl.b_coef
P0_i = np.maximum(0, P0_i)
u_min_phys = -P0_i
u_max_phys = tcl.P_m - P0_i
x_plus = (tcl.C_th * tcl.tcl_params['delta_val']) / tcl.eta if tcl.eta > 0 else 100.0

print(f"\n2. 物理边界:")
print(f"  x_plus = {x_plus:.4f}")
print(f"  u_min_phys 范围: [{u_min_phys.min():.2f}, {u_min_phys.max():.2f}]")
print(f"  u_max_phys 范围: [{u_max_phys.min():.2f}, {u_max_phys.max():.2f}]")
print(f"  P0 范围: [{P0.min():.2f}, {P0.max():.2f}]")

# 3. 坐标变换到虚拟空间
a = tcl.a
delta = tcl.delta
x0 = tcl.x0

u_min_virt = np.zeros(T)
u_max_virt = np.zeros(T)
for t in range(T):
    scale = delta / (a ** t) if abs(a ** t) > 1e-10 else delta
    u_min_virt[t] = u_min_phys[t] * scale
    u_max_virt[t] = u_max_phys[t] * scale

print(f"\n3. 虚拟功率边界:")
print(f"  u_min_virt 范围: [{u_min_virt.min():.2f}, {u_min_virt.max():.2f}]")
print(f"  u_max_virt 范围: [{u_max_virt.min():.2f}, {u_max_virt.max():.2f}]")

# 4. 虚拟累积边界
y_lower_virt = np.zeros(T)
y_upper_virt = np.zeros(T)

for t in range(T):
    x_tilde_t_plus_1_min = -x_plus / (a ** (t+1)) if abs(a ** (t+1)) > 1e-10 else -x_plus
    x_tilde_t_plus_1_max = x_plus / (a ** (t+1)) if abs(a ** (t+1)) > 1e-10 else x_plus
    
    y_lower_state = x_tilde_t_plus_1_min - x0
    y_upper_state = x_tilde_t_plus_1_max - x0
    
    y_lower_power = np.sum(u_min_virt[:t+1])
    y_upper_power = np.sum(u_max_virt[:t+1])
    
    state_range = y_upper_state - y_lower_state
    power_range = y_upper_power - y_lower_power
    
    if state_range < 0.01 * abs(power_range) and power_range > 0:
        y_lower_virt[t] = y_lower_power
        y_upper_virt[t] = y_upper_power
    else:
        y_lower_virt[t] = y_lower_state
        y_upper_virt[t] = y_upper_state

print(f"\n4. 虚拟累积边界:")
print(f"  y_lower_virt 范围: [{y_lower_virt.min():.4f}, {y_lower_virt.max():.4f}]")
print(f"  y_upper_virt 范围: [{y_upper_virt.min():.4f}, {y_upper_virt.max():.4f}]")
print(f"  有效性检查: {np.all(y_lower_virt <= y_upper_virt)}")

# 5. 创建虚拟TCL并优化
params_virtual = DERParameters(
    u_min=u_min_virt,
    u_max=u_max_virt,
    x_min=y_lower_virt,
    x_max=y_upper_virt
)
tcl_virtual = GeneralDER(params_virtual)

# 虚拟成本向量
c_virtual = np.zeros(T)
for t in range(T):
    scale = (a ** t) / delta if delta > 1e-10 else 1.0
    c_virtual[t] = prices[t] * scale

print(f"\n5. 虚拟成本向量:")
print(f"  c_phys 范围: [{prices.min():.3f}, {prices.max():.3f}]")
print(f"  c_virt 范围: [{c_virtual.min():.3f}, {c_virtual.max():.3f}]")

# 虚拟优化
u0_virtual = tcl_virtual.solve_linear_program(c_virtual)
print(f"\n6. 虚拟优化结果:")
print(f"  u0_virtual 范围: [{u0_virtual.min():.4f}, {u0_virtual.max():.4f}]")
print(f"  虚拟成本: {np.dot(c_virtual, u0_virtual):.2f}")

# 7. 逆变换到物理空间
u0_physical = np.zeros(T)
for t in range(T):
    scale = (a ** t) / delta if delta > 1e-10 else 1.0
    u0_physical[t] = u0_virtual[t] * scale

print(f"\n7. 物理优化结果:")
print(f"  u0_physical 范围: [{u0_physical.min():.4f}, {u0_physical.max():.4f}]")
print(f"  物理成本: {np.dot(prices, u0_physical):.2f}")

# 8. 总功率和成本
P_total = u0_physical + P0
total_cost = np.dot(prices, P_total)
peak_power = np.max(P_total)

print(f"\n8. 最终结果:")
print(f"  P_total 范围: [{P_total.min():.2f}, {P_total.max():.2f}]")
print(f"  总成本: {total_cost:.2f}")
print(f"  峰值功率: {peak_power:.2f}")

# 9. 验证: 测试u=-P0是否可行
print(f"\n9. 验证u=-P0:")
u_test = -P0
u_test_virt = np.zeros(T)
for t in range(T):
    scale = delta / (a ** t) if abs(a ** t) > 1e-10 else delta
    u_test_virt[t] = u_test[t] * scale

# 检查虚拟功率边界
viol_u_min = np.sum(u_test_virt < u_min_virt)
viol_u_max = np.sum(u_test_virt > u_max_virt)

# 检查虚拟累积边界
u_test_cumsum = np.cumsum(u_test_virt)
viol_y_min = np.sum(u_test_cumsum < y_lower_virt)
viol_y_max = np.sum(u_test_cumsum > y_upper_virt)

print(f"  u_test 范围: [{u_test.min():.2f}, {u_test.max():.2f}]")
print(f"  u_test_virt 范围: [{u_test_virt.min():.2f}, {u_test_virt.max():.2f}]")
print(f"  违反u_min次数: {viol_u_min}")
print(f"  违反u_max次数: {viol_u_max}")
print(f"  违反y_min次数: {viol_y_min}")
print(f"  违反y_max次数: {viol_y_max}")
print(f"  u=-P0 {'不可行' if (viol_u_min + viol_u_max + viol_y_min + viol_y_max) > 0 else '可行'} ✓")

print("\n" + "="*80)
