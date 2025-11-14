# -*- coding: utf-8 -*-
"""
诊断脚本：分析为什么G-Poly峰值优化没有达到Exact的结果

检查点：
1. 坐标变换是否完全可逆（应该是，因为成本优化成功了）
2. 列生成收敛条件是否合理
3. 顶点数量限制是否过严
4. 初始顶点生成策略是否覆盖了最优解附近
"""

import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if project_root not in sys.path: sys.path.insert(0, project_root)
if lib_path not in sys.path: sys.path.insert(0, lib_path)

print("="*80)
print("诊断 G-Polymatroid 峰值优化问题")
print("="*80)

print("\n观察到的问题:")
print("  - 成本优化: G-Poly = Exact = 46.21 (完美!)")
print("  - 峰值优化: G-Poly = 11.79, Exact = 10.60 (差距11.2%)")
print("  - 列生成在第47次迭代时因顶点数>50而停止")
print("  - 此时 ReducedCost = -5.24 < 0, 说明还能改进")

print("\n可能原因:")
print("  1. 硬编码的顶点数量限制 (max_vertices=50)")
print("  2. 峰值优化比成本优化需要更多顶点")
print("  3. 初始顶点生成策略可能不够好")

print("\n解决方案:")
print("  方案1: 增加顶点数量限制到100或更高")
print("  方案2: 改进初始顶点生成,更好地覆盖极端情况")
print("  方案3: 调整收敛容差tolerance (当前1e-3)")
print("  方案4: 使用自适应容差 (如|RC|/|ObjVal| < 0.01)")

print("\n理论分析:")
print("  - 成本优化是线性目标 → 少量顶点即可达到最优")
print("  - 峰值优化是min-max目标 → 需要更多顶点来精确平衡各时间步")
print("  - 从输出看,峰值从26.73降到10.85,需要探索更多方向")

print("\n推荐修复:")
print("  1. 将 max_vertices 从50增加到200")
print("  2. 或使用相对容差: abs(reduced_cost) / abs(current_peak) < 0.01")
print("  3. 这样既能保证收敛,又不会过早终止")

print("\n注意:")
print("  - 这不是坐标变换的问题(成本优化已证明变换正确)")
print("  - 这是列生成算法参数设置的问题")
print("  - 修复后应该能达到与Exact相同或非常接近的峰值")

print("="*80)
