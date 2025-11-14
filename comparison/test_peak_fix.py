# -*- coding: utf-8 -*-
"""
测试峰值优化修复是否有效
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if project_root not in sys.path: sys.path.insert(0, project_root)
if lib_path not in sys.path: sys.path.insert(0, lib_path)

from advanced_comparison_framework import (
    enable_only, run_advanced_comparison, print_algorithm_status
)

print("="*80)
print("测试峰值优化修复")
print("="*80)
print("\n只启用 Exact Minkowski, No Flexibility 和 G-Poly-Transform-Det")

# 只测试基准和G-Poly算法
enable_only('Exact Minkowski', 'No Flexibility', 'G-Poly-Transform-Det')
print_algorithm_status()

print("\n运行小规模测试 (5 TCLs, 12 periods)...")
run_advanced_comparison(
    num_samples=1, 
    num_households=5,
    periods=12,
    num_days=100,  # 减少不确定性数据生成时间
    num_tcls=5,
    t_horizon=12
)

print("\n检查峰值优化结果是否接近Exact...")
