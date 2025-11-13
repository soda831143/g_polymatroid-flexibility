# -*- coding: utf-8 -*-
"""
快速测试广义多面体算法
"""

import sys
import os

# 添加路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置算法选择
from comparison.advanced_comparison_framework import (
    enable_only, run_advanced_comparison
)

# 只测试基准算法 + 确定性坐标变换
enable_only('Exact Minkowski', 'No Flexibility', 'G-Poly-Transform-Det')

# 运行快速测试
run_advanced_comparison(
    num_samples=1,
    num_households=5,
    periods=24,
    num_days=100,  # 减少天数加快速度
    num_tcls=5,
    t_horizon=24
)
