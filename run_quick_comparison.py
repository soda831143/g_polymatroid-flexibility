"""
完整对比测试 - 简化版

只测试关键算法:
- Exact Minkowski (基准)
- No Flexibility (基准)
- G-Poly-Transform-Det (新方法)

配置: 小规模快速测试
"""
import sys
import os

# 添加路径
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from comparison import advanced_comparison_framework as acf

# 配置算法:只启用关键算法
acf.enable_only('Exact Minkowski', 'No Flexibility', 'G-Poly-Transform-Det')

# 打印状态
acf.print_algorithm_status()

# 运行对比测试
print("\n开始运行对比测试...")
print("配置: num_samples=3, num_households=10, periods=24")
print("="*80)

acf.run_advanced_comparison(
    num_samples=3,          # 小样本快速测试
    num_households=10,      # 10个TCL
    periods=24,             # 24小时
    num_days=100,           # 100天历史数据(用于JCC,虽然本次不测试)
    num_tcls=10,
    t_horizon=24
)

print("\n测试完成!")
print("查看结果:")
print("- 详细: comparison_results/advanced_comparison_results.csv")
print("- 摘要: comparison_results/advanced_summary.csv")
