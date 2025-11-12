"""
测试新创建的模块是否能正常导入和基础功能
"""

import numpy as np
import sys
import os

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_jcc_robust_bounds():
    """测试JCC鲁棒边界计算模块"""
    print("\n" + "="*70)
    print("测试 JCC Robust Bounds 模块")
    print("="*70)
    
    try:
        # 先检查依赖
        try:
            import cvxpy as cp
            has_cvxpy = True
        except ImportError:
            has_cvxpy = False
            print("⚠ cvxpy未安装,跳过优化测试")
        
        from flexitroid.problems.jcc_robust_bounds import TemperatureUncertaintySet
        
        # 创建简单测试数据
        T = 24
        n_shape = 50
        n_calib = 50
        
        D_shape = np.random.randn(n_shape, T) * 2.0
        D_calib = np.random.randn(n_calib, T) * 2.0
        
        epsilon = 0.05
        delta = 0.05
        
        print(f"✓ 成功导入模块")
        print(f"  创建TemperatureUncertaintySet实例...")
        
        # 测试不确定性集构建
        temp_set = TemperatureUncertaintySet(
            D_shape=D_shape,
            D_calib=D_calib,
            epsilon=epsilon,
            delta=delta,
            use_full_cov=True
        )
        
        print(f"  ✓ TemperatureUncertaintySet创建成功")
        print(f"    - 形状集样本数: {temp_set.n_shape}")
        print(f"    - 校准集样本数: {temp_set.n_calib}")
        print(f"    - 维度 T: {temp_set.T}")
        print(f"    - epsilon (JCC违反概率): {temp_set.epsilon}")
        print(f"    - delta (统计置信度): {temp_set.delta}")
        
        # 如果有cvxpy,可以进一步测试
        if has_cvxpy:
            print(f"  ✓ cvxpy可用,模块功能完整")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coordinate_transform():
    """测试坐标变换模块"""
    print("\n" + "="*70)
    print("测试 Coordinate Transform 模块")
    print("="*70)
    
    try:
        from flexitroid.utils.coordinate_transform import CoordinateTransformer
        
        print(f"✓ 成功导入模块")
        
        # 创建简单的mock TCL对象
        class MockTCL:
            def __init__(self, T, a, delta):
                self.T = T
                self.a = a
                self.delta = delta
        
        T = 24
        N = 3
        a = 0.95
        delta_val = 1.0
        
        print(f"  创建Mock TCL对象...")
        tcl_fleet = [MockTCL(T, a, delta_val) for _ in range(N)]
        
        print(f"  创建CoordinateTransformer实例...")
        transformer = CoordinateTransformer(tcl_fleet)
        
        print(f"  ✓ CoordinateTransformer创建成功")
        print(f"    - TCL数量: {transformer.N}")
        print(f"    - 时间步数 T: {transformer.T}")
        
        # 注意:完整的变换测试需要实际的TCL对象和鲁棒边界
        # 这里只测试模块可以正确导入和初始化
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("="*70)
    print("模块单元测试")
    print("="*70)
    
    results = []
    
    # 测试各个模块
    results.append(("JCC Robust Bounds", test_jcc_robust_bounds()))
    results.append(("Coordinate Transform", test_coordinate_transform()))
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ 所有测试通过!")
        return 0
    else:
        print("\n✗ 部分测试失败")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
