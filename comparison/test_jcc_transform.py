"""
测试新方案: JCC → 坐标变换 → 聚合

简单的端到端测试,验证新方案的正确性
"""

import numpy as np
import sys
import os

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lib import algo_g_polymatroid_jcc_transform as new_algo

def generate_test_data():
    """生成简单的测试数据"""
    T = 24  # 时间步数
    N = 10  # TCL数量
    
    # TCL参数
    tcls = []
    for i in range(N):
        theta_a_forecast = 30 + 5 * np.sin(np.linspace(0, 2*np.pi, T)) + np.random.randn(T) * 0.5
        
        tcl_params = {
            'T': T,
            'a': 0.95,  # 衰减因子
            'delta': 1.0,
            'C_th': 2.0,
            'eta': 2.5,
            'b': 10.0,  # 热阻
            'P_m': 5.0,  # 最大功率
            'theta_r': 22.0,  # 参考温度
            'x0': 0.0,
            'delta_val': 2.0,
            'theta_a_forecast': theta_a_forecast
        }
        tcls.append(tcl_params)
    
    # 不确定性数据 (温度预测误差)
    n_shape = 100  # 形状集样本数
    n_calib = 100  # 校准集样本数
    
    # 生成相关的温度误差 (使用简单的AR模型)
    np.random.seed(42)
    D_shape = np.random.randn(n_shape, T) * 2.0
    D_calib = np.random.randn(n_calib, T) * 2.0
    
    # 其他参数
    demands = np.random.rand(1, T) * 50  # 需求
    prices = 10 + 5 * np.sin(np.linspace(0, 2*np.pi, T))  # 电价
    
    data = {
        'tcls': tcls,
        'periods': T,
        'D_shape': D_shape,
        'D_calib': D_calib,
        'epsilon': 0.05,  # JCC违反概率上限
        'delta': 0.05,    # 统计置信度
        'demands': demands,
        'prices': prices,
        'use_full_cov': True  # 使用完整协方差矩阵
    }
    
    return data


def main():
    """主测试函数"""
    print("="*70)
    print("测试新方案: JCC → 坐标变换 → 聚合")
    print("="*70)
    
    # 生成测试数据
    print("\n生成测试数据...")
    data = generate_test_data()
    print(f"  TCL数量: {len(data['tcls'])}")
    print(f"  时间步数: {data['periods']}")
    print(f"  形状集样本数: {len(data['D_shape'])}")
    print(f"  校准集样本数: {len(data['D_calib'])}")
    
    # 运行新方案
    print("\n运行新方案...")
    try:
        results = new_algo.algo(data)
        
        if results['status'] == 'success':
            print("\n" + "="*70)
            print("✓ 测试成功!")
            print("="*70)
            print(f"\n成本优化结果:")
            print(f"  目标值: {results.get('cost_value', 'N/A'):.2f}")
            print(f"  求解时间: {results.get('cost_time', 'N/A'):.4f}s")
            
            print(f"\n峰值优化结果:")
            print(f"  目标值: {results.get('peak_value', 'N/A'):.2f}")
            print(f"  求解时间: {results.get('peak_time', 'N/A'):.4f}s")
            
            print(f"\n总运行时间: {results['algo_time']:.2f}s")
            
            # 显示不确定性集信息
            if 'U_initial' in results:
                U_init = results['U_initial']
                print(f"\nSRO椭球参数: s*={U_init['s_star']:.4f}")
            
            if 'U_final' in results:
                U_final = results['U_final']
                print(f"Re-SRO统一参数: s_unified={U_final['s_unified']:.4f}")
            
            return True
        else:
            print(f"\n✗ 测试失败: {results.get('status', 'unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
