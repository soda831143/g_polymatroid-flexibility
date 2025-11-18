"""
一键编译 Cython 加速模块

运行此脚本自动编译 b_fast 和 p_fast 模块
"""
import subprocess
import sys
import os

def main():
    print("="*70)
    print("Cython 加速模块编译器")
    print("="*70)
    
    # 检查 Cython 是否安装
    try:
        import Cython
        print(f"✓ Cython 已安装 (版本 {Cython.__version__})")
    except ImportError:
        print("✗ Cython 未安装")
        print("\n正在安装 Cython...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cython"])
        print("✓ Cython 安装完成")
    
    # 检查 numpy
    try:
        import numpy as np
        print(f"✓ NumPy 已安装 (版本 {np.__version__})")
    except ImportError:
        print("✗ NumPy 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    
    # 切换到正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("\n" + "="*70)
    print("开始编译 Cython 模块...")
    print("="*70)
    
    # 编译命令
    setup_path = os.path.join("flexitroid", "cython", "setup.py")
    
    try:
        result = subprocess.run(
            [sys.executable, setup_path, "build_ext", "--inplace"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        print("\n" + "="*70)
        print("✓ 编译成功！")
        print("="*70)
        
        # 验证生成的文件
        cython_dir = os.path.join("flexitroid", "cython")
        pyd_files = [f for f in os.listdir(cython_dir) if f.endswith(('.pyd', '.so'))]
        
        if pyd_files:
            print("\n生成的文件：")
            for f in pyd_files:
                print(f"  ✓ {f}")
        
        # 测试导入
        print("\n测试导入...")
        try:
            from flexitroid.cython.b_fast import b_fast
            from flexitroid.cython.p_fast import p_fast
            print("✓ 模块导入成功！")
            print("\n现在可以运行您的算法，将自动使用 Cython 加速版本。")
        except ImportError as e:
            print(f"✗ 导入失败: {e}")
            print("请检查编译是否成功生成 .pyd 或 .so 文件")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*70)
        print("✗ 编译失败")
        print("="*70)
        print(e.stderr)
        print("\n常见问题：")
        print("1. Windows: 需要安装 Microsoft C++ Build Tools")
        print("   下载: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print("2. 确保已安装 numpy: pip install numpy")
        print("\n详细指南请查看: CYTHON_GUIDE.md")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
