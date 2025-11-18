"""
Cython 编译脚本

使用方法:
    python flexitroid/cython/setup.py build_ext --inplace

这将生成 b_fast.pyd 和 p_fast.pyd (Windows) 或 .so (Linux/Mac)
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "flexitroid.cython.b_fast",
        ["flexitroid/cython/b_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['/O2'] if __import__('sys').platform == 'win32' else ['-O3'],
    ),
    Extension(
        "flexitroid.cython.p_fast",
        ["flexitroid/cython/p_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['/O2'] if __import__('sys').platform == 'win32' else ['-O3'],
    ),
]

setup(
    name="flexitroid_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
