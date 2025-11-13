# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("p_fast", ["p_fast.pyx"], include_dirs=[np.get_include()]),
    Extension("b_fast", ["b_fast.pyx"], include_dirs=[np.get_include()]),
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
)
