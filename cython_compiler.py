from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("perceptual_computer.encoder.cy_regression_approach", ["./perceptual_computer/encoder/cy_ext/main.pyx"],
              include_dirs=[numpy.get_include()]),
]
setup(
    ext_modules=cythonize(extensions),
)
