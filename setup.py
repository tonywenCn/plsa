import numpy
from distutils.core import setup
from Cython.Build import cythonize

ext_modules = cythonize("cplsa.pyx", language="c++")
for e in ext_modules:
    e.extra_compile_args.extend(["-std=c++0x"])
    e.extra_link_args.extend([])
setup(ext_modules= ext_modules, include_dirs=[numpy.get_include()])
