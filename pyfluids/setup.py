
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

fluids = Extension("fluids",
                   sources = ["fluids.pyx"],
                   library_dirs = ['../lib'],
                   libraries = ['fluids'],
                   include_dirs=["../include", np.get_include()])

setup(cmdclass = {'build_ext': build_ext},
      ext_modules = [fluids])
