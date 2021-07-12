import setuptools
from setuptools import setup, Extension
import numpy as np
import sys, os
from Cython.Distutils import build_ext

class build_ext_subclass( build_ext ):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        if compiler == 'msvc': # visual studio
            for e in self.extensions:
                e.extra_compile_args += ['/O2', '/openmp', '/std:c++14']
        else: # everything else that cares about following standards
            for e in self.extensions:
                e.extra_compile_args += ['-O3', '-fopenmp', '-march=native', '-std=c++11']
                e.extra_link_args += ['-fopenmp']

                # e.extra_compile_args += ['-O3', '-fopenmp', '-march=native', '-std=c++17']
                # e.extra_link_args += ['-fopenmp=libiomp5']

                # e.extra_compile_args += ['-fsanitize=address', '-static-libasan', '-ggdb', '-std=c++14']
                # e.extra_link_args += ['-fsanitize=address', '-static-libasan']

                # e.extra_compile_args += ['-O0', '-std=c++11', '-ggdb']
                # e.define_macros += [("DEBUG", None), ("_GLIBCXX_ASSERTIONS", None)]

            ## Note: apple will by default alias 'gcc' to 'clang', and will ship its own "special"
            ## 'clang' which has no OMP support and nowadays will purposefully fail to compile when passed
            ## '-fopenmp' flags. If you are using mac, and have an OMP-capable compiler,
            ## comment out the code below, or set 'use_omp' to 'True'.
            if not use_omp:
                for e in self.extensions:
                    e.extra_compile_args = [arg for arg in e.extra_compile_args if arg != '-fopenmp']
                    e.extra_link_args    = [arg for arg in e.extra_link_args    if arg != '-fopenmp']
        build_ext.build_extensions(self)

use_omp = (("enable-omp" in sys.argv)
           or ("-enable-omp" in sys.argv)
           or ("--enable-omp" in sys.argv))
if use_omp:
    sys.argv = [a for a in sys.argv if a not in ("enable-omp", "-enable-omp", "--enable-omp")]
if os.environ.get('ENABLE_OMP') is not None:
    use_omp = True
if sys.platform[:3] != "dar":
    use_omp = True

### Shorthand for apple computer:
### uncomment line below
# use_omp = True
setup(
    name  = "recometrics",
    packages = ["recometrics"],
    version = '0.1.1',
    cmdclass = {'build_ext': build_ext_subclass},
    author = 'David Cortes',
    author_email = 'david.cortes.rivera@gmail.com',
    url = 'https://github.com/david-cortes/recometrics',
    install_requires = ['numpy', 'scipy', 'cython', 'pandas'],
    description = 'Library-agnostic evaluation framework for implicit-feedback recommender systems',
    ext_modules = [
        Extension("recometrics.cpp_funs",
            language="c++",
            sources=["recometrics/wrapper.pyx", "src/recometrics_instantiated.cpp"],
            include_dirs=[np.get_include(), "src"],
            define_macros=[("_FOR_PYTHON", None)]
            )
        ]
)
