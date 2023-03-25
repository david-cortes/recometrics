import setuptools
from setuptools import setup, Extension
import numpy as np
import sys, os, subprocess, warnings, re
from Cython.Distutils import build_ext

found_omp = True
def set_omp_false():
    global found_omp
    found_omp = False

## Modify this to make the output of the compilation tests more verbose
silent_tests = not (("verbose" in sys.argv)
                    or ("-verbose" in sys.argv)
                    or ("--verbose" in sys.argv))

## Workaround for python<=3.9 on windows
try:
    EXIT_SUCCESS = os.EX_OK
except AttributeError:
    EXIT_SUCCESS = 0

class build_ext_subclass( build_ext ):
    def build_extensions(self):
        is_msvc = self.compiler.compiler_type == "msvc"
        is_clang = hasattr(self.compiler, 'compiler_cxx') and ("clang++" in self.compiler.compiler_cxx)

        compiler = self.compiler.compiler_type
        if is_msvc:
            for e in self.extensions:
                e.extra_compile_args += ['/O2', '/openmp', '/GL', '/std:c++14', '/fp:contract', '/fp:except-']
        else:
            if not self.check_for_variable_dont_set_march() and not self.check_cflags_contain_arch():
                self.add_march_native()
            else:
                if not is_windows:
                    self.add_link_time_optimization()
                for e in self.extensions:
                    e.define_macros += [("USE_BLAS", None)]
            self.add_openmp_linkage()
            self.add_restrict_qualifier()
            self.add_no_math_errno()
            self.add_no_trapping_math()
            self.add_O3()
            if not is_clang:
                self.add_std_cpp11()
            else:
                self.add_std_cpp17()

            # for e in self.extensions:
                # e.extra_compile_args += ['-O3', '-fopenmp', '-march=native', '-std=c++11']
                # e.extra_link_args += ['-fopenmp']

                # if is_clang:
                #     e.extra_compile_args += ['-std=c++17']

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
        build_ext.build_extensions(self)

    def check_cflags_contain_arch(self):
        if ("CFLAGS" in os.environ) or ("CXXFLAGS" in os.environ):
            has_cflags = "CFLAGS" in os.environ
            has_cxxflags = "CXXFLAGS" in os.environ
            arch_list = [
                "-march", "-mcpu", "-mtune", "-msse", "-msse2", "-msse3",
                "-mssse3", "-msse4", "-msse4a", "-msse4.1", "-msse4.2",
                "-mavx", "-mavx2", "-mavx512"
            ]
            for flag in arch_list:
                if has_cflags and flag in os.environ["CFLAGS"]:
                    return True
                if has_cxxflags and flag in os.environ["CXXFLAGS"]:
                    return True
        return False

    def check_for_variable_dont_set_march(self):
        return "DONT_SET_MARCH" in os.environ

    def add_march_native(self):
        is_apple = sys.platform[:3].lower() == "dar"
        args_march_native = ["-march=native", "-mcpu=native"]
        for arg_march_native in args_march_native:
            if self.test_supports_compile_arg(arg_march_native, with_c_comp=is_apple):
                for e in self.extensions:
                    e.extra_compile_args.append(arg_march_native)
                break

    def add_link_time_optimization(self):
        args_lto = ["-flto=auto", "-flto"]
        for arg_lto in args_lto:
            if self.test_supports_compile_arg(arg_lto):
                for e in self.extensions:
                    e.extra_compile_args.append(arg_lto)
                    e.extra_link_args.append(arg_lto)
                break

    def add_no_math_errno(self):
        arg_fnme = "-fno-math-errno"
        if self.test_supports_compile_arg(arg_fnme):
            for e in self.extensions:
                e.extra_compile_args.append(arg_fnme)
                e.extra_link_args.append(arg_fnme)

    def add_no_trapping_math(self):
        arg_fntm = "-fno-trapping-math"
        if self.test_supports_compile_arg(arg_fntm):
            for e in self.extensions:
                e.extra_compile_args.append(arg_fntm)
                e.extra_link_args.append(arg_fntm)

    def add_O3(self):
        arg_O3 = "-O3"
        if self.test_supports_compile_arg(arg_O3):
            for e in self.extensions:
                e.extra_compile_args.append(arg_O3)
                e.extra_link_args.append(arg_O3)

    def add_std_cpp11(self):
        arg_std_cpp11 = "-std=c++11"
        if self.test_supports_compile_arg(arg_std_cpp11):
            for e in self.extensions:
                e.extra_compile_args.append(arg_std_cpp11)
                e.extra_link_args.append(arg_std_cpp11)

    def add_std_cpp17(self):
        arg_std_cpp17 = "-std=c++17"
        if self.test_supports_compile_arg(arg_std_cpp17):
            for e in self.extensions:
                e.extra_compile_args.append(arg_std_cpp17)
                e.extra_link_args.append(arg_std_cpp17)

    def add_openmp_linkage(self):
        arg_omp1 = "-fopenmp"
        arg_omp2 = "-fopenmp=libomp"
        args_omp3 = ["-fopenmp=libomp", "-lomp"]
        arg_omp4 = "-qopenmp"
        arg_omp5 = "-xopenmp"
        is_apple = sys.platform[:3].lower() == "dar"
        args_apple_omp = ["-Xclang", "-fopenmp", "-lomp"]
        args_apple_omp2 = ["-Xclang", "-fopenmp", "-L/usr/local/lib", "-lomp", "-I/usr/local/include"]
        has_brew_omp = False
        if is_apple:
            try:
                res_brew_pref = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True)
                if res_brew_pref.returncode == EXIT_SUCCESS:
                    brew_omp_prefix = res_brew_pref.stdout.decode().strip()
                    args_apple_omp3 = ["-Xclang", "-fopenmp", f"-L{brew_omp_prefix}/lib", "-lomp", f"-I{brew_omp_prefix}/include"]
                    has_brew_omp = True
            except Exception as e:
                pass



        if self.test_supports_compile_arg(arg_omp1, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp1)
                e.extra_link_args.append(arg_omp1)
        elif is_apple and self.test_supports_compile_arg(args_apple_omp, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-lomp"]
        elif is_apple and self.test_supports_compile_arg(args_apple_omp2, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-L/usr/local/lib", "-lomp"]
                e.include_dirs += ["/usr/local/include"]
        elif is_apple and has_brew_omp and self.test_supports_compile_arg(args_apple_omp3, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += [f"-L{brew_omp_prefix}/lib", "-lomp"]
                e.include_dirs += [f"{brew_omp_prefix}/include"]
        elif self.test_supports_compile_arg(arg_omp2, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-fopenmp=libomp"]
                e.extra_link_args += ["-fopenmp"]
        elif self.test_supports_compile_arg(args_omp3, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-fopenmp=libomp"]
                e.extra_link_args += ["-fopenmp", "-lomp"]
        elif self.test_supports_compile_arg(arg_omp4, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp4)
                e.extra_link_args.append(arg_omp4)
        elif self.test_supports_compile_arg(arg_omp5, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp5)
                e.extra_link_args.append(arg_omp5)
        else:
            set_omp_false()

    # Note: in apple systems, it somehow might end up triggering the arguments with
    # the C compiler instead of the CXX compiler. What's worse, sometimes this compiler
    # thinks it's building for aarch64 even when executed in amd64.
    def test_supports_compile_arg(self, comm, with_omp=False, with_c_comp=False):
        is_supported = False
        try:
            if not hasattr(self.compiler, "compiler_cxx"):
                return False
            if not isinstance(comm, list):
                comm = [comm]
            print("--- Checking compiler support for option '%s'" % " ".join(comm))
            fname = "recometrics_compiler_testing.cpp"
            with open(fname, "w") as ftest:
                ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
            try:
                if not isinstance(self.compiler.compiler_cxx, list):
                    cmd = list(self.compiler.compiler_cxx)
                else:
                    cmd = self.compiler.compiler_cxx
            except Exception:
                cmd = self.compiler.compiler_cxx
            if with_c_comp:
                if not isinstance(self.compiler.compiler, list):
                    cmd0 = list(self.compiler.compiler)
                else:
                    cmd0 = self.compiler.compiler
            if with_omp:
                with open(fname, "w") as ftest:
                    ftest.write(u"#include <omp.h>\nint main(int argc, char**argv) {return 0;}\n")
            try:
                val = subprocess.run(cmd + comm + [fname], capture_output=silent_tests).returncode
                is_supported = (val == EXIT_SUCCESS)
                if is_supported and with_c_comp:
                    val = subprocess.run(cmd0 + comm + [fname], capture_output=silent_tests).returncode
                    is_supported = (val == EXIT_SUCCESS)
            except Exception:
                is_supported = False
        except Exception:
            pass
        try:
            os.remove(fname)
        except Exception:
            pass
        return is_supported

    def add_restrict_qualifier(self):
        supports_restrict = False
        try:
            if not hasattr(self.compiler, "compiler_cxx"):
                return None
            print("--- Checking compiler support for '__restrict' qualifier")
            fname = "recometrics_compiler_testing.cpp"
            with open(fname, "w") as ftest:
                ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
            try:
                if not isinstance(self.compiler.compiler_cxx, list):
                    cmd = list(self.compiler.compiler_cxx)
                else:
                    cmd = self.compiler.compiler_cxx
            except Exception:
                cmd = self.compiler.compiler_cxx
            try:
                with open(fname, "w") as ftest:
                    ftest.write(u"int main(int argc, char**argv) {double *__restrict x = 0; return 0;}\n")
                val = subprocess.run(cmd + comm + [fname], capture_output=silent_tests).returncode
                is_supported = (val == EXIT_SUCCESS)
            except Exception:
                return None
        except Exception:
            pass
        try:
            os.remove(fname)
        except Exception:
            pass
        
        if supports_restrict:
            for e in self.extensions:
                e.define_macros += [("SUPPORTS_RESTRICT", "1")]



setup(
    name  = "recometrics",
    packages = ["recometrics"],
    version = '0.1.6-6',
    cmdclass = {'build_ext': build_ext_subclass},
    author = 'David Cortes',
    url = 'https://github.com/david-cortes/recometrics',
    install_requires = ['numpy', 'scipy', 'cython', 'pandas'],
    description = 'Library-agnostic evaluation framework for implicit-feedback recommender systems',
    ext_modules = [
        Extension("recometrics.cpp_funs",
            language="c++",
            sources=["recometrics/wrapper.pyx", "src/recometrics_instantiated.cpp"],
            include_dirs=[np.get_include(), "src"],
            define_macros=[
                ("_FOR_PYTHON", None),
                ("NDEBUG", None),
            ]
            )
        ]
)

if not found_omp:
    omp_msg  = "\n\n\nCould not detect OpenMP. Package will be built without multi-threading capabilities. "
    omp_msg += " To enable multi-threading, first install OpenMP"
    if (sys.platform[:3] == "dar"):
        omp_msg += " - for macOS: 'brew install libomp'\n"
    else:
        omp_msg += " modules for your compiler. "
    
    omp_msg += "Then reinstall this package from scratch: 'pip install --upgrade --no-deps --force-reinstall recometrics'.\n"
    warnings.warn(omp_msg)
