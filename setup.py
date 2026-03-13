"""Build script for the pre_moe_cpp C++ extension.

Usage:
    python setup.py build_ext --inplace   # build C++ extension only
    BUILD_EXT=0 pip install -e . --no-build-isolation  # install Python package, skip C++
"""

import os
from setuptools import setup, find_packages

ext_modules = []
cmdclass = {}

if os.environ.get("BUILD_EXT", "1") != "0":
    try:
        from torch.utils.cpp_extension import BuildExtension, CppExtension

        def find_nccl():
            """Find NCCL include and library paths."""
            nccl_inc = None
            for d in [
                "/usr/local/cuda/include",
                "/usr/local/cuda/targets/x86_64-linux/include",
                "/usr/include",
                "/usr/local/include",
            ]:
                if os.path.isfile(os.path.join(d, "nccl.h")):
                    nccl_inc = d
                    break

            nccl_lib = None
            for d in [
                "/usr/lib/x86_64-linux-gnu",
                "/usr/local/cuda/lib64",
                "/usr/local/cuda/targets/x86_64-linux/lib",
                "/usr/lib64",
            ]:
                if os.path.isfile(os.path.join(d, "libnccl.so")) or os.path.isfile(
                    os.path.join(d, "libnccl.so.2")
                ):
                    nccl_lib = d
                    break

            return nccl_inc, nccl_lib

        def find_cuda_inc():
            """Find CUDA include path."""
            for d in [
                "/usr/local/cuda/include",
                "/usr/local/cuda/targets/x86_64-linux/include",
                "/usr/local/cuda-12.8/targets/x86_64-linux/include",
                "/usr/local/cuda-12.4/targets/x86_64-linux/include",
            ]:
                if os.path.isfile(os.path.join(d, "cuda_runtime.h")):
                    return d
            return None

        def find_cuda_lib():
            """Find CUDA library path (libcudart.so)."""
            import glob
            for d in [
                "/usr/local/cuda/lib64",
                "/usr/local/cuda/targets/x86_64-linux/lib",
                "/usr/local/cuda-12.8/lib64",
                "/usr/local/cuda-12.4/lib64",
                "/usr/lib/x86_64-linux-gnu",
            ]:
                if glob.glob(os.path.join(d, "libcudart.so*")):
                    return d
            return None

        cuda_inc = find_cuda_inc()
        cuda_lib = find_cuda_lib()
        nccl_inc, nccl_lib = find_nccl()

        include_dirs = []
        library_dirs = []
        if cuda_inc:
            include_dirs.append(cuda_inc)
        if cuda_lib:
            library_dirs.append(cuda_lib)
        if nccl_inc:
            include_dirs.append(nccl_inc)
        if nccl_lib:
            library_dirs.append(nccl_lib)

        ext_modules = [
            CppExtension(
                name="pre_moe_cpp",
                sources=["premoe/comm/comm_utils.cpp"],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=["nccl", "cudart"],
                extra_compile_args=["-std=c++17", "-O3"],
            ),
        ]
        cmdclass = {"build_ext": BuildExtension}
    except ImportError:
        print("[setup.py] torch.utils.cpp_extension not available, skipping C++ build")

setup(
    name="premoe-sglang",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
