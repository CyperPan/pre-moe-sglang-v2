#!/bin/bash
# setup_runpod.sh — One-time setup on RunPod (2x GPU pod)
set -eo pipefail

cd "$(dirname "$0")/.."

echo "============================================"
echo "  Pre-MoE × SGLang PoC — RunPod Setup"
echo "============================================"

# 1. GPU check
echo "[1/5] GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "ERROR: Need at least 2 GPUs for TP=2. Found $GPU_COUNT."
    exit 1
fi

# 2. HF cache (avoid filling root disk)
export HF_HOME=${HF_HOME:-/workspace/huggingface_cache}
mkdir -p "$HF_HOME"
echo "[2/5] HF_HOME=$HF_HOME"

# 3. Install SGLang (it pins its own torch + transformers versions)
echo "[3/5] Installing SGLang + dependencies..."
apt-get install -y -qq libnuma-dev 2>/dev/null || true
pip install "sglang[all]" -q
pip install numpy matplotlib ninja accelerate -q

# 4. Fix nvidia lib conflicts + build C++ extension
#    pip installs nvidia CUDA 12.8 libs, but system has CUDA 12.4.
#    PyTorch 2.9 needs 12.8 symbols. Solution: put ALL pip nvidia libs
#    before system CUDA in LD_LIBRARY_PATH.
echo "[4/5] Fixing nvidia lib paths & building C++ extension..."
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
TORCH_LIB="${SITE_PKGS}/torch/lib"

# Collect ALL pip nvidia lib dirs (cuda_runtime, nvjitlink, cusparse, etc.)
NVIDIA_LIBS=$(python -c "
import site, os, glob
sp = site.getsitepackages()[0]
dirs = sorted(glob.glob(os.path.join(sp, 'nvidia', '*', 'lib')))
print(':'.join(d for d in dirs if os.path.isdir(d)))
")

export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${TORCH_LIB}:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
unset LD_PRELOAD 2>/dev/null || true
echo "  nvidia libs: $(echo $NVIDIA_LIBS | tr ':' '\n' | wc -l) dirs"

# Verify torch loads
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# Build C++ extension
rm -rf build/ *.egg-info pre_moe_cpp*.so
python setup.py build_ext --inplace
BUILD_EXT=0 pip install -e . --no-build-isolation

# Write persistent env to bashrc (clean: remove old entries first)
sed -i '/PREMOE_ENV/d' ~/.bashrc 2>/dev/null || true
sed -i '/HF_HOME/d' ~/.bashrc 2>/dev/null || true
cat >> ~/.bashrc << BASHEOF
# PREMOE_ENV start
export HF_HOME=/workspace/huggingface_cache
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${TORCH_LIB}:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
# PREMOE_ENV end
BASHEOF

# 5. Verify everything
echo "[5/5] Verifying..."
python -c "
import pre_moe_cpp
funcs = [x for x in dir(pre_moe_cpp) if not x.startswith('_')]
print(f'  C++ OK: {funcs}')
assert len(funcs) >= 5, f'Expected >=5 functions, got {len(funcs)}: {funcs}'
"

python -c "
import torch, sglang
print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs {torch.cuda.device_count()}')
print(f'  SGLang {sglang.__version__}')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Next steps:"
echo "    1. bash scripts/run_benchmark.sh extract train"
echo "    2. bash scripts/run_ttft_benchmark.sh"
echo "============================================"
