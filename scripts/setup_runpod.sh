#!/bin/bash
# setup_runpod.sh — One-time setup on RunPod (2x GPU pod)
set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "  Pre-MoE × SGLang PoC — RunPod Setup"
echo "============================================"

# 1. GPU check
echo "[1/6] GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "ERROR: Need at least 2 GPUs for TP=2. Found $GPU_COUNT."
    exit 1
fi

# 2. HF cache (avoid filling root disk)
export HF_HOME=${HF_HOME:-/workspace/huggingface_cache}
mkdir -p "$HF_HOME"
echo 'export HF_HOME=/workspace/huggingface_cache' >> ~/.bashrc 2>/dev/null || true
echo "[2/6] HF_HOME=$HF_HOME"

# 3. Install dependencies
echo "[3/6] Installing base dependencies..."
pip install torch numpy matplotlib ninja -q 2>&1 | tail -1
pip install transformers==4.44.2 accelerate -q 2>&1 | tail -1

# 4. Install SGLang
echo "[4/6] Installing SGLang..."
pip install "sglang[all]" -q 2>&1 | tail -3

# 5. LD_LIBRARY_PATH
TORCH_LIB=$(python -c "import torch; print(torch.__path__[0])")/lib
export LD_LIBRARY_PATH=$TORCH_LIB:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=$TORCH_LIB:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc 2>/dev/null || true
echo "[5/6] LD_LIBRARY_PATH set"

# 6. Build C++ extension + install premoe package
echo "[6/6] Building C++ extension & installing premoe..."
rm -rf build/ *.egg-info pre_moe_cpp*.so
python setup.py build_ext --inplace 2>&1
BUILD_EXT=0 pip install -e . --no-build-isolation 2>&1 | tail -3

# Verify C++ extension
python -c "
import pre_moe_cpp
funcs = [x for x in dir(pre_moe_cpp) if not x.startswith('_')]
print(f'  C++ OK: {funcs}')
assert len(funcs) == 7, f'Expected 7 functions, got {len(funcs)}'
"

# Version info
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
