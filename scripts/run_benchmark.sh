#!/bin/bash
# run_benchmark.sh — Full E2E benchmark pipeline on RunPod
# Extracts traces → trains probes → runs benchmark across all delays
set -eo pipefail

cd "$(dirname "$0")/.."

TORCH_LIB=$(python -c "import torch; print(torch.__path__[0])")/lib
export LD_LIBRARY_PATH=$TORCH_LIB:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HOME=${HF_HOME:-/workspace/huggingface_cache}

mkdir -p results traces probes

echo "============================================================"
echo "  Pre-MoE × SGLang PoC — E2E Benchmark Pipeline"
echo "============================================================"

MODE=${1:-all}

# ──────────────────────────────────────────────
# Step 1: Extract traces from DeepSeek-V2-Lite
# ──────────────────────────────────────────────
if [ "$MODE" = "all" ] || [ "$MODE" = "extract" ]; then
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  Step 1/3: Extract Hidden State Traces   ║"
    echo "╚══════════════════════════════════════════╝"

    if [ -f "traces/metadata.json" ]; then
        echo "  Traces already exist. Skipping. (delete traces/ to re-extract)"
    else
        python scripts/extract_traces.py \
            --model deepseek-ai/DeepSeek-V2-Lite-Chat \
            --num-prompts 15 \
            --max-len 4096 \
            --save-dir traces \
            2>&1 | tee results/step1_extract.log
    fi
fi

# ──────────────────────────────────────────────
# Step 2: Train linear probes
# ──────────────────────────────────────────────
if [ "$MODE" = "all" ] || [ "$MODE" = "train" ]; then
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  Step 2/3: Train Linear Probes           ║"
    echo "╚══════════════════════════════════════════╝"

    if [ -f "probes/probe_summary.json" ]; then
        echo "  Probes already trained. Skipping. (delete probes/ to retrain)"
    else
        python scripts/train_probes.py \
            --traces-dir traces \
            --probes-dir probes \
            --topk 6 \
            --ep-size 2 \
            2>&1 | tee results/step2_train.log
    fi
fi

# ──────────────────────────────────────────────
# Step 3: Run E2E benchmark (sweep all delays)
# ──────────────────────────────────────────────
if [ "$MODE" = "all" ] || [ "$MODE" = "bench" ]; then
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  Step 3/3: E2E Benchmark (all delays)    ║"
    echo "╚══════════════════════════════════════════╝"

    for DELAY in 0 1000 2000 5000; do
        echo ""
        echo "  ── comm-delay = ${DELAY}μs ──"
        torchrun --nproc_per_node=2 benchmarks/bench_premoe_sglang.py \
            --traces-dir traces \
            --probe-dir probes \
            --seq-lens 1024 2048 4096 8192 \
            --iters 200 \
            --warmup 30 \
            --comm-delay-us $DELAY \
            2>&1 | tee results/step3_bench_delay${DELAY}us.log
    done
fi

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Results:"
echo "    results/premoe_sglang_bench.json           — delay=0 (NVLink)"
echo "    results/premoe_sglang_bench_delay1000us.json — delay=1ms"
echo "    results/premoe_sglang_bench_delay2000us.json — delay=2ms"
echo "    results/premoe_sglang_bench_delay5000us.json — delay=5ms"
echo "    results/step*_*.log                         — Per-step logs"
echo "============================================================"
