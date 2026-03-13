#!/bin/bash
# run_ttft_benchmark.sh — E2E TTFT & Throughput benchmark on RunPod
#
# Compares Serial EP (blocking AllToAll) vs Pre-MoE (overlapped dispatch).
#
# Serial:  attn → AllToAll delay(blocking) → gate → experts
# Pre-MoE: probe → AllToAll(comm stream) ‖ attn → gate → verify → experts
#
# When probe prediction is correct (~95%+), the AllToAll delay is hidden
# behind attention → measurable TTFT and throughput improvement.
#
# Prerequisites:
#   - bash scripts/setup_runpod.sh      (build C++ extension, install deps)
#   - bash scripts/run_benchmark.sh extract train   (get probes)
#   - pip install "sglang[all]"
set -eo pipefail

cd "$(dirname "$0")/.."

# Fix nvidia lib version mismatch (must happen before any torch import)
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
NVJITLINK_LIB="${SITE_PKGS}/nvidia/nvjitlink/lib/libnvJitLink.so.12"
[ -f "$NVJITLINK_LIB" ] && export LD_PRELOAD="$NVJITLINK_LIB"
TORCH_LIB="${SITE_PKGS}/torch/lib"
export LD_LIBRARY_PATH=$TORCH_LIB:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HOME=${HF_HOME:-/workspace/huggingface_cache}
export PYTHONPATH="$(pwd):$PYTHONPATH"

MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat"
VANILLA_PORT=30000
PREMOE_PORT=30001
NUM_PROMPTS=${1:-50}
MAX_TOKENS=${2:-128}
DELAY_US=${3:-2000}                 # simulated EP dispatch delay (μs)
export PREMOE_DELAY_US=$DELAY_US
export PREMOE_PROBE_DIR="${PREMOE_PROBE_DIR:-$(pwd)/probes}"

WAIT_TIMEOUT=300  # seconds to wait for server startup

mkdir -p results

# Cleanup trap: kill servers + revert patches on exit/error
cleanup() {
    echo "  Cleaning up..."
    kill $VANILLA_PID 2>/dev/null || true
    kill $PREMOE_PID 2>/dev/null || true
    wait $VANILLA_PID 2>/dev/null || true
    wait $PREMOE_PID 2>/dev/null || true
    PYTHONPATH=. python -m premoe.sglang_patch revert 2>/dev/null || true
}
VANILLA_PID=""
PREMOE_PID=""
trap cleanup EXIT

echo "============================================================"
echo "  Pre-MoE × SGLang — TTFT & Throughput Benchmark"
echo "============================================================"
echo "  Model:       $MODEL"
echo "  Prompts:     $NUM_PROMPTS"
echo "  Max tokens:  $MAX_TOKENS"
echo "  Dispatch delay:  ${DELAY_US}μs per MoE layer"
echo ""

# Check probes exist
if [ ! -d "$PREMOE_PROBE_DIR" ] || [ -z "$(ls "$PREMOE_PROBE_DIR"/probe_layer*.pt 2>/dev/null)" ]; then
    echo "ERROR: No probe files found in $PREMOE_PROBE_DIR"
    echo "  Run: bash scripts/run_benchmark.sh extract train"
    exit 1
fi

# ── Apply source patches (one-time, mode-agnostic) ──
PYTHONPATH=. python -m premoe.sglang_patch revert 2>/dev/null || true
PYTHONPATH=. python -m premoe.sglang_patch apply

# ──────────────────────────────────────────────
# Helper: wait for server health check
# ──────────────────────────────────────────────
wait_for_server() {
    local port=$1
    local pid=$2
    local name=$3
    for i in $(seq 1 $WAIT_TIMEOUT); do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "  $name server ready! (${i}s)"
            return 0
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "  ERROR: $name server process died."
            exit 1
        fi
        sleep 1
    done
    echo "  ERROR: $name server not ready after ${WAIT_TIMEOUT}s"
    exit 1
}

# ──────────────────────────────────────────────
# Step 1: Start SERIAL baseline SGLang server
#   PREMOE_MODE=serial → every MoE layer gets blocking delay after attention
# ──────────────────────────────────────────────
echo "╔══════════════════════════════════════════════╗"
echo "║  Step 1/4: Start Serial-Delay Baseline       ║"
echo "╚══════════════════════════════════════════════╝"

PREMOE_MODE=serial python -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp 2 \
    --port $VANILLA_PORT \
    --host 0.0.0.0 \
    --disable-cuda-graph \
    &
VANILLA_PID=$!
echo "  Baseline server PID: $VANILLA_PID"

echo "  Waiting for baseline server..."
wait_for_server $VANILLA_PORT $VANILLA_PID "Baseline"

# ──────────────────────────────────────────────
# Step 2: Benchmark serial baseline
# ──────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Step 2/4: Benchmark Serial Baseline         ║"
echo "╚══════════════════════════════════════════════╝"

python -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port $VANILLA_PORT \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len $MAX_TOKENS \
    --num-prompts $NUM_PROMPTS \
    --output-file results/serial_bench_serving.jsonl \
    2>&1 | tee results/serial_ttft.log

echo "  Stopping baseline server..."
kill $VANILLA_PID 2>/dev/null || true
wait $VANILLA_PID 2>/dev/null || true
VANILLA_PID=""
sleep 3

# ──────────────────────────────────────────────
# Step 3: Start Pre-MoE SGLang server
#   PREMOE_MODE=premoe → anchor layers overlap, others serial
# ──────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Step 3/4: Start Pre-MoE Overlapped Server   ║"
echo "╚══════════════════════════════════════════════╝"

PREMOE_MODE=premoe python -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp 2 \
    --port $PREMOE_PORT \
    --host 0.0.0.0 \
    --disable-cuda-graph \
    &
PREMOE_PID=$!
echo "  Pre-MoE server PID: $PREMOE_PID"

echo "  Waiting for Pre-MoE server..."
wait_for_server $PREMOE_PORT $PREMOE_PID "Pre-MoE"

# ──────────────────────────────────────────────
# Step 4: Benchmark Pre-MoE
# ──────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Step 4/4: Benchmark Pre-MoE Overlapped      ║"
echo "╚══════════════════════════════════════════════╝"

python -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port $PREMOE_PORT \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len $MAX_TOKENS \
    --num-prompts $NUM_PROMPTS \
    --output-file results/premoe_bench_serving.jsonl \
    2>&1 | tee results/premoe_ttft.log

echo "  Stopping Pre-MoE server..."
kill $PREMOE_PID 2>/dev/null || true
wait $PREMOE_PID 2>/dev/null || true
PREMOE_PID=""

# ──────────────────────────────────────────────
# Compare
# ──────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  COMPARISON  (dispatch delay = ${DELAY_US}μs per MoE layer)"
echo "============================================================"

python -c "
from pathlib import Path
import re

def parse_log(path):
    m = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        for pat, key in [
            (r'Mean TTFT.*?([\d.]+)', 'ttft_mean'),
            (r'Median TTFT.*?([\d.]+)', 'ttft_p50'),
            (r'P99 TTFT.*?([\d.]+)', 'ttft_p99'),
            (r'Output token throughput.*?([\d.]+)', 'throughput'),
            (r'Request throughput.*?([\d.]+)', 'req_tps'),
        ]:
            found = re.search(pat, line)
            if found:
                m[key] = float(found.group(1))
    return m

s = parse_log('results/serial_ttft.log')
p = parse_log('results/premoe_ttft.log')

if not s or not p:
    print('  Could not parse results.')
    exit()

print(f\"  {'Metric':<32} {'Serial':>10} {'Pre-MoE':>10} {'Delta':>10}\")
print(f\"  {'-'*63}\")
for k, name, unit in [
    ('ttft_mean', 'TTFT mean (ms)', 'ms'),
    ('ttft_p50',  'TTFT median (ms)', 'ms'),
    ('ttft_p99',  'TTFT p99 (ms)', 'ms'),
    ('throughput','Output throughput (tok/s)', ''),
    ('req_tps',   'Request throughput (req/s)', ''),
]:
    sv, pv = s.get(k, 0), p.get(k, 0)
    d = pv - sv
    sign = '+' if d >= 0 else ''
    print(f\"  {name:<32} {sv:>10.1f} {pv:>10.1f} {sign}{d:>8.1f}{unit}\")

if s.get('ttft_p50', 0) > 0 and p.get('ttft_p50', 0) > 0:
    print(f\"\n  TTFT speedup (median): {s['ttft_p50']/p['ttft_p50']:.3f}x\")
if s.get('throughput', 0) > 0 and p.get('throughput', 0) > 0:
    print(f\"  Throughput ratio:     {p['throughput']/s['throughput']:.3f}x\")
"

echo ""
echo "============================================================"
echo "  Done!  Results in results/serial_ttft.log"
echo "                    results/premoe_ttft.log"
echo "============================================================"
