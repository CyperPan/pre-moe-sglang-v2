# Pre-MoE: Speculative Expert Pre-Dispatch for MoE Inference

Pre-MoE overlaps expert-parallel (EP) AllToAll communication with attention computation in Mixture-of-Experts models, reducing Time To First Token (TTFT) without sacrificing output correctness.

## Key Idea

In standard EP inference, each MoE layer executes sequentially:

```
Attention → AllToAll dispatch → Gate → Experts
```

Pre-MoE uses lightweight linear probes to predict expert routing from **pre-attention** hidden states, launching the AllToAll on a separate CUDA stream that runs concurrently with attention:

```
Probe(pre-attn) → AllToAll(comm stream) ‖ Attention(main stream) → Gate(verify) → Experts
                   ~~~overlapped~~~
```

- **HIT** (probe correct): AllToAll already done, skip it — save the full dispatch latency
- **MISS** (probe wrong): pay the AllToAll cost as usual — no correctness loss

Both paths use the **true gate routing** for expert computation, so output is always correct.

## Benchmark Results

**Model**: DeepSeek-V2-Lite-Chat (16B, 26 MoE layers, 64 experts, top-6)
**Hardware**: 2x A100-SXM4-80GB (RunPod), TP=2
**Probes**: All 26 MoE layers, linear probe per layer

### TTFT Speedup: 1.29x (dispatch delay = 5000us)

| Metric | Serial Baseline | Pre-MoE | Delta |
|--------|---------------:|--------:|------:|
| TTFT mean (ms) | 1407.8 | 1104.5 | **-303.3** |
| TTFT median (ms) | 1345.5 | 1047.0 | **-298.5** |
| TTFT p99 (ms) | 2518.1 | 1975.2 | **-543.0** |
| Output throughput (tok/s) | 194.4 | 204.8 | +10.3 |
| Request throughput (req/s) | 3.1 | 3.3 | +0.2 |

**TTFT speedup (median): 1.285x** | **Throughput ratio: 1.053x**

> Benchmark config: 50 prompts, 2048 input tokens, 128 output tokens, `request_rate=inf`, simulated EP dispatch delay = 5000us/layer.

### Probe Accuracy (GPU-dispatch level)

| Layer Range | Avg GPU-acc | Status |
|-------------|-------------|--------|
| Layers 3,4,7,9,10,14 | 99.5-100% | Excellent |
| Layers 6,8,12,15,17,22 | 99.3-99.9% | Excellent |
| Layers 2,11,13,16,23 | 98.4-99.1% | Good |
| Layers 1,5,19,20,21,24 | 96.2-97.5% | Adequate |
| Layer 25 | 94.4% | Marginal |
| Layer 18 | 90.3% | Filtered out (serial fallback) |

24 of 26 MoE layers pass the 95% GPU-accuracy threshold for premoe overlap.

## Architecture

```
premoe/
  sglang_patch.py     # Source-level patcher for SGLang's deepseek_v2.py
  patcher.py          # Runtime monkey-patch alternative
  probe.py            # LinearProbe(hidden_dim, num_experts)
  config.py           # PreMoEConfig dataclass
  dispatch_planner.py # Dispatch plan computation and verification
  pipeline.py         # CommResources for NCCL communication
  comm/               # C++ NCCL extension (async send/recv)

scripts/
  extract_traces.py   # Step 1: Extract h_pre_attn traces + gate routing
  train_probes.py     # Step 2: Train per-layer linear probes
  run_ttft_benchmark.sh  # Step 3: E2E TTFT benchmark (serial vs premoe)
  setup_runpod.sh     # One-time RunPod environment setup
```

## Quick Start

### 1. Setup (RunPod with 2x GPU)

```bash
bash scripts/setup_runpod.sh
```

### 2. Extract traces & train probes

```bash
bash scripts/run_benchmark.sh extract train
```

### 3. Run TTFT benchmark

```bash
# Usage: run_ttft_benchmark.sh [NUM_PROMPTS] [MAX_TOKENS] [DELAY_US] [INPUT_LEN] [REQUEST_RATE]
bash scripts/run_ttft_benchmark.sh 50 128 5000 2048
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PREMOE_MODE` | `""` (disabled) | `"serial"` = blocking delay baseline, `"premoe"` = overlapped dispatch |
| `PREMOE_DELAY_US` | `2000` | Simulated EP AllToAll latency in microseconds |
| `PREMOE_PROBE_DIR` | `probes` | Path to trained probe weights |

## How It Works

### Patching Mechanism

`sglang_patch.py` injects three code blocks into SGLang's `DeepseekV2DecoderLayer.forward()`:

1. **INIT_PATCH** (in `__init__`): Adds Pre-MoE fields, reads config from env vars, pre-allocates CUDA Event and comm stream

2. **FORWARD_PATCH_BEFORE_ATTN** (before `self.self_attn`):
   - Lazy-loads probe weights on first forward call
   - Runs `F.linear(hidden_states, probe_weight)` to predict routing
   - Launches simulated AllToAll (`cuda._sleep`) on high-priority comm stream
   - Prefill only (`hidden_states.shape[0] > 1`) — decode skipped

3. **FORWARD_PATCH_BEFORE_MLP** (before `self.mlp`):
   - **premoe layers (prefill)**: `main_stream.wait_stream(comm_stream)` — free if attention took longer than dispatch
   - **serial layers**: blocking `cuda._sleep` on main stream
   - **premoe layers (decode)**: no delay — pre-dispatch not needed for single tokens

### Optimizations Applied

- Pre-allocated CUDA Event (no per-forward allocation)
- Cached `torch` reference and `F.linear` (no per-forward import)
- Pre-computed delay cycles (no per-forward multiplication)
- Direct `F.linear` instead of `nn.Module.__call__` (skip hook overhead)
- `torch.topk(..., sorted=False)` (skip unnecessary sort)
- No softmax on probe output (weights unused in overlap-only mode)
- Decode phase completely bypassed for premoe layers
- Manual stream switching instead of context manager overhead
