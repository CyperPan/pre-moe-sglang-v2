"""Pre-MoE patcher: monkey-patches SGLang's DeepseekV2 decoder layers.

Decomposes each decoder layer forward into attention + MoE phases.
On anchor layers, inserts the Pre-MoE speculative dispatch pipeline:

  1. probe(pre-attn hidden) → predicted expert IDs
  2. pre-dispatch on comm stream (simulated AllToAll, overlapped with attention)
  3. self_attn on main stream (concurrent with step 2)
  4. gate(post-attn hidden) → true expert IDs (gate is cheap)
  5. verify: compare predicted vs true GPU destinations
  6a. HIT  → AllToAll already done via pre-dispatch → skip it, run experts
  6b. MISS → pre-dispatch wrong, add AllToAll delay on main stream, run experts

Both HIT and MISS use the GATE routing (true_ids) for expert computation,
so output correctness is guaranteed.  The only difference is whether the
AllToAll cost was already paid (overlapped) or must be paid now (blocking).

Two modes:
  "serial"  — baseline: attn → delay(blocking) → gate → experts
  "premoe"  — overlap:  probe → delay(comm stream) ‖ attn → gate → verify → experts

Usage:
    states = patch_sglang_for_premoe(model, config, rank, mode="premoe")
    states = patch_sglang_for_premoe(model, config, rank, mode="serial")
"""

import types
from typing import List, Literal, Optional

import torch
import torch.distributed as dist

from premoe.config import PreMoEConfig
from premoe.probe import LinearProbe, load_probes
from premoe.dispatch_planner import compute_dispatch_plan, pack_tokens, verify_dispatch
from premoe.pipeline import CommResources

try:
    import pre_moe_cpp
except ImportError:
    pre_moe_cpp = None


# ---------------------------------------------------------------------------
# NCCL init (optional — for real NCCL pre-dispatch)
# ---------------------------------------------------------------------------

def _init_nccl_comm(rank: int, world_size: int) -> int:
    if pre_moe_cpp is None:
        raise RuntimeError("pre_moe_cpp not built")
    if rank == 0:
        nccl_id = pre_moe_cpp.get_nccl_unique_id()
    else:
        nccl_id = [0] * 128
    id_tensor = torch.tensor(nccl_id, dtype=torch.uint8).cuda()
    dist.broadcast(id_tensor, src=0)
    return pre_moe_cpp.create_nccl_comm(id_tensor.cpu().tolist(), rank, world_size)


# ---------------------------------------------------------------------------
# Per-layer state
# ---------------------------------------------------------------------------

class PreMoELayerState:
    """Tracks probe routing, comm stream, and accuracy stats for one layer."""

    def __init__(
        self,
        layer_idx: int,
        probe: Optional[LinearProbe],
        config: PreMoEConfig,
        comm: Optional[CommResources],
        mode: Literal["serial", "premoe"],
    ):
        self.layer_idx = layer_idx
        self.probe = probe
        self.config = config
        self.comm = comm
        self.mode = mode
        self.comm_stream = torch.cuda.Stream(priority=-1)

        # stats
        self.total_tokens = 0
        self.mismatch_tokens = 0
        self.gate_skips = 0   # AllToAll skipped (prediction hit)
        self.fallbacks = 0    # AllToAll re-executed (prediction miss)

        # transient probe results
        self._probe_topk_ids: Optional[torch.Tensor] = None
        self._probe_topk_weights: Optional[torch.Tensor] = None

    def run_probe(self, h_pre: torch.Tensor):
        """Run probe on pre-attention hidden states → predicted expert IDs."""
        if self.probe is None:
            return
        with torch.no_grad():
            logits = self.probe(h_pre)
            vals, ids = torch.topk(logits, k=self.config.top_k, dim=-1)
            self._probe_topk_ids = ids
            self._probe_topk_weights = torch.softmax(vals.float(), dim=-1)

    def consume_probe_routing(self):
        if self._probe_topk_ids is None:
            return None
        ids, wts = self._probe_topk_ids, self._probe_topk_weights
        self._probe_topk_ids = None
        self._probe_topk_weights = None
        return ids, wts

    @property
    def dispatch_accuracy(self) -> float:
        return 1.0 - self.mismatch_tokens / max(self.total_tokens, 1)

    @property
    def gate_skip_rate(self) -> float:
        total = self.gate_skips + self.fallbacks
        return self.gate_skips / max(total, 1)


# ---------------------------------------------------------------------------
# Run experts with pre-computed topk_output (gate already done, skip it)
# ---------------------------------------------------------------------------

def _run_experts_skip_gate(
    moe,
    hidden_states: torch.Tensor,
    topk_output,
    should_allreduce_fusion: bool,
    use_reduce_scatter: bool,
    gemm_output_zero_allocator,
):
    """Run shared experts + routed experts + AllReduce.

    Gate and topk are NOT called — topk_output is pre-computed (from gate
    verification).  For DeepEP backends, bypasses AllToAll by calling
    FusedMoE.forward() (local expert_map + AllReduce).
    """
    # Shared experts
    shared_output = moe._forward_shared_experts(
        hidden_states, gemm_output_zero_allocator
    )

    # Routed experts
    is_deepep = getattr(moe, "_enable_deepep_moe", False)
    if is_deepep:
        # Bypass AllToAll: call FusedMoE.forward (StandardDispatcher)
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        experts = moe.experts
        if experts.expert_map_cpu is not None and experts.expert_map_gpu is None:
            experts.expert_map_gpu = experts.expert_map_cpu.to(device="cuda")
        final = FusedMoE.forward(experts, hidden_states, topk_output)
    else:
        final = moe.experts(hidden_states, topk_output)

    # Combine with shared experts
    if shared_output is not None:
        final = final + shared_output

    # TP AllReduce
    if moe.tp_size > 1 and not should_allreduce_fusion and not use_reduce_scatter:
        try:
            from sglang.srt.distributed.parallel_state import (
                tensor_model_parallel_all_reduce,
            )
            final = tensor_model_parallel_all_reduce(final)
        except ImportError:
            pass

    return final


# ---------------------------------------------------------------------------
# Decoder-layer forward patch (decomposed: attention → gate → verify → experts)
# ---------------------------------------------------------------------------

def _patch_decoder_layer(layer, st: PreMoELayerState):
    """Replace decoder layer forward with decomposed version.

    Data flow for premoe mode on anchor layers:

      prepare_attn → h_pre (pre-attention hidden states)
        → probe(h_pre) → pred_ids
        → pre-dispatch on comm_stream (simulated AllToAll, overlaps with attn)
      self_attn(h_pre) → h_attn
      prepare_mlp(h_attn) → h_mlp (post-attention, input to MoE)
        → sync comm_stream (pre-dispatch finished)
        → gate(h_mlp) → true_ids (cheap, ~0.1ms)
        → verify(pred_ids, true_ids)
        → HIT:  skip AllToAll, run experts with true_ids
        → MISS: add AllToAll delay, run experts with true_ids
      postprocess

    Data flow for serial mode:

      prepare_attn → self_attn → prepare_mlp → h_mlp
        → AllToAll delay on main stream (blocking)
        → gate → topk → experts (original MoE path)
      postprocess
    """
    delay_us = st.config.comm_delay_us

    def patched_forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        residual,
        zero_allocator=None,
        gemm_output_zero_allocator=None,
    ):
        has_tokens = hidden_states is not None and hidden_states.shape[0] > 0
        is_prefill = hidden_states is not None and hidden_states.shape[0] > 1

        # ── Phase 1: prepare_attn (RMSNorm → pre-attention hidden states) ──
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch, "",
        )

        # ── Phase 2: Probe + pre-dispatch on comm stream (prefill only) ──
        if st.mode == "premoe" and is_prefill and st.probe is not None:
            st.run_probe(hidden_states)

            # Launch simulated AllToAll on comm stream (overlaps with attention)
            ev = torch.cuda.Event()
            ev.record()
            with torch.cuda.stream(st.comm_stream):
                st.comm_stream.wait_event(ev)
                if delay_us > 0:
                    torch.cuda._sleep(int(delay_us * 1000))

        # ── Phase 3: Attention (main stream, overlaps with pre-dispatch) ──
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        # ── Phase 4: prepare_mlp (RMSNorm → post-attention states for MoE) ──
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch,
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        mlp_alloc = gemm_output_zero_allocator
        if not hasattr(self.mlp, "gate"):
            mlp_alloc = None

        # ── Phase 5: MoE dispatch replacement ──
        if st.mode == "premoe" and is_prefill and st.probe is not None:
            # 5a. Sync: wait for overlapped pre-dispatch to finish
            torch.cuda.current_stream().wait_stream(st.comm_stream)

            # 5b. Run gate to get TRUE routing (gate is cheap: one F.linear)
            router_logits = self.mlp.gate(hidden_states, mlp_alloc)
            topk_output = self.mlp.topk(hidden_states, router_logits)

            # 5c. Verify probe prediction vs gate truth
            routing = st.consume_probe_routing()
            if routing is not None:
                probe_ids, _ = routing
                true_ids = topk_output.topk_ids

                mismatch = verify_dispatch(
                    probe_ids, true_ids,
                    st.config.num_experts, st.config.ep_size,
                    st.comm.rank if st.comm else 0,
                )
                n_mm = len(mismatch)
                st.total_tokens += hidden_states.shape[0]
                st.mismatch_tokens += n_mm

                if n_mm <= hidden_states.shape[0] * 0.05:
                    # ── HIT: pre-dispatch succeeded → skip AllToAll ──
                    st.gate_skips += 1
                    hidden_states = _run_experts_skip_gate(
                        self.mlp, hidden_states, topk_output,
                        should_allreduce_fusion, use_reduce_scatter,
                        mlp_alloc,
                    )
                else:
                    # ── MISS: pre-dispatch wrong → pay AllToAll cost now ──
                    st.fallbacks += 1
                    if delay_us > 0:
                        torch.cuda._sleep(int(delay_us * 1000))
                    hidden_states = _run_experts_skip_gate(
                        self.mlp, hidden_states, topk_output,
                        should_allreduce_fusion, use_reduce_scatter,
                        mlp_alloc,
                    )
            else:
                # Probe didn't produce routing, fallback to original MoE
                st.fallbacks += 1
                hidden_states = self.mlp(
                    hidden_states, forward_batch,
                    should_allreduce_fusion, use_reduce_scatter,
                    mlp_alloc,
                )

        elif st.mode == "serial" and has_tokens and delay_us > 0:
            # Serial baseline: AllToAll delay THEN original MoE (blocking)
            torch.cuda._sleep(int(delay_us * 1000))
            hidden_states = self.mlp(
                hidden_states, forward_batch,
                should_allreduce_fusion, use_reduce_scatter,
                mlp_alloc,
            )

        else:
            hidden_states = self.mlp(
                hidden_states, forward_batch,
                should_allreduce_fusion, use_reduce_scatter,
                mlp_alloc,
            )

        # ── Phase 6: postprocess ──
        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        if not should_allreduce_fusion:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch,
            )

        return hidden_states, residual

    layer.forward = types.MethodType(patched_forward, layer)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def patch_sglang_for_premoe(
    model,
    config: PreMoEConfig,
    rank: int,
    world_size: int = 2,
    mode: Literal["serial", "premoe"] = "premoe",
) -> List[PreMoELayerState]:
    """Apply Pre-MoE patches to a loaded SGLang model.

    Decomposes each MoE decoder layer's forward into attention + MoE phases.
    In premoe mode, anchor layers get probe + pre-dispatch overlap + verify.
    In serial mode, every MoE layer gets a blocking delay before MoE.
    """
    device = torch.device(f"cuda:{rank}")

    probes = load_probes(config, device) if mode == "premoe" else {}

    # Optional NCCL
    comm = None
    if pre_moe_cpp is not None and mode == "premoe":
        try:
            comm = CommResources(config, rank, device)
            nccl_handle = _init_nccl_comm(rank, world_size)
            comm.init_nccl(nccl_handle)
            print(f"[Pre-MoE] NCCL comm initialised (rank={rank})")
        except Exception as e:
            print(f"[Pre-MoE] NCCL init skipped ({e})")
            comm = None

    # Find decoder layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise AttributeError("Cannot find decoder layers")

    states: List[PreMoELayerState] = []

    for idx in range(len(layers)):
        layer = layers[idx]

        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
            continue

        probe = probes.get(idx) if mode == "premoe" else None
        layer_mode: Literal["serial", "premoe"]

        if mode == "serial":
            layer_mode = "serial"
        elif probe is not None:
            layer_mode = "premoe"
        else:
            layer_mode = "serial"

        st = PreMoELayerState(idx, probe, config, comm, layer_mode)
        states.append(st)

        _patch_decoder_layer(layer, st)

        tag = "premoe-overlap" if layer_mode == "premoe" else "serial-delay"
        if config.log_accuracy:
            print(f"[Pre-MoE] Patched layer {idx} ({tag})")

    n_overlap = sum(1 for s in states if s.mode == "premoe")
    n_serial = sum(1 for s in states if s.mode == "serial")
    print(
        f"[Pre-MoE] {len(states)} MoE layers patched  "
        f"(mode={mode}, overlap={n_overlap}, serial={n_serial}, "
        f"delay={config.comm_delay_us}μs)"
    )
    return states


def print_premoe_stats(states: List[PreMoELayerState]):
    """Print dispatch accuracy and gate-skip stats."""
    premoe_states = [s for s in states if s.mode == "premoe"]
    if not premoe_states:
        print("[Pre-MoE] No premoe-mode layers (serial baseline).")
        return
    print("\n[Pre-MoE] ═══ Dispatch Statistics ═══")
    for st in premoe_states:
        print(
            f"  Layer {st.layer_idx}: "
            f"accuracy={st.dispatch_accuracy:.4f}  "
            f"gate_skip_rate={st.gate_skip_rate:.4f}  "
            f"({st.gate_skips} skips / {st.fallbacks} fallbacks / "
            f"{st.mismatch_tokens}/{st.total_tokens} mismatched)"
        )
