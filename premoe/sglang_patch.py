"""Patch SGLang's DeepseekV2DecoderLayer in-place for Pre-MoE.

This module directly modifies the installed SGLang package's deepseek_v2.py
to insert Pre-MoE speculative dispatch into the decoder layer forward path.

The patches auto-configure from environment variables — no need for a
post-model-load hook.  Probes are lazy-loaded on first forward() call.

Two modes (set via PREMOE_MODE env var):

  serial:   Adds a CUDA delay AFTER attention on every MoE layer,
            simulating EP AllToAll dispatch cost (baseline).

  premoe:   On anchor layers (those with trained probes):
              - probe BEFORE attention → predicted routing
              - simulated AllToAll on comm stream (overlapped with attention)
              - gate AFTER attention → true routing (cheap)
              - verify prediction; HIT → skip AllToAll; MISS → pay delay
            On non-anchor MoE layers: same serial delay as baseline.

Usage:
    python -m premoe.sglang_patch apply    # apply source patches
    python -m premoe.sglang_patch revert   # restore original

    Then launch server with env vars:
    PREMOE_MODE=serial  PREMOE_DELAY_US=2000 python -m sglang.launch_server ...
    PREMOE_MODE=premoe  PREMOE_DELAY_US=2000 PREMOE_PROBE_DIR=probes python -m sglang.launch_server ...

Environment variables (read at runtime by patched code):
    PREMOE_MODE       — "serial" | "premoe" (default: "" = no Pre-MoE)
    PREMOE_DELAY_US   — dispatch delay in μs (default 2000)
    PREMOE_PROBE_DIR  — path to trained probe weights (default "probes")
"""

import importlib
import importlib.util
import os
import shutil
import sys
from pathlib import Path


def _get_deepseek_v2_path() -> Path:
    """Find the installed SGLang deepseek_v2.py without importing it.

    Avoids triggering the full SGLang import chain (which may fail if
    system libs like libnuma are missing).
    """
    # Method 1: use importlib to find the package without executing it
    try:
        spec = importlib.util.find_spec("sglang.srt.models.deepseek_v2")
        if spec and spec.origin:
            return Path(spec.origin)
    except (ModuleNotFoundError, ValueError):
        pass

    # Method 2: find sglang package root and construct path
    try:
        spec = importlib.util.find_spec("sglang")
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                p = Path(loc) / "srt" / "models" / "deepseek_v2.py"
                if p.exists():
                    return p
    except (ModuleNotFoundError, ValueError):
        pass

    # Method 3: search common site-packages paths
    for base in [
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path("/usr/local/lib") / f"python{sys.version_info.major}.{sys.version_info.minor}" / "dist-packages",
    ]:
        p = base / "sglang" / "srt" / "models" / "deepseek_v2.py"
        if p.exists():
            return p

    raise FileNotFoundError("Cannot find SGLang's deepseek_v2.py")


# ---------------------------------------------------------------------------
# Init patch: add Pre-MoE fields, auto-configured from env vars
# ---------------------------------------------------------------------------
INIT_PATCH = '''
        # ── Pre-MoE dispatch fields (auto-configured from env) ──
        import os as _os
        _premoe_env = _os.environ.get("PREMOE_MODE", "")
        self._premoe_mode = _premoe_env if _premoe_env in ("serial", "premoe") else None
        self._premoe_delay_us = int(_os.environ.get("PREMOE_DELAY_US", "2000"))
        self._premoe_probe_dir = _os.environ.get("PREMOE_PROBE_DIR", "probes")
        self._premoe_probe = None
        self._premoe_probe_loaded = False
        self._premoe_comm_stream = None
        self._premoe_topk = 6
        self._premoe_num_experts = 64
        self._premoe_ep_size = 2
        self._premoe_rank = 0
        self._premoe_pred_ids = None
        self._premoe_pred_weights = None
        self._premoe_stats = {"total": 0, "mismatches": 0, "skips": 0, "fallbacks": 0}
'''

# ---------------------------------------------------------------------------
# Forward patch BEFORE attention:
#   1. Lazy-load probe on first call
#   2. If premoe+probe: run probe + launch pre-dispatch on comm stream
# ---------------------------------------------------------------------------
FORWARD_PATCH_BEFORE_ATTN = '''
        # ── Pre-MoE: lazy probe load + speculative dispatch ──
        if not self._premoe_probe_loaded and self._premoe_mode is not None:
            self._premoe_probe_loaded = True
            import os as _os, torch as _torch
            _is_moe = hasattr(self, 'mlp') and hasattr(self.mlp, 'gate')
            if self._premoe_mode == "premoe" and _is_moe:
                _probe_path = _os.path.join(self._premoe_probe_dir, f"probe_layer{self.layer_id}.pt")
                if _os.path.exists(_probe_path):
                    from premoe.probe import LinearProbe
                    _state = _torch.load(_probe_path, map_location=hidden_states.device, weights_only=True)
                    _dim = _state["linear.weight"].shape[1]
                    _n_exp = _state["linear.weight"].shape[0]
                    self._premoe_probe = LinearProbe(_dim, _n_exp).to(device=hidden_states.device, dtype=hidden_states.dtype)
                    self._premoe_probe.load_state_dict({k: v.to(hidden_states.dtype) for k, v in _state.items()})
                    self._premoe_probe.eval()
                    self._premoe_num_experts = _n_exp
                    self._premoe_comm_stream = _torch.cuda.Stream(priority=-1)
                    self.mlp._premoe_routing = None
                    print(f"[Pre-MoE] Layer {self.layer_id}: premoe-overlap (probe loaded, delay={self._premoe_delay_us}us)")
                else:
                    self._premoe_mode = "serial"
                    print(f"[Pre-MoE] Layer {self.layer_id}: serial-delay (no probe, delay={self._premoe_delay_us}us)")
            elif self._premoe_mode == "serial" and _is_moe:
                print(f"[Pre-MoE] Layer {self.layer_id}: serial-delay (delay={self._premoe_delay_us}us)")
            else:
                self._premoe_mode = None
        if self._premoe_mode == "premoe" and self._premoe_probe is not None and hidden_states.shape[0] > 1:
            import torch as _torch
            with _torch.no_grad():
                _probe_logits = self._premoe_probe(hidden_states)
                _vals, _ids = _torch.topk(_probe_logits, k=self._premoe_topk, dim=-1)
                _wts = _torch.softmax(_vals.float(), dim=-1)
                self._premoe_pred_ids = _ids
                self._premoe_pred_weights = _wts
            if self._premoe_comm_stream is not None and self._premoe_delay_us > 0:
                _ev = _torch.cuda.Event()
                _ev.record()
                with _torch.cuda.stream(self._premoe_comm_stream):
                    self._premoe_comm_stream.wait_event(_ev)
                    _torch.cuda._sleep(int(self._premoe_delay_us * 1000))
'''

# ---------------------------------------------------------------------------
# Forward patch BEFORE self.mlp() call:
#   premoe → sync comm stream, pass probe routing to MoE
#   serial → blocking delay on main stream
# ---------------------------------------------------------------------------
FORWARD_PATCH_BEFORE_MLP = '''
        # ── Pre-MoE: sync pre-dispatch / serial delay ──
        if self._premoe_mode == "premoe" and self._premoe_pred_ids is not None and self._premoe_comm_stream is not None:
            import torch as _torch
            _torch.cuda.current_stream().wait_stream(self._premoe_comm_stream)
        elif self._premoe_mode in ("serial", "premoe") and self._premoe_delay_us > 0:
            import torch as _torch
            if hidden_states is not None and hidden_states.shape[0] > 0:
                _torch.cuda._sleep(int(self._premoe_delay_us * 1000))
        if self._premoe_mode == "premoe" and self._premoe_pred_ids is not None:
            if hasattr(self, 'mlp'):
                self.mlp._premoe_routing = (
                    self._premoe_pred_ids,
                    self._premoe_pred_weights,
                    self,
                )
            self._premoe_pred_ids = None
            self._premoe_pred_weights = None
'''

# ---------------------------------------------------------------------------
# MoE forward_normal patch: gate verify + conditional AllToAll skip
# ---------------------------------------------------------------------------
MOE_GATE_BYPASS = '''
        # ── Pre-MoE: gate verify + conditional AllToAll skip ──
        _premoe_r = getattr(self, '_premoe_routing', None)
        if _premoe_r is not None:
            self._premoe_routing = None
            _p_ids, _p_wts, _layer_ref = _premoe_r
            if _p_ids is not None and hidden_states.shape[0] > 0:
                import torch as _torch
                # Gate → true routing (cheap: one F.linear)
                _true_logits = self.gate(hidden_states, gemm_output_zero_allocator) if gemm_output_zero_allocator is not None else self.gate(hidden_states)
                _true_topk = self.topk(hidden_states, _true_logits)
                _true_ids = _true_topk.topk_ids if hasattr(_true_topk, 'topk_ids') else _true_topk[1]
                # Verify probe prediction vs gate truth
                _epg = _layer_ref._premoe_num_experts // _layer_ref._premoe_ep_size
                _pred_gpu = _p_ids // _epg
                _true_gpu = _true_ids // _epg
                _pred_peer = (_pred_gpu != _layer_ref._premoe_rank).any(dim=-1)
                _true_peer = (_true_gpu != _layer_ref._premoe_rank).any(dim=-1)
                _n_mm = int((_pred_peer != _true_peer).sum().item())
                _layer_ref._premoe_stats["total"] += hidden_states.shape[0]
                _layer_ref._premoe_stats["mismatches"] += _n_mm
                if _n_mm <= hidden_states.shape[0] * 0.05:
                    # HIT: pre-dispatch succeeded → skip AllToAll
                    _layer_ref._premoe_stats["skips"] += 1
                else:
                    # MISS: pre-dispatch wrong → pay AllToAll cost now
                    _layer_ref._premoe_stats["fallbacks"] += 1
                    if _layer_ref._premoe_delay_us > 0:
                        _torch.cuda._sleep(int(_layer_ref._premoe_delay_us * 1000))
                # Both paths use gate routing (true_ids) for correctness
                _sh_out = self._forward_shared_experts(hidden_states, gemm_output_zero_allocator) if hidden_states.shape[0] > 0 else None
                _final = self.experts(hidden_states, _true_topk)
                if _sh_out is not None:
                    _final = _final + _sh_out
                if self.tp_size > 1 and not should_allreduce_fusion and not use_reduce_scatter:
                    import torch.distributed as _dist
                    _dist.all_reduce(_final)
                return _final
'''


def apply_patch(dry_run: bool = False) -> str:
    """Apply Pre-MoE source patches to SGLang's deepseek_v2.py.

    The patches are mode-agnostic — behavior is controlled at runtime
    via PREMOE_MODE env var.
    """
    src = _get_deepseek_v2_path()
    backup = src.with_suffix(".py.premoe_backup")
    content = src.read_text()

    if "Pre-MoE" in content:
        print(f"[Pre-MoE] Already patched: {src}")
        return str(src)

    if not backup.exists():
        shutil.copy2(src, backup)
        print(f"[Pre-MoE] Backup: {backup}")

    # 1. Init fields after LayerCommunicator(...)
    marker = "self.layer_communicator = LayerCommunicator("
    pos = content.find(marker)
    if pos != -1:
        depth, end = 0, pos
        for i in range(pos, len(content)):
            if content[i] == '(':
                depth += 1
            elif content[i] == ')':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        nl = content.find('\n', end)
        content = content[:nl] + '\n' + INIT_PATCH + content[nl:]

    # 2. Probe + pre-dispatch BEFORE self_attn
    attn_m = "        hidden_states = self.self_attn(\n            positions=positions,"
    attn_pos = content.find(attn_m)
    if attn_pos == -1:
        attn_m = "        hidden_states = self.self_attn("
        attn_pos = content.find(attn_m)
    if attn_pos != -1:
        content = content[:attn_pos] + FORWARD_PATCH_BEFORE_ATTN + '\n' + content[attn_pos:]

    # 3. Sync / serial delay BEFORE self.mlp(...)
    mlp_call_m = "        hidden_states = self.mlp(\n"
    mlp_call_pos = content.find(mlp_call_m, attn_pos if attn_pos != -1 else 0)
    if mlp_call_pos == -1:
        mlp_call_m = "        hidden_states = self.mlp("
        mlp_call_pos = content.find(mlp_call_m, attn_pos if attn_pos != -1 else 0)
    if mlp_call_pos != -1:
        content = content[:mlp_call_pos] + FORWARD_PATCH_BEFORE_MLP + '\n' + content[mlp_call_pos:]

    # 4. Gate bypass in forward_normal
    fn_m = "    def forward_normal("
    fn_pos = content.find(fn_m)
    if fn_pos != -1:
        shape_m = "        if hidden_states.shape[0] > 0:"
        shape_pos = content.find(shape_m, fn_pos)
        if shape_pos != -1:
            content = content[:shape_pos] + MOE_GATE_BYPASS + '\n' + content[shape_pos:]

    if dry_run:
        print(f"[Pre-MoE] DRY RUN: {src}")
        return str(src)

    src.write_text(content)
    print(f"[Pre-MoE] Patched: {src}")
    return str(src)


def revert_patch():
    """Restore original deepseek_v2.py from backup."""
    src = _get_deepseek_v2_path()
    backup = src.with_suffix(".py.premoe_backup")
    if not backup.exists():
        print(f"[Pre-MoE] No backup at {backup}")
        return
    shutil.copy2(backup, src)
    print(f"[Pre-MoE] Reverted: {src}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m premoe.sglang_patch [apply|revert|dry-run]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd in ("apply", "apply-serial", "apply-premoe"):
        apply_patch()
    elif cmd == "revert":
        revert_patch()
    elif cmd == "dry-run":
        apply_patch(dry_run=True)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
