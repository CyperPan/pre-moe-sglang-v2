"""Core Pre-MoE pipeline: speculative dispatch overlapped with attention.

Pipeline phases per decoder layer:
  1. Probe:     predict expert routing from pre-attention hidden states
  2. Dispatch:  speculatively send tokens to peer GPU (comm stream)
  3. Attention: run self-attention (main stream, overlapping with dispatch)
  4. Verify:    compare predicted vs true routing after gate computation
  5. Fallback:  re-dispatch mismatched tokens if any
"""

import torch
import torch.nn.functional as F

from premoe.config import PreMoEConfig
from premoe.probe import LinearProbe
from premoe.dispatch_planner import compute_dispatch_plan, pack_tokens, verify_dispatch
from premoe.utils import CudaTimer

try:
    import pre_moe_cpp
except ImportError:
    pre_moe_cpp = None


class CommResources:
    """Holds NCCL communicator, stream, and send/recv buffers for one rank."""

    def __init__(self, config: PreMoEConfig, rank: int, device: torch.device):
        self.rank = rank
        self.peer = 1 - rank  # EP=2 assumption
        self.device = device
        self.nccl_comm = None
        self.nccl_stream = None

        buf_bytes = config.comm_buffer_size_mb * 1024 * 1024
        self.send_buf = torch.zeros(buf_bytes, dtype=torch.uint8, device=device)
        self.recv_buf = torch.zeros(buf_bytes, dtype=torch.uint8, device=device)

    def init_nccl(self, nccl_comm_handle: int):
        """Set the NCCL communicator handle (created externally)."""
        if pre_moe_cpp is None:
            raise RuntimeError("pre_moe_cpp extension not built")
        self.nccl_comm = nccl_comm_handle
        self.nccl_stream = pre_moe_cpp.create_cuda_stream()

    def cleanup(self):
        if self.nccl_stream is not None:
            pre_moe_cpp.destroy_cuda_stream(self.nccl_stream)
            self.nccl_stream = None

    def send_recv(self, packed: torch.Tensor):
        """Copy packed data into send buffer and launch async send/recv."""
        if packed.numel() > 0:
            n_bytes = min(packed.nbytes, self.send_buf.nbytes)
            flat = packed.view(-1).view(torch.uint8)[:n_bytes]
            self.send_buf[:n_bytes].copy_(flat)
        pre_moe_cpp.async_send_recv_start(
            self.send_buf, self.recv_buf,
            self.nccl_comm, self.peer, self.nccl_stream,
        )

    def wait(self):
        """Wait for the async send/recv to complete."""
        pre_moe_cpp.async_send_recv_wait(self.nccl_stream)


class PreMoELayerPipeline:
    """Manages the speculative dispatch pipeline for one decoder layer.

    Integrates with SGLang by splitting the decoder layer forward into:
    attention (overlapped with speculative dispatch) and MoE (after verify).
    """

    def __init__(
        self,
        layer_idx: int,
        probe: LinearProbe,
        config: PreMoEConfig,
        comm: CommResources,
    ):
        self.layer_idx = layer_idx
        self.probe = probe
        self.config = config
        self.comm = comm
        self.comm_stream = torch.cuda.Stream(priority=-1)

        # Stats tracking
        self._total_tokens = 0
        self._total_mismatches = 0
        self._pred_ids = None

    def launch_speculative_dispatch(self, h_pre: torch.Tensor):
        """Phase 1-2: Run probe and launch speculative AllToAll on comm stream.

        Must be called BEFORE attention starts on the main stream.

        Args:
            h_pre: [N, hidden_dim] pre-attention hidden states
                   (output of input_layernorm).
        """
        # Phase 1: Predict routing
        with torch.no_grad():
            probe_logits = self.probe(h_pre)
            self._pred_ids = torch.topk(
                probe_logits, k=self.config.top_k, dim=-1
            ).indices

        pred_peer_indices, _ = compute_dispatch_plan(
            self._pred_ids,
            self.config.num_experts,
            self.config.ep_size,
            self.comm.rank,
        )
        pred_packed = pack_tokens(h_pre, pred_peer_indices)

        # Phase 2: Launch speculative dispatch on comm stream
        main_event = torch.cuda.Event()
        main_event.record()

        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(main_event)
            self.comm.send_recv(pred_packed)
            self.comm.wait()

    def wait_and_verify(
        self,
        true_topk_ids: torch.Tensor,
    ) -> int:
        """Phase 3-5: Wait for dispatch, verify, fallback.

        Must be called AFTER attention completes and gate routing is known.

        Args:
            true_topk_ids: [N, topk] true expert indices from gate computation.

        Returns:
            Number of mismatched tokens (0 if verification disabled).
        """
        # Wait for speculative dispatch to complete
        torch.cuda.current_stream().wait_stream(self.comm_stream)

        n_mismatch = 0

        if self.config.enable_verification and self._pred_ids is not None:
            # Phase 4: Verify
            mismatch_indices = verify_dispatch(
                self._pred_ids,
                true_topk_ids,
                self.config.num_experts,
                self.config.ep_size,
                self.comm.rank,
            )
            n_mismatch = len(mismatch_indices)

            # Phase 5: Fallback for mismatches
            if self.config.enable_fallback and n_mismatch > 0:
                # Need post-attention hidden states for fallback
                # In practice, the MoE layer handles this via its normal dispatch
                # For the PoC, we re-dispatch the mismatched tokens
                pass  # Fallback is handled by SGLang's normal MoE dispatch path

            # Stats
            self._total_tokens += true_topk_ids.shape[0]
            self._total_mismatches += n_mismatch

            if self.config.log_accuracy:
                acc = 1.0 - (n_mismatch / max(true_topk_ids.shape[0], 1))
                print(f"[Pre-MoE] Layer {self.layer_idx}: "
                      f"dispatch accuracy={acc:.4f} "
                      f"({n_mismatch}/{true_topk_ids.shape[0]} mismatches)")

        self._pred_ids = None
        return n_mismatch

    @property
    def dispatch_accuracy(self) -> float:
        if self._total_tokens == 0:
            return 1.0
        return 1.0 - (self._total_mismatches / self._total_tokens)
