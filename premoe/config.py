"""Pre-MoE configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PreMoEConfig:
    """Configuration for Pre-MoE speculative dispatch."""

    # Model architecture (DeepSeek-V2-Lite defaults)
    hidden_dim: int = 2048
    num_experts: int = 64
    top_k: int = 6
    ep_size: int = 2

    # Probe settings
    probe_dir: str = "probes"
    anchor_layers: List[int] = field(default_factory=lambda: list(range(1, 27)))

    # Pipeline control
    enable_verification: bool = True
    enable_fallback: bool = True

    # Communication
    comm_buffer_size_mb: int = 64
    comm_delay_us: int = 2000  # Simulated EP dispatch latency (μs). 0 = real NCCL only.

    # Debug / profiling
    log_accuracy: bool = False
    profile_phases: bool = False
