"""Adaptive device policy (the spirit of CLAUDE.md's old ``tier.rs``).

On the 4 GB target the LLM and the SD+IP-Adapter stack cannot co-reside in VRAM,
so the default is **LLM on CPU, SD on GPU** with no swapping. On larger cards the
LLM is promoted to GPU. Environment variables ``KIDSBOOK_LLM_DEVICE`` /
``KIDSBOOK_SD_DEVICE`` override the automatic choice.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Below this much VRAM, keep the LLM on CPU (SD+IP-Adapter needs ~4 GB alone).
VRAM_GPU_LLM_THRESHOLD_GB = 6.0


@dataclass(frozen=True)
class DevicePolicy:
    """Resolved device placement for the two heavy stacks."""

    llm_device: str  # "cpu" | "cuda"
    sd_device: str   # "cpu" | "cuda"
    vram_gb: float
    reason: str


def _detect_vram_gb() -> float:
    """Total VRAM of GPU 0 in GiB, or 0.0 if CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def pick_devices(
    force_llm: str | None = None,
    force_sd: str | None = None,
) -> DevicePolicy:
    """Decide where the LLM and SD stacks run.

    Args:
        force_llm: override LLM device ("cpu"/"cuda"); falls back to env then auto.
        force_sd: override SD device ("cpu"/"cuda"); falls back to env then auto.
    """
    vram = _detect_vram_gb()
    has_cuda = vram > 0.0

    force_llm = force_llm or os.getenv("KIDSBOOK_LLM_DEVICE")
    force_sd = force_sd or os.getenv("KIDSBOOK_SD_DEVICE")

    sd_device = force_sd or ("cuda" if has_cuda else "cpu")

    if force_llm:
        llm_device = force_llm
    elif has_cuda and vram >= VRAM_GPU_LLM_THRESHOLD_GB:
        llm_device = "cuda"
    else:
        llm_device = "cpu"

    reason = (
        f"vram={vram:.1f}GiB; "
        f"llm->{llm_device} (threshold {VRAM_GPU_LLM_THRESHOLD_GB}GiB), "
        f"sd->{sd_device}"
    )
    return DevicePolicy(llm_device=llm_device, sd_device=sd_device, vram_gb=vram, reason=reason)
