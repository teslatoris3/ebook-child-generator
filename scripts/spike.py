"""PLAN.md Phase 2 feasibility spike (run BEFORE building real logic).

Goal: prove SD 1.5 + IP-Adapter + LCM-LoRA fit in 4 GB VRAM at usable speed.
Generates one reference image and one IP-Adapter-conditioned page, then prints
peak VRAM and per-image wall time so we can lock the memory settings.

Run:  python scripts/spike.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402


def main() -> None:
    """TODO (Phase 2): load DreamShaper V8 + LCM-LoRA + IP-Adapter, generate
    1 reference + 1 page, print torch.cuda.max_memory_allocated() and timings.
    Decide attention_slicing/vae_tiling vs enable_model_cpu_offload from results.
    """
    config = Config()
    _ = config  # placeholder until implemented
    raise NotImplementedError("Phase 2 spike not implemented yet")


if __name__ == "__main__":
    main()
