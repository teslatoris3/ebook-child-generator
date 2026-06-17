"""Central configuration: model paths, generation defaults, IO locations.

This is the single source of truth for paths and tunables. Nothing here loads a
model or does heavy work — it only describes *where things are* and *what knobs
exist*. Pipeline modules receive a ``Config`` instance; they never hardcode paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Filesystem anchors -----------------------------------------------------
MODELS_ROOT = Path("~/models").expanduser()
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class Paths:
    """Locations of every model artifact on disk (see CLAUDE.md)."""

    models_root: Path = MODELS_ROOT
    # Text LLM (GGUF, runs via llama-cpp-python).
    llm: Path = MODELS_ROOT / "llm" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    llm_fallback: Path = MODELS_ROOT / "llm" / "smollm2-1.7b-instruct-q4_k_m.gguf"
    # Stable Diffusion 1.5 base (diffusers format dir).
    sd_base: Path = MODELS_ROOT / "sd" / "dreamshaper-8"
    # IP-Adapter weights + CLIP image encoder.
    ip_adapter_dir: Path = MODELS_ROOT / "sd" / "ip-adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight: str = "ip-adapter_sd15.bin"
    # LCM-LoRA for SD 1.5 (few-step sampling).
    lcm_lora_dir: Path = MODELS_ROOT / "sd" / "lcm-lora-sdv1-5"


@dataclass
class Config:
    """All runtime tunables for one book-generation session."""

    paths: Paths = field(default_factory=Paths)

    # --- Book shape (locked in CLAUDE.md) ---
    num_pages: int = 8
    age_range: str = "3-6"
    lines_per_page: int = 4

    # --- Image generation ---
    image_size: tuple[int, int] = (512, 512)
    lcm_steps: int = 6            # LCM-LoRA: 4-8 steps
    guidance_scale: float = 1.5   # low CFG for LCM
    ip_scale: float = 0.6         # IP-Adapter conditioning strength (child identity)
    base_seed: int = 0            # 0 -> derive deterministically from child name

    # --- Text generation (llama-cpp-python) ---
    llm_ctx: int = 4096
    llm_threads: int = 0          # 0 -> let llama.cpp auto-pick
    llm_gpu_layers: int = 0       # set by device policy; 0 = CPU
    llm_max_tokens: int = 512
    llm_temperature: float = 0.8

    # --- IO ---
    output_dir: Path = OUTPUT_DIR
    fonts_dir: Path = FONTS_DIR

    def ensure_dirs(self) -> None:
        """Create output/asset dirs if missing (safe to call repeatedly)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
