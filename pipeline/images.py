"""Illustration generation: diffusers SD 1.5 + IP-Adapter + LCM-LoRA.

Owns the GPU. ``load()`` builds the pipeline once (DreamShaper V8 base, LCM-LoRA
fused for few-step sampling, IP-Adapter loaded, memory optimizations applied).
The hero child is generated once as a *reference* image, then every page is
conditioned on that reference via IP-Adapter to keep the child consistent.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .device import DevicePolicy

if TYPE_CHECKING:
    from PIL.Image import Image

    from config import Config

    from .questionnaire import Answers


class ImageStudio:
    """Lazy wrapper around the diffusers SD+IP-Adapter pipeline."""

    def __init__(self, config: "Config", device_policy: DevicePolicy) -> None:
        self.config = config
        self.device = device_policy
        self._pipe = None  # StableDiffusionPipeline, created in load()

    # -- lifecycle -----------------------------------------------------------
    def load(self) -> None:
        """Build the SD pipeline (DreamShaper V8 + LCM-LoRA + IP-Adapter).

        TODO (see PLAN.md Phase 2 for exact memory settings):
          - StableDiffusionPipeline.from_single_file/from_pretrained(sd_base, fp16)
          - load + fuse LCM-LoRA; set LCMScheduler
          - pipe.load_ip_adapter(ip_adapter_dir, subfolder, weight)
          - enable_attention_slicing(); enable_vae_tiling()
          - move to device; fall back to enable_model_cpu_offload() on OOM
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Release the pipeline and empty the CUDA cache. TODO: empty_cache()."""
        self._pipe = None

    # -- generation ----------------------------------------------------------
    def make_reference(self, answers: "Answers", seed: int) -> "Image":
        """Generate the one hero reference image (child only, no IP conditioning).

        TODO: prompt = character_sheet(answers) + style; fixed seed; return PIL.
        """
        raise NotImplementedError

    def make_page(
        self,
        prompt: str,
        reference: "Image",
        seed: int,
        ip_scale: float | None = None,
    ) -> "Image":
        """Generate one page illustration conditioned on ``reference`` via IP-Adapter.

        TODO: set ip_adapter_scale(ip_scale or config.ip_scale); run pipe with
        ip_adapter_image=reference, num_inference_steps=config.lcm_steps,
        guidance_scale=config.guidance_scale; return PIL.
        """
        raise NotImplementedError
