"""Illustration generation: diffusers SD 1.5 + IP-Adapter + LCM-LoRA.

Owns the GPU. ``load()`` builds the pipeline once (DreamShaper V8 base, LCM-LoRA
fused for few-step sampling, IP-Adapter loaded, memory optimizations applied).
The hero child is generated once as a *reference* image, then every page is
conditioned on that reference via IP-Adapter to keep the child consistent.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .device import DevicePolicy
from .prompts import NEGATIVE_PROMPT

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

        Memory strategy (CLAUDE.md / Phase 2):
          - fp16 to halve VRAM baseline.
          - LCM-LoRA fused so we keep the base UNet (no separate LoRA weight in VRAM).
          - IP-Adapter loaded (CLIP image encoder + cross-attn projections).
          - attention_slicing + vae_tiling always on.
          - pipe.to("cuda") attempted first; falls back to enable_model_cpu_offload()
            if that OOMs (e.g. first boot before the Phase-2 spike was run).
        """
        import torch
        from diffusers import LCMScheduler, StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            str(self.config.paths.sd_base),
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Fuse LCM-LoRA so it counts as part of the UNet (no extra adapter layers kept).
        pipe.load_lora_weights(str(self.config.paths.lcm_lora_dir))
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        # Load IP-Adapter before memory optimizations so the image encoder is covered.
        pipe.load_ip_adapter(
            str(self.config.paths.ip_adapter_dir),
            subfolder=self.config.paths.ip_adapter_subfolder,
            weight_name=self.config.paths.ip_adapter_weight,
        )

        pipe.enable_attention_slicing()
        pipe.enable_vae_tiling()

        try:
            pipe = pipe.to(self.device.sd_device)
        except RuntimeError:
            # OOM on direct .to("cuda") — fall back to sequential CPU offload.
            print(
                "[ImageStudio] OOM moving full pipeline to GPU; "
                "enabling model_cpu_offload (slower but fits 4 GB)."
            )
            pipe.enable_model_cpu_offload()

        self._pipe = pipe

    def unload(self) -> None:
        """Release the pipeline and empty the CUDA cache."""
        import torch
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- generation ----------------------------------------------------------

    def make_reference(self, answers: "Answers", seed: int) -> "Image":
        """Generate the one hero reference image (child only, no IP conditioning)."""
        import torch
        from .prompts import character_sheet
        from .questionnaire import ART_STYLE_FRAGMENT

        assert self._pipe is not None, "Call load() first"

        # Disable IP-Adapter for the reference — no conditioning image exists yet.
        self._pipe.set_ip_adapter_scale(0.0)

        style = ART_STYLE_FRAGMENT[answers.art_style]
        hero = character_sheet(answers)
        prompt = (
            f"{style}, {hero}, portrait, front-facing, full body, "
            "plain background, children's book illustration, "
            "soft lighting, vibrant colors"
        )
        generator = torch.Generator(device=self.device.sd_device).manual_seed(seed)
        result = self._pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=self.config.lcm_steps,
            guidance_scale=self.config.guidance_scale,
            width=self.config.image_size[0],
            height=self.config.image_size[1],
            generator=generator,
        )
        return result.images[0]

    def make_page(
        self,
        prompt: str,
        reference: "Image",
        seed: int,
        ip_scale: float | None = None,
    ) -> "Image":
        """Generate one page illustration conditioned on ``reference`` via IP-Adapter."""
        import torch

        assert self._pipe is not None, "Call load() first"

        scale = ip_scale if ip_scale is not None else self.config.ip_scale
        self._pipe.set_ip_adapter_scale(scale)

        generator = torch.Generator(device=self.device.sd_device).manual_seed(seed)
        result = self._pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            ip_adapter_image=reference,
            num_inference_steps=self.config.lcm_steps,
            guidance_scale=self.config.guidance_scale,
            width=self.config.image_size[0],
            height=self.config.image_size[1],
            generator=generator,
        )
        return result.images[0]
