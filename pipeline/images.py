"""Illustration generation: diffusers SD 1.5 + IP-Adapter + LCM-LoRA.

Owns the GPU. ``load()`` builds the pipeline once (DreamShaper V8 base, LCM-LoRA
fused for few-step sampling, IP-Adapter loaded, memory optimizations applied).
The hero child is generated once as a *reference* image, then every page is
conditioned on that reference via IP-Adapter to keep the child consistent.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .device import DevicePolicy
from .prompts import character_sheet

if TYPE_CHECKING:
    from PIL.Image import Image

    from config import Config

    from .questionnaire import Answers

# Neutral skin-tone fill for the reference image's IP-Adapter slot. The CLIP
# image encoder needs *some* valid input even though ip scale is 0 there.
_NEUTRAL_IP_FILL = (210, 180, 155)


class ImageStudio:
    """Lazy wrapper around the diffusers SD+IP-Adapter pipeline."""

    def __init__(self, config: "Config", device_policy: DevicePolicy) -> None:
        self.config = config
        self.device = device_policy
        self._pipe = None  # StableDiffusionPipeline, created in load()

    # -- lifecycle -----------------------------------------------------------
    def load(self) -> None:
        """Build the SD pipeline (DreamShaper V8 + LCM-LoRA + IP-Adapter).

        Mirrors ``scripts/spike.py``: fp16 variant load, LCMScheduler, LCM-LoRA
        weights, IP-Adapter, then ``enable_model_cpu_offload`` (the locked memory
        mode for this 4 GB machine — see config.sd_memory_mode).
        """
        import torch
        from diffusers import LCMScheduler, StableDiffusionPipeline

        paths = self.config.paths
        pipe = StableDiffusionPipeline.from_pretrained(
            str(paths.sd_base),
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
        )

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(str(paths.lcm_lora_dir))

        pipe.load_ip_adapter(
            str(paths.ip_adapter_dir),
            subfolder=paths.ip_adapter_subfolder,
            weight_name=paths.ip_adapter_weight,
        )
        pipe.set_ip_adapter_scale(0.0)  # neutral; set per-call

        pipe.enable_model_cpu_offload()

        self._pipe = pipe

    def unload(self) -> None:
        """Release the pipeline and empty the CUDA cache."""
        self._pipe = None
        import torch

        torch.cuda.empty_cache()

    # -- generation ----------------------------------------------------------
    def _generator(self, seed: int):
        import torch

        return torch.Generator(device="cuda").manual_seed(seed)

    def make_reference(self, answers: "Answers", seed: int) -> "Image":
        """Generate the one hero reference image (child only, no IP conditioning).

        Plain text-to-image: IP scale 0 with a neutral fill image so the CLIP
        encoder gets valid input but contributes nothing. The prompt is the
        frozen character sheet plus the chosen art style, framed as a portrait.
        """
        from PIL import Image
        from .questionnaire import ART_STYLE_FRAGMENT

        style = ART_STYLE_FRAGMENT.get(answers.art_style, answers.art_style)
        prompt = (
            f"{style}, portrait of {character_sheet(answers)}, "
            f"front and center, friendly, cute"
        )
        neutral = Image.new("RGB", (224, 224), _NEUTRAL_IP_FILL)

        self._pipe.set_ip_adapter_scale(0.0)
        out = self._pipe(
            prompt=prompt,
            ip_adapter_image=neutral,
            height=self.config.image_size[1],
            width=self.config.image_size[0],
            num_inference_steps=self.config.lcm_steps,
            guidance_scale=self.config.guidance_scale,
            generator=self._generator(seed),
        )
        return out.images[0]

    def make_page(
        self,
        prompt: str,
        reference: "Image",
        seed: int,
        ip_scale: float | None = None,
    ) -> "Image":
        """Generate one page illustration conditioned on ``reference`` via IP-Adapter.

        The hero child is kept consistent by conditioning every page on the same
        reference image at ``config.ip_scale`` (override per call with ``ip_scale``).
        """
        scale = self.config.ip_scale if ip_scale is None else ip_scale
        self._pipe.set_ip_adapter_scale(scale)
        out = self._pipe(
            prompt=prompt,
            ip_adapter_image=reference,
            height=self.config.image_size[1],
            width=self.config.image_size[0],
            num_inference_steps=self.config.lcm_steps,
            guidance_scale=self.config.guidance_scale,
            generator=self._generator(seed),
        )
        return out.images[0]
