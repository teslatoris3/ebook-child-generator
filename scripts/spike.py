"""PLAN.md Phase 2 feasibility spike.

Proves SD 1.5 + IP-Adapter + LCM-LoRA fit in 3.8 GB VRAM at usable speed.
Generates one reference image and one IP-Adapter-conditioned page, then prints
peak VRAM and per-image wall time so we can lock the memory settings.

Run:  python scripts/spike.py
      python scripts/spike.py --cpu-offload   # force enable_model_cpu_offload path
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402

SEPARATOR = "-" * 60


def _vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


def _reset_vram_peak() -> None:
    torch.cuda.reset_peak_memory_stats()


def _load_pipeline(config: Config, cpu_offload: bool):
    from diffusers import LCMScheduler, StableDiffusionPipeline

    print("Loading SD pipeline (fp16)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        str(config.paths.sd_base),
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )

    print("Setting LCMScheduler + loading LCM-LoRA...")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(config.paths.lcm_lora_dir))

    print("Loading IP-Adapter...")
    pipe.load_ip_adapter(
        str(config.paths.ip_adapter_dir),
        subfolder=config.paths.ip_adapter_subfolder,
        weight_name=config.paths.ip_adapter_weight,
    )
    pipe.set_ip_adapter_scale(0.0)  # start at 0; set per-call

    if cpu_offload:
        print("Applying enable_model_cpu_offload...")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
        print("Applying attention_slicing + VAE tiling...")
        pipe.enable_attention_slicing()
        pipe.enable_vae_tiling()

    return pipe


def _gen_reference(pipe, config: Config) -> tuple:
    """Generate child reference image — plain text-to-image, IP scale=0.

    Pass a neutral PIL image so the CLIP encoder gets valid input; the scale=0
    means it contributes nothing to the UNet. No negative_prompt: LCM at low
    guidance + CFG can produce NaN in fp16.
    """
    from PIL import Image

    prompt = (
        "portrait of a young girl with blonde hair and light skin, "
        "watercolor children's book illustration, soft colors, cute, friendly"
    )
    neutral = Image.new("RGB", (224, 224), (210, 180, 155))  # skin-tone neutral

    pipe.set_ip_adapter_scale(0.0)
    _reset_vram_peak()
    t0 = time.perf_counter()

    out = pipe(
        prompt=prompt,
        ip_adapter_image=neutral,
        height=config.image_size[1],
        width=config.image_size[0],
        num_inference_steps=config.lcm_steps,
        guidance_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )
    elapsed = time.perf_counter() - t0
    vram = _vram_mb()
    return out.images[0], elapsed, vram


def _gen_page(pipe, ref_image, config: Config) -> tuple:
    """Generate page illustration — IP-Adapter conditioned on reference image."""
    prompt = (
        "a young girl with blonde hair explores an enchanted forest, "
        "watercolor children's book illustration, soft colors, magical"
    )

    pipe.set_ip_adapter_scale(config.ip_scale)
    _reset_vram_peak()
    t0 = time.perf_counter()

    out = pipe(
        prompt=prompt,
        ip_adapter_image=ref_image,
        height=config.image_size[1],
        width=config.image_size[0],
        num_inference_steps=config.lcm_steps,
        guidance_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(99),
    )
    elapsed = time.perf_counter() - t0
    vram = _vram_mb()
    return out.images[0], elapsed, vram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Force enable_model_cpu_offload (slower, less VRAM)")
    args = parser.parse_args()

    config = Config()
    config.ensure_dirs()
    out_dir = config.output_dir / "spike"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(SEPARATOR)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print(f"SD base: {config.paths.sd_base}")
    print(f"LCM-LoRA: {config.paths.lcm_lora_dir}")
    print(f"IP-Adapter: {config.paths.ip_adapter_dir}/{config.paths.ip_adapter_subfolder}/{config.paths.ip_adapter_weight}")
    print(f"Mode: {'cpu_offload' if args.cpu_offload else 'cuda + attention_slicing + vae_tiling'}")
    print(SEPARATOR)

    try:
        t_load = time.perf_counter()
        pipe = _load_pipeline(config, cpu_offload=args.cpu_offload)
        load_elapsed = time.perf_counter() - t_load
        print(f"Pipeline loaded in {load_elapsed:.1f}s")
        print(SEPARATOR)

        print("Generating reference image (text-to-image, IP scale=0)...")
        ref_img, ref_time, ref_vram = _gen_reference(pipe, config)
        ref_img.save(out_dir / "reference.png")
        print(f"  Time: {ref_time:.1f}s  |  Peak VRAM: {ref_vram:.0f} MB")

        print("Generating page 1 (IP-Adapter conditioned, IP scale=0.6)...")
        page_img, page_time, page_vram = _gen_page(pipe, ref_img, config)
        page_img.save(out_dir / "page_01.png")
        print(f"  Time: {page_time:.1f}s  |  Peak VRAM: {page_vram:.0f} MB")

        print(SEPARATOR)
        print("RESULTS SUMMARY")
        print(SEPARATOR)
        print(f"  Load time:         {load_elapsed:.1f}s")
        print(f"  Reference gen:     {ref_time:.1f}s  (peak {ref_vram:.0f} MB)")
        print(f"  Page gen:          {page_time:.1f}s  (peak {page_vram:.0f} MB)")
        print(f"  Est. 8-page total: ~{8 * page_time:.0f}s")
        total_vram = max(ref_vram, page_vram)
        budget_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        headroom_mb = budget_mb - total_vram
        print(f"  Peak VRAM:         {total_vram:.0f} MB / {budget_mb:.0f} MB  ({headroom_mb:.0f} MB headroom)")
        print(SEPARATOR)
        if args.cpu_offload:
            if headroom_mb >= 200:
                print("VERDICT: enable_model_cpu_offload fits with comfortable headroom. Use this path.")
            else:
                print("VERDICT: cpu_offload is tight — consider enable_sequential_cpu_offload.")
        else:
            if headroom_mb >= 200:
                print("VERDICT: fp16 + attention_slicing + vae_tiling fits. Use this path.")
            else:
                print("VERDICT: tight — re-run with --cpu-offload.")
        print(f"Images saved to {out_dir}")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\nOOM: {e}")
        if not args.cpu_offload:
            print("Retry with --cpu-offload")
        sys.exit(1)


if __name__ == "__main__":
    main()
