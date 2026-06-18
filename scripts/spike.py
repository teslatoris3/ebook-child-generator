"""PLAN.md Phase 2 feasibility spike.

Goal: prove SD 1.5 + IP-Adapter + LCM-LoRA fit in 4 GB VRAM at usable speed.
Generates one reference image and one IP-Adapter-conditioned page, then prints
peak VRAM and per-image wall time so we can lock the memory settings for images.py.

Run:  python scripts/spike.py
Outputs saved to:  output/spike/
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402


def main() -> None:
    import torch
    from diffusers import LCMScheduler, StableDiffusionPipeline

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This spike requires a GPU.")
        sys.exit(1)

    config = Config()

    for label, path in [
        ("SD base", config.paths.sd_base),
        ("LCM-LoRA", config.paths.lcm_lora_dir),
        ("IP-Adapter", config.paths.ip_adapter_dir),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            print("Run the Phase 1 downloads first (see PLAN.md).")
            sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # ---- Load pipeline ----
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    print(f"Loading DreamShaper V8 from {config.paths.sd_base} …")

    pipe = StableDiffusionPipeline.from_pretrained(
        str(config.paths.sd_base),
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.load_lora_weights(str(config.paths.lcm_lora_dir))
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(
        str(config.paths.ip_adapter_dir),
        subfolder=config.paths.ip_adapter_subfolder,
        weight_name=config.paths.ip_adapter_weight,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()

    cpu_offload = False
    try:
        pipe = pipe.to("cuda")
        print("Loaded to GPU directly.")
    except RuntimeError as exc:
        print(f"OOM moving to GPU ({exc})\nFalling back to enable_model_cpu_offload().")
        pipe.enable_model_cpu_offload()
        cpu_offload = True

    load_time = time.time() - t0
    load_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"Pipeline ready in {load_time:.1f}s | peak VRAM: {load_vram:.2f} GB\n")

    neg = (
        "blurry, deformed, extra limbs, bad anatomy, text, watermark, "
        "signature, low quality, scary, nsfw, photo, realistic"
    )

    # ---- Reference image (no IP-Adapter) ----
    pipe.set_ip_adapter_scale(0.0)
    gen = torch.Generator("cuda").manual_seed(42)
    torch.cuda.reset_peak_memory_stats()

    print("Generating reference image (IP-Adapter disabled) …")
    t1 = time.time()
    ref_result = pipe(
        prompt=(
            "watercolor children's book, a little girl with blonde hair and light skin, "
            "big friendly eyes, portrait, plain background, children's book illustration"
        ),
        negative_prompt=neg,
        num_inference_steps=config.lcm_steps,
        guidance_scale=config.guidance_scale,
        width=config.image_size[0],
        height=config.image_size[1],
        generator=gen,
    )
    ref_img = ref_result.images[0]
    ref_time = time.time() - t1
    ref_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  done: {ref_time:.1f}s | peak VRAM: {ref_vram:.2f} GB")

    # ---- Page image (IP-Adapter conditioned) ----
    pipe.set_ip_adapter_scale(config.ip_scale)
    gen2 = torch.Generator("cuda").manual_seed(123)
    torch.cuda.reset_peak_memory_stats()

    print("\nGenerating page 1 (IP-Adapter conditioned on reference) …")
    t2 = time.time()
    page_result = pipe(
        prompt=(
            "watercolor children's book, a little girl with blonde hair and light skin, "
            "big friendly eyes, playing in an enchanted forest with a friendly little dragon, "
            "children's book illustration, soft lighting, vibrant colors, whimsical"
        ),
        negative_prompt=neg,
        ip_adapter_image=ref_img,
        num_inference_steps=config.lcm_steps,
        guidance_scale=config.guidance_scale,
        width=config.image_size[0],
        height=config.image_size[1],
        generator=gen2,
    )
    page_time = time.time() - t2
    page_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  done: {page_time:.1f}s | peak VRAM: {page_vram:.2f} GB")

    # ---- Save outputs ----
    out = Path("output/spike")
    out.mkdir(parents=True, exist_ok=True)
    ref_img.save(out / "spike_reference.png")
    page_result.images[0].save(out / "spike_page1.png")
    print(f"\nImages saved to {out}/")

    # ---- Decision gate (PLAN.md Phase 2) ----
    peak = max(ref_vram, page_vram)
    print("\n========== SPIKE SUMMARY ==========")
    print(f"Load:      {load_time:.1f}s  |  {load_vram:.2f} GB peak")
    print(f"Reference: {ref_time:.1f}s   |  {ref_vram:.2f} GB peak")
    print(f"Page:      {page_time:.1f}s   |  {page_vram:.2f} GB peak")
    print(f"CPU offload used: {cpu_offload}")
    print()
    if peak < 3.8:
        print("RESULT: Fits comfortably — attention_slicing + vae_tiling is sufficient.")
        print("ACTION: Keep current images.py memory settings (no cpu_offload needed).")
    elif peak < 4.2:
        print("RESULT: Tight fit — monitor for OOM on larger batches.")
        print("ACTION: Keep attention_slicing + vae_tiling; consider cpu_offload as fallback.")
    else:
        print("RESULT: Exceeds 4 GB — enable_model_cpu_offload() is required.")
        print("ACTION: Set KIDSBOOK_SD_DEVICE=cpu or keep cpu_offload fallback in images.py.")
    print("====================================")


if __name__ == "__main__":
    main()
