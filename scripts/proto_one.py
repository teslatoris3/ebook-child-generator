"""Safe one-image-at-a-time scene prototype.

Each invocation loads the stack, generates the reference PLUS exactly ONE page
variant (passed by name), saves it, prints VRAM+RAM, and exits — so nothing runs
back-to-back unattended. Run repeatedly with different --variant values.

  python scripts/proto_one.py --variant baseline
  python scripts/proto_one.py --variant scene
  python scripts/proto_one.py --variant multichar
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from config import Config  # noqa: E402
from pipeline.device import pick_devices  # noqa: E402
from pipeline.images import ImageStudio  # noqa: E402
from pipeline.prompts import character_sheet, companion_desc  # noqa: E402
from pipeline.questionnaire import Answers  # noqa: E402

STYLE = "watercolor children's book illustration, soft colors"
ANSWERS = Answers(
    child_name="Luna", pronoun="girl", hair_color="blonde", skin_tone="light",
    favourite_animal="dragon", loved_one="Mom", theme="being brave",
    setting="enchanted forest", art_style="watercolor children's book",
)
SHEET = character_sheet(ANSWERS)
COMPANION = companion_desc(ANSWERS)

PLACE = "a cozy kitchen"
ACTIVITY = "baking cookies"
EXTRA = "a cat stirring a bowl, a rabbit holding a spoon, mom pouring flour"

PROMPTS = {
    "baseline": (f"{STYLE}, {SHEET}, {COMPANION}, in {PLACE}, "
                 f"children's book illustration", 0.6),
    "scene": (f"{STYLE}, {SHEET} {ACTIVITY}, {COMPANION}, in {PLACE}, "
              f"full scene, children's book illustration", 0.6),
    "multichar": (f"{STYLE}, in {PLACE}: {SHEET} {ACTIVITY}, with {EXTRA}, "
                  f"four characters doing different things, lively scene, "
                  f"children's book illustration", 0.4),
    # Probe: even lower lock + explicit wide full-body group framing.
    "wide_ip03": (f"{STYLE}, wide shot, full body, group scene in {PLACE}: "
                  f"{SHEET} {ACTIVITY}, together with {EXTRA}, "
                  f"several characters spread across the room, "
                  f"children's book illustration", 0.3),
    # Probe: low lock, plain multi-subject, no heavy activity emphasis.
    "group_ip03": (f"{STYLE}, in {PLACE}, a group of friends together: "
                   f"{SHEET}, a cat, a rabbit, and mom, "
                   f"wide cozy scene, children's book illustration", 0.3),
}


def mem() -> str:
    vram = torch.cuda.max_memory_allocated() / 1024**2
    ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB->MB
    return f"peak VRAM {vram:.0f} MB | peak RAM {ram:.0f} MB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(PROMPTS))
    args = ap.parse_args()
    prompt, ip = PROMPTS[args.variant]

    config = Config()
    out = config.output_dir / "proto"
    out.mkdir(parents=True, exist_ok=True)

    studio = ImageStudio(config, pick_devices())
    print("Loading studio...", flush=True)
    studio.load()
    print(f"  loaded | {mem()}", flush=True)

    ref_path = out / "reference.png"
    if ref_path.exists():
        from PIL import Image
        ref = Image.open(ref_path)
        print("  reusing existing reference.png", flush=True)
    else:
        print("Generating reference...", flush=True)
        ref = studio.make_reference(ANSWERS, seed=42)
        ref.save(ref_path)
        print(f"  reference done | {mem()}", flush=True)

    print(f"Generating variant '{args.variant}' (ip={ip})...", flush=True)
    print(f"  {prompt}", flush=True)
    t0 = time.perf_counter()
    img = studio.make_page(prompt, ref, seed=1234, ip_scale=ip)
    img.save(out / f"kitchen_{args.variant}.png")
    print(f"  done in {time.perf_counter() - t0:.1f}s | {mem()}", flush=True)

    studio.unload()
    print(f"Saved {out / f'kitchen_{args.variant}.png'}", flush=True)


if __name__ == "__main__":
    main()
