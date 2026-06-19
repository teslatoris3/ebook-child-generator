"""Fast LLM-only trial: judge a model's text quality + kid-safety, no SD/GPU image work.

Loads a GGUF (default: the configured one; --model to override), runs the real
StoryWriter outline + stanza prompts, and prints the structured beats (place +
activity per page) and the matching stanzas so we can judge whether the model
varies places and matches each poem to its activity — before any full book run.

  python scripts/try_llm.py --model /path/to/model.gguf --gpu-layers 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from pipeline.device import DevicePolicy  # noqa: E402
from pipeline.questionnaire import Answers  # noqa: E402
from pipeline.story import StoryWriter  # noqa: E402

ANSWERS = Answers(
    child_name="Luna", pronoun="girl", hair_color="blonde", skin_tone="light",
    favourite_animal="dragon", loved_one="Mom", theme="being brave",
    setting="enchanted forest", art_style="watercolor children's book",
    favourite_activities="cooking, bathing, painting, dancing, reading",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="GGUF path (default: config)")
    ap.add_argument("--gpu-layers", type=int, default=0, help="layers to offload to GPU")
    ap.add_argument("--pages", type=int, default=4, help="how many stanzas to sample")
    args = ap.parse_args()

    config = Config()
    if args.model:
        # Config.paths is frozen; rebuild with the override.
        import dataclasses
        config.paths = dataclasses.replace(config.paths, llm=Path(args.model))
    config.num_pages = args.pages

    device = DevicePolicy(
        llm_device=("cuda" if args.gpu_layers else "cpu"),
        sd_device="cuda", vram_gb=3.8, reason="trial",
    )
    config.llm_gpu_layers = args.gpu_layers

    print(f"Model: {config.paths.llm}")
    print(f"GPU layers: {args.gpu_layers}")
    print("-" * 60)

    writer = StoryWriter(config, device)
    t0 = time.perf_counter()
    writer.load()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    print("=== OUTLINE (place + activity per page) ===")
    t0 = time.perf_counter()
    outline = writer.generate_outline(ANSWERS)
    print(f"(outline in {time.perf_counter() - t0:.1f}s)")
    for i, b in enumerate(outline):
        print(f"  p{i+1}: place={b.place!r:30} activity={b.activity!r}")
        print(f"        beat: {b.text}")

    print("=== STANZAS (should match each activity) ===")
    t0 = time.perf_counter()
    stanzas = writer.generate_stanzas(outline, ANSWERS)
    dt = time.perf_counter() - t0
    for i, (b, s) in enumerate(zip(outline, stanzas)):
        print(f"--- page {i+1} (activity: {b.activity}) ---")
        print(s)
    print(f"\n({len(stanzas)} stanzas in {dt:.1f}s, {dt/max(len(stanzas),1):.1f}s each)")

    print("=== TITLE ===")
    print(writer.generate_title(ANSWERS, outline))
    writer.unload()


if __name__ == "__main__":
    main()
