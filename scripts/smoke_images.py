"""Manual GPU smoke test for pipeline/images.ImageStudio.

Drives the real diffusers stack (no mocks) to visually confirm ImageStudio
produces a consistent hero across a reference + one page. Mirrors scripts/spike.py
but exercises the actual module the app will use.

Run:  python scripts/smoke_images.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from pipeline.device import pick_devices  # noqa: E402
from pipeline.images import ImageStudio  # noqa: E402
from pipeline.prompts import page_image_prompt  # noqa: E402
from pipeline.questionnaire import Answers  # noqa: E402

SEPARATOR = "-" * 60


def main() -> None:
    answers = Answers(
        child_name="Luna",
        pronoun="girl",
        hair_color="blonde",
        skin_tone="light",
        favourite_animal="dragon",
        loved_one="Mom",
        theme="being brave",
        setting="enchanted forest",
        art_style="watercolor children's book",
    )

    config = Config()
    config.ensure_dirs()
    out_dir = config.output_dir / "smoke_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    studio = ImageStudio(config, pick_devices())

    print(SEPARATOR)
    print("Loading ImageStudio (real diffusers stack)...")
    t0 = time.perf_counter()
    studio.load()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")
    print(SEPARATOR)

    print("Generating reference (IP scale 0)...")
    t0 = time.perf_counter()
    ref = studio.make_reference(answers, seed=42)
    ref.save(out_dir / "reference.png")
    print(f"  {time.perf_counter() - t0:.1f}s -> {out_dir / 'reference.png'}")

    beat = "explores a glowing clearing with the little dragon"
    prompt = page_image_prompt(beat, answers)
    print(f"Generating page (IP scale {config.ip_scale}) ...")
    print(f"  prompt: {prompt}")
    t0 = time.perf_counter()
    page = studio.make_page(prompt, ref, seed=99)
    page.save(out_dir / "page_01.png")
    print(f"  {time.perf_counter() - t0:.1f}s -> {out_dir / 'page_01.png'}")

    studio.unload()
    print(SEPARATOR)
    print(f"Done. Images in {out_dir}")


if __name__ == "__main__":
    main()
