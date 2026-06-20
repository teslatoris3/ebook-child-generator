"""End-to-end smoke run: questionnaire -> finished book PDF (real models).

Drives the full BookGenerator with the real StoryWriter (LLM on CPU) and
ImageStudio (SD on GPU) — no mocks — to confirm the whole pipeline produces a
book.pdf + book.json + page images. Mirrors what app.py will do.

Run:  python scripts/smoke_book.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from pipeline.orchestrator import BookGenerator, BookResult, Progress  # noqa: E402
from pipeline.questionnaire import Answers  # noqa: E402

SEP = "-" * 60


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--character-type", default="child", help="hero kind: child, dinosaur, alien, …")
    ap.add_argument("--name", default="Luna")
    ap.add_argument("--skin", default="light", help="skin/body color")
    args = ap.parse_args()

    answers = Answers(
        child_name=args.name,
        character_type=args.character_type,
        pronoun="girl",
        hair_color="blonde",
        skin_tone=args.skin,
        favourite_animal="dragon",
        loved_one="Mom",
        theme="being brave",
        setting="enchanted forest",
        art_style="watercolor children's book",
        favourite_activities="cooking, bathing, painting, dancing, reading",
    )

    config = Config()
    config.ensure_dirs()
    gen = BookGenerator(config)

    print(SEP)
    print(f"Device policy: {gen.device.reason}")
    print(SEP)

    t0 = time.perf_counter()
    result: BookResult | None = None
    for event in gen.generate(answers):
        if isinstance(event, Progress):
            print(f"  [{event.stage}] {event.current}/{event.total} — {event.message}")
        elif isinstance(event, BookResult):
            result = event

    elapsed = time.perf_counter() - t0
    print(SEP)
    assert result is not None, "generate() did not yield a BookResult"
    print(f"Title:      {result.title}")
    print(f"Output dir: {result.output_dir}")
    print(f"PDF:        {result.pdf_path}  ({result.pdf_path.stat().st_size // 1024} KB)")
    print(f"Pages:      {len(result.page_image_paths)} images")
    print(f"Total time: {elapsed:.1f}s")
    print(SEP)
    print("First stanza:\n" + result.stanzas[0])


if __name__ == "__main__":
    main()
