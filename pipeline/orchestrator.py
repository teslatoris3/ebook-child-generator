"""Top-level orchestrator: questionnaire -> finished PDF, emitting progress.

Ties the stages together (story -> prompts -> reference -> pages -> compose ->
PDF), writes the per-book output folder, and yields ``Progress`` updates so the
Gradio UI can stream status + previews. ``book.json`` records answers, prompts
and seeds so any single page can be regenerated reproducibly.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from . import compose, prompts as pmod
from .device import DevicePolicy, pick_devices
from .images import ImageStudio
from .story import StoryWriter

if TYPE_CHECKING:
    from config import Config

    from .questionnaire import Answers


@dataclass
class Progress:
    """One status tick streamed to the UI."""

    stage: str            # e.g. "story", "reference", "page", "compose"
    current: int
    total: int
    message: str
    preview_path: Path | None = None  # optional image to show as it's produced


@dataclass
class BookResult:
    """Everything a finished book produced."""

    title: str
    output_dir: Path
    pdf_path: Path
    reference_path: Path
    page_image_paths: list[Path] = field(default_factory=list)
    stanzas: list[str] = field(default_factory=list)


def _safe_name(child_name: str) -> str:
    return "".join(c for c in child_name if c.isalnum() or c in " _-").strip().replace(" ", "_")


class BookGenerator:
    """Owns the writer + studio for one session; produces books."""

    def __init__(self, config: "Config", device_policy: DevicePolicy | None = None) -> None:
        self.config = config
        self.device = device_policy or pick_devices()
        self.writer = StoryWriter(config, self.device)
        self.studio = ImageStudio(config, self.device)

    def _seed_for(self, answers: "Answers") -> int:
        """Deterministic per-book seed derived from the child's name."""
        if self.config.base_seed:
            return self.config.base_seed
        digest = hashlib.md5(answers.child_name.encode()).hexdigest()
        return int(digest[:8], 16) % (2 ** 31)

    def generate(self, answers: "Answers") -> Iterator[Progress | BookResult]:
        """Run the full pipeline, yielding ``Progress`` then a final ``BookResult``."""
        answers.validate()

        out_dir = (
            self.config.output_dir
            / f"{_safe_name(answers.child_name)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        base_seed = self._seed_for(answers)

        # --- Stage 1: Story (LLM on CPU) ---
        yield Progress("story", 1, 3, "Writing story outline…")
        self.writer.load()
        outline = self.writer.generate_outline(answers)

        yield Progress("story", 2, 3, "Writing rhyming stanzas…")
        stanzas = self.writer.generate_stanzas(outline, answers)

        yield Progress("story", 3, 3, "Writing title…")
        title = self.writer.generate_title(answers, outline)
        self.writer.unload()
        yield Progress("story", 3, 3, f'Story done: "{title}"')

        # --- Stage 2: Images (SD on GPU) ---
        yield Progress("reference", 0, 1, "Generating child reference image…")
        self.studio.load()
        reference = self.studio.make_reference(answers, seed=base_seed)
        ref_path = out_dir / "reference.png"
        reference.save(ref_path)
        yield Progress("reference", 1, 1, "Reference image ready.", preview_path=ref_path)

        font = compose.load_font(self.config.fonts_dir, size=22)
        page_image_paths: list[Path] = []
        book_pages: list = []
        prompts_used: list[str] = []
        seeds_used: list[int] = []

        for i, (beat, stanza) in enumerate(zip(outline, stanzas)):
            page_num = i + 1
            total = self.config.num_pages
            yield Progress("page", page_num, total, f"Illustrating page {page_num}/{total}…")

            prompt = pmod.page_image_prompt(beat, answers)
            seed = base_seed + page_num
            art = self.studio.make_page(prompt, reference, seed)

            img_path = out_dir / f"page_{page_num:02d}.png"
            art.save(img_path)
            page_image_paths.append(img_path)
            prompts_used.append(prompt)
            seeds_used.append(seed)

            composed = compose.compose_page(art, stanza, font)
            book_pages.append(composed)
            yield Progress("page", page_num, total, f"Page {page_num} done.", preview_path=img_path)

        self.studio.unload()

        # --- Stage 3: Compose + PDF ---
        yield Progress("compose", 1, 2, "Composing cover…")
        cover = compose.make_cover(title, answers.child_name, reference.copy(), font)
        all_pages = [cover] + book_pages

        yield Progress("compose", 2, 2, "Exporting PDF…")
        pdf_path = out_dir / "book.pdf"
        compose.to_pdf(all_pages, pdf_path)

        # Persist everything needed for reproducible reroll.
        book_json = {
            "answers": answers.to_dict(),
            "title": title,
            "outline": outline,
            "stanzas": stanzas,
            "prompts": prompts_used,
            "seeds": seeds_used,
            "ref_seed": base_seed,
            "output_dir": str(out_dir),
        }
        (out_dir / "book.json").write_text(json.dumps(book_json, indent=2))

        yield BookResult(
            title=title,
            output_dir=out_dir,
            pdf_path=pdf_path,
            reference_path=ref_path,
            page_image_paths=page_image_paths,
            stanzas=stanzas,
        )

    def regenerate_page(
        self,
        answers: "Answers",
        page_index: int,
        new_seed: int | None = None,
    ) -> Path:
        """Re-render a single page (reusing stored prompts/seeds) -> new image path."""
        from PIL import Image as PILImage

        # Find the most recent output dir for this child.
        candidates = sorted(
            self.config.output_dir.glob(f"{_safe_name(answers.child_name)}_*"),
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No previous run found for '{answers.child_name}'")
        out_dir = candidates[0]

        book_data = json.loads((out_dir / "book.json").read_text())
        prompt = book_data["prompts"][page_index]
        stanza = book_data["stanzas"][page_index]
        seed = new_seed if new_seed is not None else book_data["seeds"][page_index]

        reference = PILImage.open(out_dir / "reference.png")

        self.studio.load()
        art = self.studio.make_page(prompt, reference, seed)
        self.studio.unload()

        img_path = out_dir / f"page_{page_index + 1:02d}.png"
        art.save(img_path)

        font = compose.load_font(self.config.fonts_dir, size=22)

        # Rebuild all composed pages and the PDF.
        title = book_data["title"]
        cover = compose.make_cover(title, answers.child_name, reference.copy(), font)
        all_pages = [cover]
        for j, jstanza in enumerate(book_data["stanzas"]):
            raw = PILImage.open(out_dir / f"page_{j + 1:02d}.png")
            all_pages.append(compose.compose_page(raw, jstanza, font))

        compose.to_pdf(all_pages, out_dir / "book.pdf")

        # Update stored seed in book.json.
        book_data["seeds"][page_index] = seed
        (out_dir / "book.json").write_text(json.dumps(book_data, indent=2))

        return img_path
