"""Top-level orchestrator: questionnaire -> finished PDF, emitting progress.

Ties the stages together (story -> prompts -> reference -> pages -> compose ->
PDF), writes the per-book output folder, and yields ``Progress`` updates so the
Gradio UI can stream status + previews. The ``BookRecord`` (``book.json``)
records answers, prompts and seeds so any single page can be regenerated
reproducibly.

The two heavy stacks are run one at a time via ``exclusive`` so they never
co-reside on the 4 GB GPU (see CONTEXT.md "Model Stack").
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from . import compose, prompts
from .device import DevicePolicy, pick_devices
from .images import ImageStudio
from .record import REFERENCE_INDEX, BookRecord, PageRecord, seed_for
from .stack import exclusive
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


class BookGenerator:
    """Owns the writer + studio for one session; produces books."""

    def __init__(self, config: "Config", device_policy: DevicePolicy | None = None) -> None:
        self.config = config
        self.device = device_policy or pick_devices()
        self.writer = StoryWriter(config, self.device)
        self.studio = ImageStudio(config, self.device)

    # -- helpers -------------------------------------------------------------
    def _output_dir_for(self, answers: "Answers") -> Path:
        """``output/<child_name>_<timestamp>/`` (name is validated fs-safe)."""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = Path(self.config.output_dir) / f"{answers.child_name}_{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _font(self):
        return compose.load_font(Path(self.config.fonts_dir), size=28)

    # -- generation ----------------------------------------------------------
    def generate(self, answers: "Answers") -> Iterator[Progress | BookResult]:
        """Run the full pipeline, yielding ``Progress`` then a final ``BookResult``.

        Text first (LLM), then images (SD), each phase wrapped in ``exclusive``
        so only one stack is resident at a time. The ``BookRecord`` is built as
        we go and saved as ``book.json`` for reproducible rerolls.
        """
        answers.validate()
        out_dir = self._output_dir_for(answers)
        record = BookRecord.for_answers(answers, self.config)
        total = self.config.num_pages

        # --- Text phase (LLM) ---
        yield Progress("story", 0, total, "Writing the story…")
        with exclusive(self.writer):
            outline = self.writer.generate_outline(answers)
            stanzas = self.writer.generate_stanzas(outline, answers)
            title = self.writer.generate_title(answers, outline)
        record.title = title

        # --- Image phase (SD) ---
        font = self._font()
        page_paths: list[Path] = []
        with exclusive(self.studio):
            yield Progress("reference", 0, total, "Designing the hero…")
            reference = self.studio.make_reference(answers, seed=record.reference_seed)
            reference_path = out_dir / "reference.png"
            reference.save(reference_path)
            record.reference_path = reference_path.name
            yield Progress("reference", 0, total, "Hero ready.", preview_path=reference_path)

            composed_pages = []
            for i, (beat, stanza) in enumerate(zip(outline, stanzas)):
                prompt = prompts.page_image_prompt(beat, answers)
                seed = seed_for(answers, i, self.config)
                illustration = self.studio.make_page(prompt, reference, seed)
                page = compose.compose_page(illustration, stanza, font)
                page_path = out_dir / f"page_{i + 1:02d}.png"
                page.save(page_path)

                composed_pages.append(page)
                page_paths.append(page_path)
                record.pages.append(
                    PageRecord(prompt=prompt, seed=seed, stanza=stanza, image_path=page_path.name)
                )
                yield Progress("page", i + 1, total, f"Illustrating page {i + 1}…",
                               preview_path=page_path)

            # Dedicated cover, conditioned on the reference hero.
            yield Progress("cover", total, total, "Painting the cover…")
            cover_prompt = prompts.cover_image_prompt(title, answers)
            cover_seed = seed_for(answers, REFERENCE_INDEX, self.config) + 1
            cover_img = self.studio.make_page(cover_prompt, reference, cover_seed)
            cover = compose.make_cover(title, answers.child_name, cover_img, font)
            cover_path = out_dir / "cover.png"
            cover.save(cover_path)

        # --- Compose + persist ---
        yield Progress("compose", total, total, "Binding the book…")
        pdf_path = out_dir / "book.pdf"
        compose.to_pdf([cover] + composed_pages, pdf_path)
        record.save(out_dir)

        yield BookResult(
            title=title,
            output_dir=out_dir,
            pdf_path=pdf_path,
            reference_path=reference_path,
            page_image_paths=page_paths,
            stanzas=stanzas,
        )

    def regenerate_page(self, output_dir: Path, page_index: int, new_seed: int | None = None) -> Path:
        """Re-render one page from the stored ``BookRecord`` -> new image path.

        Loads ``book.json``, rerolls the page's seed (``new_seed`` or stored + 1),
        re-renders that page conditioned on the saved reference, recomposes it,
        re-saves the record, and rebuilds the PDF. Returns the page image path.
        """
        out_dir = Path(output_dir)
        record = BookRecord.load(out_dir)
        answers = record.answers
        font = self._font()

        seed = record.bump_seed(page_index, new_seed)
        page_rec = record.pages[page_index]

        with exclusive(self.studio):
            reference = self.studio.make_reference(answers, seed=record.reference_seed)
            illustration = self.studio.make_page(page_rec.prompt, reference, seed)

        page = compose.compose_page(illustration, page_rec.stanza, font)
        page_path = out_dir / page_rec.image_path
        page.save(page_path)

        record.save(out_dir)
        self._rebuild_pdf(out_dir, record)
        return page_path

    def _rebuild_pdf(self, out_dir: Path, record: BookRecord) -> Path:
        """Reassemble book.pdf from the saved cover + page images on disk."""
        from PIL import Image

        pages = []
        cover_path = out_dir / "cover.png"
        if cover_path.exists():
            pages.append(Image.open(cover_path))
        pages.extend(
            Image.open(out_dir / p.image_path) for p in record.pages if p.image_path
        )
        pdf_path = out_dir / "book.pdf"
        compose.to_pdf(pages, pdf_path)
        return pdf_path
