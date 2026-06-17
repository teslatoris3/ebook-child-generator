"""Top-level orchestrator: questionnaire -> finished PDF, emitting progress.

Ties the stages together (story -> prompts -> reference -> pages -> compose ->
PDF), writes the per-book output folder, and yields ``Progress`` updates so the
Gradio UI can stream status + previews. ``book.json`` records answers, prompts
and seeds so any single page can be regenerated reproducibly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from . import compose, prompts
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


class BookGenerator:
    """Owns the writer + studio for one session; produces books."""

    def __init__(self, config: "Config", device_policy: DevicePolicy | None = None) -> None:
        self.config = config
        self.device = device_policy or pick_devices()
        self.writer = StoryWriter(config, self.device)
        self.studio = ImageStudio(config, self.device)

    def _seed_for(self, answers: "Answers") -> int:
        """Deterministic per-book seed (so the same inputs reproduce). TODO."""
        raise NotImplementedError

    def generate(self, answers: "Answers") -> Iterator[Progress | BookResult]:
        """Run the full pipeline, yielding ``Progress`` then a final ``BookResult``.

        Intended flow (TODO: implement):
          1. answers.validate(); create output dir
          2. writer.load(); outline -> stanzas -> title
          3. studio.load(); reference image
          4. for each page: build prompt -> make_page(reference) -> compose
          5. compose cover; to_pdf(...); write book.json
          6. yield BookResult
        """
        raise NotImplementedError

    def regenerate_page(self, answers: "Answers", page_index: int, new_seed: int | None = None) -> Path:
        """Re-render a single page (reusing stored prompts/seeds) -> new image path.

        TODO: load book.json for prompts, re-run make_page, recompose, rebuild PDF.
        """
        raise NotImplementedError
