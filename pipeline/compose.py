"""Page composition + PDF export (Pillow only, no HTML/PDF engine).

Each page = illustration with its stanza rendered onto/below it. A cover/title
page is prepended. All pages are combined into a single PDF.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from PIL.Image import Image
    from PIL.ImageFont import FreeTypeFont


def load_font(fonts_dir: Path, size: int) -> "FreeTypeFont":
    """Load the bundled kid-friendly font at ``size``.

    TODO: pick a bundled .ttf from fonts_dir; fall back to ImageFont.load_default().
    """
    raise NotImplementedError


def compose_page(image: "Image", stanza: str, font: "FreeTypeFont") -> "Image":
    """Draw ``stanza`` onto a copy of ``image`` (text band) -> composited page.

    TODO: add a semi-opaque band, word-wrap the stanza, center it, return image.
    """
    raise NotImplementedError


def make_cover(title: str, child_name: str, hero_image: "Image", font: "FreeTypeFont") -> "Image":
    """Build the cover page from the hero image + title + "A story for <name>". TODO."""
    raise NotImplementedError


def to_pdf(pages: Sequence["Image"], out_path: Path) -> Path:
    """Combine composited pages (cover first) into a single PDF at ``out_path``.

    TODO: pages[0].save(out_path, save_all=True, append_images=pages[1:]).
    """
    raise NotImplementedError
