"""Page composition + PDF export (Pillow only, no HTML/PDF engine).

Each page = illustration with its stanza rendered onto/below it. A cover/title
page is prepended. All pages are combined into a single PDF.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont
import PIL.JpegImagePlugin   # force JPEG encoder registration for PDF export


def load_font(fonts_dir: Path, size: int) -> ImageFont.FreeTypeFont:
    """Load a bundled .ttf from fonts_dir at size; fall back to Pillow default."""
    for ttf in sorted(fonts_dir.glob("*.ttf")):
        try:
            return ImageFont.truetype(str(ttf), size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def compose_page(image: Image.Image, stanza: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Draw stanza onto a copy of image in a semi-opaque bottom band."""
    page = image.copy().convert("RGB")
    w, h = page.size
    band_h = max(80, h // 4)

    band = Image.new("RGBA", (w, band_h), (0, 0, 0, 160))
    base = page.convert("RGBA")
    base.paste(band, (0, h - band_h), band)
    page = base.convert("RGB")

    draw = ImageDraw.Draw(page)
    wrapped = textwrap.fill(stanza, width=40)
    draw.multiline_text(
        (w // 2, h - band_h + 10),
        wrapped,
        font=font,
        fill=(255, 255, 255),
        anchor="ma",
        align="center",
    )
    return page


def make_cover(title: str, child_name: str, hero_image: Image.Image, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Cover page: hero image with title + 'A story for <name>' overlaid."""
    cover = hero_image.copy().convert("RGB")
    w, h = cover.size
    draw = ImageDraw.Draw(cover)

    draw.multiline_text(
        (w // 2, h // 8),
        title,
        font=font,
        fill=(255, 255, 255),
        anchor="ma",
        align="center",
    )
    draw.text(
        (w // 2, h // 8 + 40),
        f"A story for {child_name}",
        font=font,
        fill=(220, 220, 220),
        anchor="ma",
    )
    return cover


def to_pdf(pages: Sequence[Image.Image], out_path: Path) -> Path:
    """Combine composited pages (cover first) into a single PDF."""
    if not pages:
        raise ValueError("to_pdf requires at least one page")
    # Embed as RGB. RGBA would force Pillow's JPXDecode (JPEG2000) path, which
    # most PDF viewers reject as "corrupted"; RGB uses widely-supported DCT/Flate.
    rgb_pages = [p.convert("RGB") for p in pages]
    rgb_pages[0].save(
        out_path,
        save_all=True,
        append_images=rgb_pages[1:],
    )
    return out_path
