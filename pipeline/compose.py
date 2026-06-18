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
    """Load the bundled kid-friendly font at ``size``; fall back to PIL default."""
    from PIL import ImageFont
    for ttf in sorted(fonts_dir.glob("*.ttf")):
        try:
            return ImageFont.truetype(str(ttf), size)
        except Exception:
            pass
    # ImageFont.load_default() returns a basic bitmap font (no size param pre-10.x).
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text_width(draw: "ImageDraw.ImageDraw", text: str, font: "FreeTypeFont") -> int:  # type: ignore[name-defined]
    """Width of ``text`` rendered with ``font``, compatible with old Pillow."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    except AttributeError:
        # Pillow < 8.0 fallback
        return draw.textsize(text, font=font)[0]  # type: ignore[attr-defined]


def compose_page(image: "Image", stanza: str, font: "FreeTypeFont") -> "Image":
    """Draw stanza onto a copy of image in a semi-opaque band at the bottom."""
    from PIL import Image as PILImage, ImageDraw
    page = image.copy().convert("RGB")
    w, h = page.size
    lines = stanza.strip().splitlines()
    band_h = max(int(h * 0.28), len(lines) * 34 + 20)

    band = PILImage.new("RGBA", (w, band_h), (0, 0, 0, 180))
    page_rgba = page.convert("RGBA")
    page_rgba.paste(band, (0, h - band_h), band)
    page = page_rgba.convert("RGB")

    draw = ImageDraw.Draw(page)
    line_h = band_h // (len(lines) + 1)
    y = h - band_h + line_h // 2
    for line in lines:
        tw = _text_width(draw, line, font)
        draw.text(((w - tw) // 2, y), line, font=font, fill=(255, 250, 220))
        y += line_h
    return page


def make_cover(
    title: str,
    child_name: str,
    hero_image: "Image",
    font: "FreeTypeFont",
) -> "Image":
    """Build the cover page from the hero image + title text."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    cover = hero_image.copy().convert("RGB")
    w, h = cover.size

    band_h = int(h * 0.22)
    band = PILImage.new("RGBA", (w, band_h), (20, 10, 50, 210))
    cover_rgba = cover.convert("RGBA")
    cover_rgba.paste(band, (0, 0), band)
    cover = cover_rgba.convert("RGB")
    draw = ImageDraw.Draw(cover)

    tw = _text_width(draw, title, font)
    draw.text(((w - tw) // 2, band_h // 6), title, font=font, fill=(255, 235, 80))

    sub = f"A story for {child_name}"
    try:
        # Try to load a smaller version of the same font
        small_font = ImageFont.truetype(font.path, max(12, int(font.size * 0.65)))
    except Exception:
        small_font = font
    sw = _text_width(draw, sub, small_font)
    draw.text(((w - sw) // 2, band_h // 2 + 6), sub, font=small_font, fill=(190, 220, 255))
    return cover


def to_pdf(pages: Sequence["Image"], out_path: Path) -> Path:
    """Combine composited pages (cover first) into a single PDF."""
    if not pages:
        raise ValueError("No pages to write")
    rgb_pages = [p.convert("RGB") for p in pages]
    rgb_pages[0].save(
        out_path,
        save_all=True,
        append_images=rgb_pages[1:],
    )
    return out_path
