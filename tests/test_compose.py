"""Tests for compose.py — Pillow page composition and PDF export."""
import pytest
from pathlib import Path
from PIL import Image as PILImage
from pipeline import compose


@pytest.fixture
def tiny_image() -> PILImage.Image:
    return PILImage.new("RGB", (64, 64), color=(100, 149, 237))


@pytest.fixture
def font(tmp_path):
    return compose.load_font(tmp_path, size=12)  # tmp_path has no .ttf -> default


# --- tracer bullet: load_font returns something usable ---

def test_load_font_returns_a_font(font):
    assert font is not None


# --- compose_page ---

def test_compose_page_returns_rgb_image(tiny_image, font):
    result = compose.compose_page(tiny_image, "Hello world\nThis is a test", font)
    assert result.mode == "RGB"


def test_compose_page_preserves_dimensions(tiny_image, font):
    result = compose.compose_page(tiny_image, "A stanza line here", font)
    assert result.size == tiny_image.size


def test_compose_page_does_not_mutate_input(tiny_image, font):
    original_pixels = tiny_image.tobytes()
    compose.compose_page(tiny_image, "text", font)
    assert tiny_image.tobytes() == original_pixels


# --- make_cover ---

def test_make_cover_returns_rgb_image(tiny_image, font):
    cover = compose.make_cover("The Dragon's Quest", "Luna", tiny_image, font)
    assert cover.mode == "RGB"


def test_make_cover_preserves_dimensions(tiny_image, font):
    cover = compose.make_cover("Title", "Alex", tiny_image, font)
    assert cover.size == tiny_image.size


# --- to_pdf ---

def test_to_pdf_creates_file(tmp_path, tiny_image):
    pages = [tiny_image] * 3
    out = tmp_path / "book.pdf"
    result = compose.to_pdf(pages, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_to_pdf_has_pdf_magic_bytes(tmp_path, tiny_image):
    out = tmp_path / "book.pdf"
    compose.to_pdf([tiny_image, tiny_image], out)
    assert out.read_bytes()[:4] == b"%PDF"


def test_to_pdf_does_not_use_jpeg2000(tmp_path, tiny_image):
    """JPXDecode (JPEG2000) pages render as 'corrupted' in most viewers.

    Pages must be embedded with a widely-supported filter (DCT/Flate), so the
    JPXDecode signature must be absent from the output.
    """
    out = tmp_path / "book.pdf"
    compose.to_pdf([tiny_image, tiny_image], out)
    assert b"JPXDecode" not in out.read_bytes()


def test_to_pdf_single_page(tmp_path, tiny_image):
    out = tmp_path / "single.pdf"
    compose.to_pdf([tiny_image], out)
    assert out.exists()


def test_to_pdf_empty_raises(tmp_path):
    with pytest.raises((ValueError, Exception)):
        compose.to_pdf([], tmp_path / "empty.pdf")
