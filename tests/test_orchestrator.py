"""Tests for pipeline/orchestrator.py — full pipeline wiring with fake stacks.

The two heavy stacks are injected as fakes through the ModelStack seam, so the
orchestration logic (stage order, BookRecord building, compose, PDF, cleanup)
runs end-to-end on CPU with no GPU/models. compose + record are exercised for real.
"""
from pathlib import Path

import pytest
from PIL import Image as PILImage

from config import Config
from pipeline.device import DevicePolicy
from pipeline.orchestrator import BookGenerator, BookResult, Progress
from pipeline.record import BookRecord


# --- Fakes (ModelStack adapters) ---

class FakeWriter:
    """Stand-in StoryWriter: ModelStack lifecycle + deterministic text."""

    def __init__(self, config, device, num_pages=8):
        self.config = config
        self.events = []
        self._num_pages = num_pages

    def load(self):
        self.events.append("load")

    def unload(self):
        self.events.append("unload")

    def generate_outline(self, answers):
        from pipeline.story import Beat
        return [
            Beat(text=f"beat {i}", place=f"place {i}", activity=f"activity {i}")
            for i in range(self._num_pages)
        ]

    def generate_stanzas(self, outline, answers):
        return [f"stanza about {b.activity}\nline two\nline three\nline four" for b in outline]

    def generate_title(self, answers, outline):
        return f"{answers.child_name}'s Adventure"


class FakeStudio:
    """Stand-in ImageStudio: ModelStack lifecycle + tiny solid images."""

    def __init__(self, config, device):
        self.config = config
        self.events = []
        self.loaded = False

    def load(self):
        self.events.append("load")
        self.loaded = True

    def unload(self):
        self.events.append("unload")
        self.loaded = False

    def make_reference(self, answers, seed):
        return PILImage.new("RGB", (64, 64), (200, 180, 160))

    def make_page(self, prompt, reference, seed, ip_scale=None):
        # vary colour by seed so pages are distinguishable
        return PILImage.new("RGB", (64, 64), (seed % 255, 120, 120))


def _policy():
    return DevicePolicy(llm_device="cpu", sd_device="cuda", vram_gb=4.0, reason="test")


@pytest.fixture
def gen(tmp_path):
    """A BookGenerator with fake stacks and output redirected to tmp_path."""
    cfg = Config()
    cfg.output_dir = tmp_path
    g = BookGenerator(cfg, _policy())
    g.writer = FakeWriter(cfg, g.device, num_pages=cfg.num_pages)
    g.studio = FakeStudio(cfg, g.device)
    return g


def _run(gen, answers):
    """Drain generate(); return (progress_events, final_result)."""
    events = list(gen.generate(answers))
    result = events[-1]
    progress = events[:-1]
    return progress, result


# --- tracer bullet ---

def test_generate_yields_book_result(gen, valid_answers):
    _, result = _run(gen, valid_answers)
    assert isinstance(result, BookResult)


# --- artifacts on disk ---

def test_generate_writes_all_page_images(gen, valid_answers):
    _, result = _run(gen, valid_answers)
    assert len(result.page_image_paths) == gen.config.num_pages
    assert all(p.exists() for p in result.page_image_paths)


def test_generate_writes_reference(gen, valid_answers):
    _, result = _run(gen, valid_answers)
    assert result.reference_path.exists()


def test_generate_writes_pdf(gen, valid_answers):
    _, result = _run(gen, valid_answers)
    assert result.pdf_path.exists()
    assert result.pdf_path.read_bytes()[:4] == b"%PDF"


def test_generate_writes_book_json(gen, valid_answers):
    _, result = _run(gen, valid_answers)
    record = BookRecord.load(result.output_dir)
    assert record.title == result.title
    assert len(record.pages) == gen.config.num_pages
    assert all(p.prompt and p.stanza for p in record.pages)


def test_generate_output_dir_includes_child_name(gen, valid_answers):
    _, result = _run(gen, valid_answers)
    assert valid_answers.child_name in result.output_dir.name


# --- progress stream ---

def test_generate_emits_progress(gen, valid_answers):
    progress, _ = _run(gen, valid_answers)
    assert all(isinstance(p, Progress) for p in progress)
    stages = {p.stage for p in progress}
    assert {"story", "reference", "page", "compose"} <= stages


def test_generate_emits_one_progress_per_page(gen, valid_answers):
    progress, _ = _run(gen, valid_answers)
    page_ticks = [p for p in progress if p.stage == "page"]
    assert len(page_ticks) == gen.config.num_pages


# --- the exclusive() invariant ---

def test_generate_unloads_both_stacks(gen, valid_answers):
    _run(gen, valid_answers)
    assert gen.writer.events == ["load", "unload"]
    assert gen.studio.events == ["load", "unload"]


def test_generate_unloads_writer_before_loading_studio(gen, valid_answers):
    """The 4 GB invariant: text stack gone before the image stack loads."""
    order = []
    orig_w_unload = gen.writer.unload
    orig_s_load = gen.studio.load

    def w_unload():
        order.append("writer_unload")
        orig_w_unload()

    def s_load():
        order.append("studio_load")
        orig_s_load()

    gen.writer.unload = w_unload
    gen.studio.load = s_load
    _run(gen, valid_answers)
    assert order.index("writer_unload") < order.index("studio_load")


def test_generate_validates_answers(gen, valid_answers):
    import dataclasses
    bad = dataclasses.replace(valid_answers, child_name="")  # empty name is invalid
    with pytest.raises(ValueError):
        list(gen.generate(bad))


# --- regenerate_page (reroll) ---

@pytest.fixture
def generated(gen, valid_answers):
    """Run a full generation; return (gen, output_dir)."""
    _, result = _run(gen, valid_answers)
    # reset stack event logs so regen assertions start clean
    gen.writer.events.clear()
    gen.studio.events.clear()
    return gen, result.output_dir


def test_regenerate_page_returns_existing_path(generated):
    gen, out_dir = generated
    before = BookRecord.load(out_dir).pages[2].image_path
    path = gen.regenerate_page(out_dir, 2)
    assert path == out_dir / before
    assert path.exists()


def test_regenerate_page_default_bumps_seed(generated):
    gen, out_dir = generated
    old_seed = BookRecord.load(out_dir).pages[2].seed
    gen.regenerate_page(out_dir, 2)
    assert BookRecord.load(out_dir).pages[2].seed == old_seed + 1


def test_regenerate_page_explicit_seed_persisted(generated):
    gen, out_dir = generated
    gen.regenerate_page(out_dir, 1, new_seed=4321)
    assert BookRecord.load(out_dir).pages[1].seed == 4321


def test_regenerate_page_rebuilds_pdf(generated):
    """The downloadable PDF should reflect the rerolled page."""
    gen, out_dir = generated
    pdf = out_dir / "book.pdf"
    before_mtime = pdf.stat().st_mtime_ns
    gen.regenerate_page(out_dir, 0, new_seed=777)
    assert pdf.exists()
    assert pdf.read_bytes()[:4] == b"%PDF"
    assert pdf.stat().st_mtime_ns != before_mtime


def test_regenerate_page_uses_only_studio(generated):
    """Reroll re-renders an image; it must not touch the LLM stack."""
    gen, out_dir = generated
    gen.regenerate_page(out_dir, 0)
    assert gen.studio.events == ["load", "unload"]
    assert gen.writer.events == []
