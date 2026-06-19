"""Tests for pipeline/stack.py — the ModelStack seam + exclusive() invariant."""
import pytest

from pipeline.stack import ModelStack, exclusive


class FakeStack:
    """A ModelStack adapter for tests: records load/unload calls, no GPU."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.loaded = False

    def load(self) -> None:
        self.events.append("load")
        self.loaded = True

    def unload(self) -> None:
        self.events.append("unload")
        self.loaded = False


# --- exclusive: load on enter, unload on exit ---

def test_exclusive_loads_then_unloads():
    stack = FakeStack()
    with exclusive(stack):
        assert stack.loaded is True
    assert stack.loaded is False
    assert stack.events == ["load", "unload"]


def test_exclusive_unloads_on_exception():
    """The 4 GB invariant must hold even when the block fails mid-phase."""
    stack = FakeStack()
    with pytest.raises(RuntimeError):
        with exclusive(stack):
            raise RuntimeError("boom")
    assert stack.loaded is False
    assert stack.events == ["load", "unload"]


def test_exclusive_yields_the_stack():
    stack = FakeStack()
    with exclusive(stack) as s:
        assert s is stack


# --- the two real adapters satisfy the seam (two adapters = a real seam) ---

def test_story_writer_is_a_model_stack():
    from pipeline.story import StoryWriter
    assert isinstance(StoryWriter, type)
    # runtime_checkable Protocol: instances structurally conform.
    w = StoryWriter.__new__(StoryWriter)
    assert isinstance(w, ModelStack)


def test_image_studio_is_a_model_stack():
    from pipeline.images import ImageStudio
    s = ImageStudio.__new__(ImageStudio)
    assert isinstance(s, ModelStack)
