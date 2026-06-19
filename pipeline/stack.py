"""The ModelStack seam: a load/unload lifecycle the heavy stacks share.

StoryWriter (LLM, CPU) and ImageStudio (SD+IP-Adapter, GPU) are the two adapters
of this seam. Naming it lets the orchestrator depend on the contract rather than
the concrete stacks, so tests inject a fake stack with no GPU/model.

``exclusive`` is the single home for the central 4 GB invariant: never keep two
stacks resident at once. See CONTEXT.md ("Model Stack").
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class ModelStack(Protocol):
    """Anything with the heavy-model lifecycle: load it, use it, unload it."""

    def load(self) -> None: ...

    def unload(self) -> None: ...


@contextmanager
def exclusive(stack: ModelStack) -> Iterator[ModelStack]:
    """Load ``stack`` for the duration of the block, then always unload it.

    Wrapping each pipeline phase in ``exclusive`` guarantees the previous stack
    is unloaded (freeing the device) before the next one loads — the "never two
    stacks resident on the 4 GB GPU at once" rule, enforced in one place.
    """
    stack.load()
    try:
        yield stack
    finally:
        stack.unload()
