"""Text generation: outline + per-page rhyming stanzas + title.

Wraps ``llama-cpp-python`` running the GGUF LLM (Qwen2.5-1.5B by default, on CPU
at 4 GB). Generation is *structured*: an 8-beat outline first, then each stanza
in sequence with prior stanzas carried in context so rhyme/meter hold. Few-shot
rhyme examples in the system prompt replace the (cut) RAG corpus.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .device import DevicePolicy

if TYPE_CHECKING:
    from config import Config

    from .questionnaire import Answers


class StoryWriter:
    """Lazy wrapper around the local LLM. ``load()`` does the heavy import."""

    def __init__(self, config: "Config", device_policy: DevicePolicy) -> None:
        self.config = config
        self.device = device_policy
        self._llm = None  # llama_cpp.Llama instance, created in load()

    # -- lifecycle -----------------------------------------------------------
    def load(self) -> None:
        """Instantiate ``llama_cpp.Llama`` from ``config.paths.llm``.

        TODO: import llama_cpp here; set n_gpu_layers from device policy
        (0 on cpu), n_ctx=config.llm_ctx, n_threads=config.llm_threads.
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Drop the model reference (free RAM)."""
        self._llm = None

    # -- generation ----------------------------------------------------------
    def generate_outline(self, answers: "Answers") -> list[str]:
        """Return ``config.num_pages`` one-line story beats.

        TODO: prompt the LLM (JSON-constrained) for an N-beat arc tied to
        theme/setting/cast; parse into a list of strings.
        """
        raise NotImplementedError

    def generate_stanzas(self, outline: list[str], answers: "Answers") -> list[str]:
        """Return one rhyming stanza (``config.lines_per_page`` lines) per beat.

        TODO: iterate beats, generating each stanza with previous stanzas in
        context for rhyme/flow continuity.
        """
        raise NotImplementedError

    def generate_title(self, answers: "Answers", outline: list[str]) -> str:
        """Return a short, child-friendly book title. TODO."""
        raise NotImplementedError
