"""Text generation: outline + per-page rhyming stanzas + title.

Wraps ``llama-cpp-python`` running the GGUF LLM (Qwen2.5-1.5B by default, on CPU
at 4 GB). Generation is *structured*: an 8-beat outline first, then each stanza
in sequence with prior stanzas carried in context so rhyme/meter hold. Few-shot
rhyme examples in the system prompt replace the (cut) RAG corpus.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .device import DevicePolicy

if TYPE_CHECKING:
    from config import Config

    from .questionnaire import Answers


_FEW_SHOT_SYSTEM = """\
You write rhyming children's picture-book text for ages 3-6.
Each stanza is exactly 4 lines with an AABB or ABAB rhyme scheme.

Example stanza:
The little bear walked through the trees,
She found some honey made by bees.
She shared it with her rabbit friend,
A perfect way for days to end.
"""


def _outline_messages(answers: "Answers", num_pages: int) -> list[dict]:
    user = (
        f"Write a {num_pages}-beat story outline for a children's picture book.\n"
        f"Child: {answers.child_name} ({answers.pronoun})\n"
        f"Theme: {answers.theme}\n"
        f"Setting: {answers.setting}\n"
        f"Favourite animal companion: {answers.favourite_animal}\n"
        f"Loved one who appears: {answers.loved_one}\n\n"
        f'Return JSON with a single key "beats" whose value is a list of '
        f"{num_pages} one-sentence story beats."
    )
    return [
        {"role": "system", "content": _FEW_SHOT_SYSTEM},
        {"role": "user", "content": user},
    ]


def _stanza_messages(
    beat: str,
    beat_idx: int,
    prior_stanzas: list[str],
    answers: "Answers",
    config: "Config",
) -> list[dict]:
    prior_block = ""
    if prior_stanzas:
        prior_block = "\n\nPrevious stanzas (for rhyme/flow continuity):\n" + "\n---\n".join(
            prior_stanzas
        )

    user = (
        f"Write stanza {beat_idx + 1} of {config.num_pages} "
        f"({config.lines_per_page} lines, AABB or ABAB rhyme).\n"
        f"Scene: {beat}\n"
        f"Child: {answers.child_name} ({answers.pronoun}), "
        f"{answers.hair_color} hair, {answers.skin_tone} skin.\n"
        f"Companion: {answers.favourite_animal}."
        + prior_block
    )
    return [
        {"role": "system", "content": _FEW_SHOT_SYSTEM},
        {"role": "user", "content": user},
    ]


def _title_messages(answers: "Answers", outline: list[str]) -> list[dict]:
    beats_summary = "; ".join(outline[:3])
    user = (
        f"Create a short, child-friendly title (5 words or fewer) for a picture book "
        f"about {answers.child_name}.\n"
        f"Theme: {answers.theme}. Setting: {answers.setting}.\n"
        f"Story begins: {beats_summary}\n"
        f"Return only the title, nothing else."
    )
    return [
        {"role": "system", "content": "You write titles for children's picture books."},
        {"role": "user", "content": user},
    ]


class StoryWriter:
    """Lazy wrapper around the local LLM. ``load()`` does the heavy import."""

    def __init__(self, config: "Config", device_policy: DevicePolicy) -> None:
        self.config = config
        self.device = device_policy
        self._llm = None  # llama_cpp.Llama instance, created in load()

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> None:
        """Instantiate ``llama_cpp.Llama`` from ``config.paths.llm``."""
        import llama_cpp

        gpu_layers = 0 if self.device.llm_device == "cpu" else self.config.llm_gpu_layers
        self._llm = llama_cpp.Llama(
            model_path=str(self.config.paths.llm),
            n_ctx=self.config.llm_ctx,
            n_threads=self.config.llm_threads,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )

    def unload(self) -> None:
        """Drop the model reference (free RAM)."""
        self._llm = None

    # -- generation ----------------------------------------------------------

    def generate_outline(self, answers: "Answers") -> list[str]:
        """Return ``config.num_pages`` one-line story beats."""
        messages = _outline_messages(answers, self.config.num_pages)
        resp = self._llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
        )
        content = resp["choices"][0]["message"]["content"]
        data = json.loads(content)
        beats = data.get("beats", [])
        return [str(b) for b in beats[: self.config.num_pages]]

    def generate_stanzas(self, outline: list[str], answers: "Answers") -> list[str]:
        """Return one rhyming stanza (``config.lines_per_page`` lines) per beat."""
        stanzas: list[str] = []
        for i, beat in enumerate(outline):
            messages = _stanza_messages(beat, i, stanzas, answers, self.config)
            resp = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
            stanzas.append(resp["choices"][0]["message"]["content"].strip())
        return stanzas

    def generate_title(self, answers: "Answers", outline: list[str]) -> str:
        """Return a short, child-friendly book title."""
        messages = _title_messages(answers, outline)
        resp = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=64,
            temperature=self.config.llm_temperature,
        )
        return resp["choices"][0]["message"]["content"].strip()
