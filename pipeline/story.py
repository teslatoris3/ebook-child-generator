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

_FEW_SHOT = """\
Example rhyming stanza (AABB):
  The forest was quiet, the moon shining bright,
  Emma and Dragon soared up through the night.
  They twirled past the stars with a laugh and a glow,
  And painted the darkness with colors below.

Example rhyming stanza (ABAB):
  Mia found a tiny shell beside the sea,
  Its colors shimmered, pink and blue and gold.
  "Come with me," the little crab said with glee,
  And off they swam into a world untold.
"""


class StoryWriter:
    """Lazy wrapper around the local LLM. ``load()`` does the heavy import."""

    def __init__(self, config: "Config", device_policy: DevicePolicy) -> None:
        self.config = config
        self.device = device_policy
        self._llm = None  # llama_cpp.Llama instance, created in load()

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> None:
        """Instantiate ``llama_cpp.Llama`` from the configured GGUF path."""
        import llama_cpp

        llm_path = self.config.paths.llm
        if not llm_path.exists():
            llm_path = self.config.paths.llm_fallback
            if not llm_path.exists():
                raise FileNotFoundError(
                    f"No LLM GGUF found at {self.config.paths.llm} "
                    f"or {self.config.paths.llm_fallback}"
                )

        n_gpu_layers = self.config.llm_gpu_layers if self.device.llm_device == "cuda" else 0
        kwargs: dict = dict(
            model_path=str(llm_path),
            n_ctx=self.config.llm_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        if self.config.llm_threads:
            kwargs["n_threads"] = self.config.llm_threads

        self._llm = llama_cpp.Llama(**kwargs)

    def unload(self) -> None:
        """Drop the model reference (free RAM)."""
        self._llm = None

    # -- generation ----------------------------------------------------------

    def generate_outline(self, answers: "Answers") -> list[str]:
        """Return ``config.num_pages`` one-line story beats as a list of strings."""
        from .questionnaire import PRONOUN_SUBJECT

        assert self._llm is not None, "Call load() first"
        pronoun = PRONOUN_SUBJECT[answers.pronoun]
        loved_one_line = (
            f"Loved one who appears: {answers.loved_one}"
            if answers.loved_one != "none"
            else ""
        )
        system = (
            "You are a creative children's book author writing for ages 3-6.\n"
            f"Output ONLY valid JSON: {{\"beats\": [\"beat 1\", ..., \"beat {self.config.num_pages}\"]}}\n"
            "Each beat is one short sentence describing the scene on that page."
        )
        user = (
            f"Write exactly {self.config.num_pages} story beats for a children's book.\n"
            f"Hero: {answers.child_name} ({pronoun})\n"
            f"Companion: a {answers.favourite_animal}\n"
            f"Theme: {answers.theme}\n"
            f"Setting: {answers.setting}\n"
            f"{loved_one_line}\n"
            "Keep each beat to one vivid sentence. Output JSON only."
        )
        resp = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=self.config.llm_max_tokens,
            temperature=0.7,
        )
        text = resp["choices"][0]["message"]["content"]
        try:
            beats: list[str] = json.loads(text).get("beats", [])
        except (json.JSONDecodeError, AttributeError):
            beats = []

        # Pad / trim to exactly num_pages beats
        while len(beats) < self.config.num_pages:
            beats.append(f"The adventure continues in the {answers.setting}.")
        return beats[: self.config.num_pages]

    def generate_stanzas(self, outline: list[str], answers: "Answers") -> list[str]:
        """Return one rhyming stanza (``config.lines_per_page`` lines) per beat."""
        from .questionnaire import PRONOUN_SUBJECT

        assert self._llm is not None, "Call load() first"
        pronoun = PRONOUN_SUBJECT[answers.pronoun]
        system = (
            "You write rhyming stanzas for a children's picture book (ages 3-6).\n"
            f"Write exactly {self.config.lines_per_page} lines that rhyme (AABB or ABAB).\n"
            "Keep language simple, joyful, and age-appropriate.\n\n"
            + _FEW_SHOT
        )
        stanzas: list[str] = []
        for i, beat in enumerate(outline):
            prior = ""
            if stanzas:
                recent = stanzas[-2:]
                prior = "\n\nPrevious stanzas for rhyme continuity:\n" + "\n\n".join(recent)
            user = (
                f"Write a {self.config.lines_per_page}-line rhyming stanza for page {i + 1}.\n"
                f"Child: {answers.child_name} ({pronoun})\n"
                f"Scene: {beat}\n"
                f"Companion: a {answers.favourite_animal}\n"
                f"{prior}\n\n"
                f"Output exactly {self.config.lines_per_page} rhyming lines. Nothing else."
            )
            resp = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=150,
                temperature=self.config.llm_temperature,
            )
            stanzas.append(resp["choices"][0]["message"]["content"].strip())
        return stanzas

    def generate_title(self, answers: "Answers", outline: list[str]) -> str:
        """Return a short, child-friendly book title."""
        assert self._llm is not None, "Call load() first"
        first_beat = outline[0] if outline else ""
        system = (
            "You write short, catchy children's book titles (ages 3-6). "
            "Output only the title, no quotes, no explanation."
        )
        user = (
            f"Write a short magical title (max 7 words) for a story about "
            f"{answers.child_name} and a {answers.favourite_animal} "
            f"in a {answers.setting}. Theme: {answers.theme}. "
            f"First scene: {first_beat}. Output only the title."
        )
        resp = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=30,
            temperature=0.9,
        )
        return resp["choices"][0]["message"]["content"].strip().strip('"\'')
