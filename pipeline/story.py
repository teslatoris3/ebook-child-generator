"""Text generation: outline + per-page rhyming stanzas + title.

Wraps ``llama-cpp-python`` running the GGUF LLM (Qwen2.5-1.5B by default, on CPU
at 4 GB). Generation is *structured*: an 8-beat outline first, then each stanza
in sequence with prior stanzas carried in context so rhyme/meter hold. Few-shot
rhyme examples in the system prompt replace the (cut) RAG corpus.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .device import DevicePolicy

if TYPE_CHECKING:
    from config import Config

    from .questionnaire import Answers


@dataclass
class Beat:
    """One story beat with its own place and activity (so pages vary).

    ``activity`` is fed to both the stanza (poem matches the activity) and the
    image prompt (image shows the activity) — keeping poem and picture in sync.
    """

    text: str
    place: str
    activity: str


def _hero_desc(answers: "Answers") -> str:
    """One-line hero description for story prompts: human child or any creature."""
    from .questionnaire import HUMAN_CHARACTER_TYPES

    ctype = (answers.character_type or "child").strip()
    if ctype.lower() in HUMAN_CHARACTER_TYPES:
        return (
            f"{answers.child_name} ({answers.pronoun}), "
            f"{answers.hair_color} hair, {answers.skin_tone} skin"
        )
    return f"{answers.child_name}, a {answers.pronoun} {ctype}"


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
    activities_hint = (
        f"Favourite activities to draw from (use as inspiration, invent more so "
        f"every page is different): {answers.favourite_activities}\n"
        if answers.favourite_activities.strip()
        else ""
    )
    user = (
        f"Plan a {num_pages}-beat story for a children's picture book.\n"
        f"Hero: {_hero_desc(answers)}\n"
        f"Theme: {answers.theme}\n"
        f"Overall setting flavour: {answers.setting}\n"
        f"Favourite animal companion: {answers.favourite_animal}\n"
        f"Loved one who appears: {answers.loved_one}\n"
        f"{activities_hint}\n"
        f"Make every page DIFFERENT: each beat happens in a DIFFERENT place and "
        f"shows the child doing a DIFFERENT activity.\n"
        f'Return JSON with a single key "beats": a list of {num_pages} objects, '
        f'each with keys "beat" (one sentence describing what happens), '
        f'"place" (where it happens), and "activity" (what the child is doing, '
        f"e.g. cooking, bathing, painting)."
    )
    return [
        {"role": "system", "content": _FEW_SHOT_SYSTEM},
        {"role": "user", "content": user},
    ]


def _coerce_beat(raw, answers: "Answers", idx: int) -> "Beat":
    """Build a Beat from a (possibly messy) LLM beat object.

    Tolerates a plain string or a dict missing keys: place falls back to the
    overall setting, activity to a favourite activity (or a generic verb).
    """
    if isinstance(raw, dict):
        text = str(raw.get("beat") or raw.get("text") or "").strip()
        place = str(raw.get("place") or "").strip()
        activity = str(raw.get("activity") or "").strip()
    else:
        text, place, activity = str(raw).strip(), "", ""

    if not place:
        place = answers.setting
    if not activity:
        hints = [a.strip() for a in answers.favourite_activities.split(",") if a.strip()]
        activity = hints[idx % len(hints)] if hints else "playing"
    if not text:
        text = f"{answers.child_name} enjoys {activity} in {place}"
    return Beat(text=text, place=place, activity=activity)


def _stanza_messages(
    beat: "Beat",
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
        f"Scene: {beat.text}\n"
        f"Place: {beat.place}\n"
        f"The poem MUST be about this activity: {beat.activity}\n"
        f"Hero: {_hero_desc(answers)}.\n"
        f"Companion: {answers.favourite_animal}."
        + prior_block
    )
    return [
        {"role": "system", "content": _FEW_SHOT_SYSTEM},
        {"role": "user", "content": user},
    ]


def _title_messages(answers: "Answers", outline: list["Beat"]) -> list[dict]:
    beats_summary = "; ".join(b.text for b in outline[:3])
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


# --- Free-text extraction -------------------------------------------------------
# In-prompt few-shot examples replace a vector-DB RAG pipeline: for the small
# number of book-description patterns three examples give equivalent quality at
# zero infrastructure cost.  If the example library grows past ~20 entries,
# replace _FEW_SHOT_EXTRACTION with a FAISS index over stored examples
# (the embed/ directory is reserved for this) and retrieve the two closest ones.

_FEW_SHOT_EXTRACTION = """\
Example 1 —
Input: "Make a book about a little girl named Luna with blonde hair and light skin \
who loves cooking and dancing. Include her mom. Theme: being brave. Setting: enchanted forest."
Output: {"child_name":"Luna","character_type":"child","pronoun":"she/her",\
"hair_color":"blonde","skin_tone":"light","favourite_animal":"rabbit","loved_one":"Mom",\
"theme":"being brave","setting":"enchanted forest","art_style":"cartoon children's book",\
"favourite_activities":"cooking, dancing"}

Example 2 —
Input: "Rex is a green baby dinosaur who loves cooking in the jungle. \
His dad comes along. Art: watercolor."
Output: {"child_name":"Rex","character_type":"dinosaur","pronoun":"he/him",\
"hair_color":"green","skin_tone":"green","favourite_animal":"parrot","loved_one":"Dad",\
"theme":"adventure","setting":"jungle","art_style":"watercolor children's book",\
"favourite_activities":"cooking"}

Example 3 —
Input: "Story about Zara, an alien with purple skin who enjoys painting and reading. \
She explores outer space. Her grandma is in the story."
Output: {"child_name":"Zara","character_type":"alien","pronoun":"she/her",\
"hair_color":"purple","skin_tone":"purple","favourite_animal":"space cat",\
"loved_one":"Grandma","theme":"curiosity","setting":"outer space",\
"art_style":"cartoon children's book","favourite_activities":"painting, reading"}
"""

_EXTRACTION_SCHEMA = (
    "child_name, character_type (child / any creature), pronoun (she/her | he/him | they/them), "
    "hair_color (or body colour for non-humans), skin_tone (or body colour), "
    "favourite_animal (companion), loved_one (family member or empty string), "
    "theme (moral / lesson), setting (world / overall place), art_style, "
    "favourite_activities (comma-separated list of activities)"
)


def _extraction_messages(text: str) -> list[dict]:
    """Chat messages that instruct the LLM to parse *text* into book fields."""
    user = (
        f"Extract children's book details from the description below.\n"
        f"Return valid JSON with exactly these keys: {_EXTRACTION_SCHEMA}.\n\n"
        f"Few-shot examples:\n{_FEW_SHOT_EXTRACTION}\n"
        f"Description: {text}\n\n"
        f"Return only the JSON object. Use sensible defaults for omitted fields."
    )
    return [
        {
            "role": "system",
            "content": (
                "You extract structured children's book details from free-text descriptions. "
                "Output only valid JSON, nothing else."
            ),
        },
        {"role": "user", "content": user},
    ]


def _extract_answers_dict(text: str, llm) -> dict:
    """Call *llm* to parse *text* → Answers-compatible dict.

    Returns the raw parsed dict; callers merge with ``ANSWER_DEFAULTS`` from
    ``questionnaire`` to fill any keys the LLM omitted.
    """
    import json

    messages = _extraction_messages(text)
    resp = llm.create_chat_completion(
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=512,
        temperature=0.1,
    )
    content = resp["choices"][0]["message"]["content"]
    return json.loads(content)


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

    def generate_outline(self, answers: "Answers") -> list["Beat"]:
        """Return ``config.num_pages`` beats, each with its own place + activity."""
        messages = _outline_messages(answers, self.config.num_pages)
        resp = self._llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
        )
        content = resp["choices"][0]["message"]["content"]
        data = json.loads(content)
        raw = data.get("beats", [])[: self.config.num_pages]
        return [_coerce_beat(b, answers, i) for i, b in enumerate(raw)]

    def generate_stanzas(self, outline: list["Beat"], answers: "Answers") -> list[str]:
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

    def extract_from_prompt(self, text: str) -> dict:
        """Parse a free-text book description → Answers-compatible dict."""
        return _extract_answers_dict(text, self._llm)
