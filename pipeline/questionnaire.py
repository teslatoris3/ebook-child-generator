"""The 9-field personalization questionnaire.

This module is pure *data*: the dropdown catalogs, the value->prompt-fragment
maps, and the ``Answers`` dataclass that the rest of the pipeline consumes.
The logic that *uses* these fragments to build prompts lives in ``prompts.py``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# --- Dropdown catalogs (anything affecting the art is constrained) ----------
PRONOUNS = ["girl", "boy", "child"]
HAIR_COLORS = ["black", "brown", "blonde", "red", "dark curly"]
SKIN_TONES = ["light", "medium", "tan", "dark"]
ANIMALS = ["cat", "dog", "dragon", "rabbit", "lion", "dinosaur", "unicorn", "bear"]
LOVED_ONES = ["Mom", "Dad", "Grandma", "Grandpa", "big sister", "big brother", "none"]
THEMES = [
    "making friends",
    "being brave",
    "bedtime",
    "sharing",
    "kindness",
    "trying new things",
]
SETTINGS = [
    "enchanted forest",
    "ocean",
    "outer space",
    "cozy home",
    "magical castle",
    "jungle",
]
ART_STYLES = [
    "watercolor children's book",
    "soft pastel cartoon",
    "crayon doodle",
    "classic storybook illustration",
]

# --- Value -> prompt-fragment maps (consumed by prompts.py) -----------------
# How the hero is described visually for the image prompts.
PRONOUN_DESC = {"girl": "a little girl", "boy": "a little boy", "child": "a young child"}
# Grammatical subject pronoun for the story text.
PRONOUN_SUBJECT = {"girl": "she", "boy": "he", "child": "they"}
PRONOUN_OBJECT = {"girl": "her", "boy": "him", "child": "them"}
# Art-style values map 1:1 to prompt fragments today, but the indirection lets us
# tune each independently later (e.g. add trigger words / quality boosters).
ART_STYLE_FRAGMENT = {s: s for s in ART_STYLES}


@dataclass
class Answers:
    """One filled-in questionnaire -> one book."""

    child_name: str
    pronoun: str
    hair_color: str
    skin_tone: str
    favourite_animal: str
    loved_one: str
    theme: str
    setting: str
    art_style: str

    _UNSAFE_FS_CHARS = frozenset(r'\/:*?"<>|')

    def validate(self) -> None:
        """Raise ``ValueError`` if any dropdown value is out of its catalog."""
        if not self.child_name or not self.child_name.strip():
            raise ValueError("child_name must not be empty")
        if any(c in self._UNSAFE_FS_CHARS for c in self.child_name):
            raise ValueError(f"child_name contains filesystem-unsafe character: {self.child_name!r}")

        catalog_checks = [
            ("pronoun", self.pronoun, PRONOUNS),
            ("hair_color", self.hair_color, HAIR_COLORS),
            ("skin_tone", self.skin_tone, SKIN_TONES),
            ("favourite_animal", self.favourite_animal, ANIMALS),
            ("loved_one", self.loved_one, LOVED_ONES),
            ("theme", self.theme, THEMES),
            ("setting", self.setting, SETTINGS),
            ("art_style", self.art_style, ART_STYLES),
        ]
        for field, value, catalog in catalog_checks:
            if value not in catalog:
                raise ValueError(f"{field}={value!r} is not in the allowed catalog")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
