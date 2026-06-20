"""The 9-field personalization questionnaire.

This module is pure *data*: the dropdown catalogs, the value->prompt-fragment
maps, and the ``Answers`` dataclass that the rest of the pipeline consumes.
The logic that *uses* these fragments to build prompts lives in ``prompts.py``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# --- Dropdown catalogs (anything affecting the art is constrained) ----------
# What the hero IS. "child" keeps the human hair+skin description; anything else
# is drawn as that creature (custom values welcome — type "phoenix" if you like).
CHARACTER_TYPES = [
    "child", "cat", "dog", "dragon", "dinosaur", "bunny", "bear",
    "lion", "fox", "unicorn", "robot", "alien", "monster",
]
# Types treated as a human child (use pronoun + hair + skin). Everything else is
# described as a non-human creature.
HUMAN_CHARACTER_TYPES = frozenset({"child", "girl", "boy", "human", "kid", "baby", "toddler"})
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
# Suggested kid activities (hints for the free-text favourite_activities field).
ACTIVITIES = [
    "cooking", "playing", "bathing", "reading", "painting",
    "gardening", "dancing", "building blocks", "watching cartoons", "exercising",
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
    favourite_activities: str = ""  # free text, e.g. "cooking, bathing, brushing"
    character_type: str = "child"   # the hero: child, dinosaur, alien, robot, …

    _UNSAFE_FS_CHARS = frozenset(r'\/:*?"<>|')
    # Fields offered as dropdowns but which also accept a typed custom value;
    # they must be non-empty but need NOT be in their catalog.
    _REQUIRED_TEXT_FIELDS = (
        "pronoun", "hair_color", "skin_tone", "favourite_animal",
        "loved_one", "theme", "setting", "art_style", "character_type",
    )

    def validate(self) -> None:
        """Validate inputs. Catalog values are *suggestions*: any non-empty typed
        value is accepted (the UI allows custom entries). Only the child name is
        strictly guarded (non-empty + filesystem-safe). ``favourite_activities``
        is optional free text.
        """
        if not self.child_name or not self.child_name.strip():
            raise ValueError("child_name must not be empty")
        if any(c in self._UNSAFE_FS_CHARS for c in self.child_name):
            raise ValueError(f"child_name contains filesystem-unsafe character: {self.child_name!r}")

        for field in self._REQUIRED_TEXT_FIELDS:
            value = getattr(self, field)
            if not value or not str(value).strip():
                raise ValueError(f"{field} must not be empty")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
