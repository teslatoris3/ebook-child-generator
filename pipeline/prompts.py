"""Prompt construction: questionnaire -> character sheet + per-page image prompts.

Pure string assembly, no models. Kept separate from ``images.py`` so prompt
wording can be iterated without touching the diffusers code.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .questionnaire import Answers

# Shared negative prompt for every image (kid-safe + quality).
NEGATIVE_PROMPT = (
    "blurry, deformed, extra limbs, extra fingers, bad anatomy, text, watermark, "
    "signature, low quality, scary, creepy, nsfw, photo, realistic"
)


def character_sheet(answers: "Answers") -> str:
    """Frozen visual description of the hero child, reused on every page."""
    from .questionnaire import PRONOUN_DESC
    desc = PRONOUN_DESC[answers.pronoun]
    return (
        f"{desc} with {answers.hair_color} hair and {answers.skin_tone} skin, "
        "big friendly eyes, cheerful expression"
    )


def companion_desc(answers: "Answers") -> str:
    """Soft-consistent description of the favourite-animal companion."""
    return f"a friendly little {answers.favourite_animal} companion"


def page_image_prompt(beat: str, answers: "Answers") -> str:
    """Full positive prompt for one page's illustration."""
    from .questionnaire import ART_STYLE_FRAGMENT
    style = ART_STYLE_FRAGMENT[answers.art_style]
    hero = character_sheet(answers)
    companion = companion_desc(answers)
    return (
        f"{style}, {hero}, {companion}, {beat}, "
        f"in a {answers.setting}, children's book illustration, "
        "soft lighting, vibrant colors, whimsical"
    )


def cover_image_prompt(title: str, answers: "Answers") -> str:
    """Prompt for the cover hero shot — child front-and-center."""
    from .questionnaire import ART_STYLE_FRAGMENT
    style = ART_STYLE_FRAGMENT[answers.art_style]
    hero = character_sheet(answers)
    return (
        f"{style}, {hero}, centered composition, front-facing, "
        f"title page illustration, magical {answers.setting}, "
        "children's book cover art, vibrant, welcoming"
    )
