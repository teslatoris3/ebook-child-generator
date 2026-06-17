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
    """Frozen visual description of the hero child, reused on every page.

    e.g. "a little girl with blonde hair and light skin, big friendly eyes".
    TODO: build from PRONOUN_DESC[pronoun] + hair_color + skin_tone.
    """
    raise NotImplementedError


def companion_desc(answers: "Answers") -> str:
    """Soft-consistent description of the favourite-animal companion.

    TODO: short stable phrase, e.g. "a friendly little dragon companion".
    """
    raise NotImplementedError


def page_image_prompt(beat: str, answers: "Answers") -> str:
    """Compose the full positive prompt for one page's illustration.

    Layout: ``<art style>, <character sheet>, <companion>, <scene from beat>,
    in a <setting>, children's book illustration``.
    TODO: assemble using character_sheet/companion_desc + ART_STYLE_FRAGMENT.
    """
    raise NotImplementedError


def cover_image_prompt(title: str, answers: "Answers") -> str:
    """Prompt for the cover hero shot. TODO: hero front-and-center, titley framing."""
    raise NotImplementedError
