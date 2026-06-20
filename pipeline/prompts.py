"""Prompt construction: questionnaire -> character sheet + per-page image prompts.

Pure string assembly, no models. Kept separate from ``images.py`` so prompt
wording can be iterated without touching the diffusers code.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .questionnaire import Answers
    from .story import Beat

# Shared negative prompt for every image (kid-safe + quality).
NEGATIVE_PROMPT = (
    "blurry, deformed, extra limbs, extra fingers, bad anatomy, text, watermark, "
    "signature, low quality, scary, creepy, nsfw, photo, realistic"
)


def character_sheet(answers: "Answers") -> str:
    """Frozen visual description of the hero, reused on every page.

    The hero can be a human child OR any creature (``character_type``: dinosaur,
    alien, robot, cat, …). Humans get the pronoun + hair + skin description; a
    non-human is named as that creature with a single colour cue — "hair"/"skin"
    phrasing is dropped because it makes no sense for a dinosaur or robot.
    """
    from .questionnaire import HUMAN_CHARACTER_TYPES, PRONOUN_DESC

    ctype = (answers.character_type or "child").strip()
    if ctype.lower() in HUMAN_CHARACTER_TYPES:
        # Custom typed pronouns fall back to "a little <value>".
        desc = PRONOUN_DESC.get(answers.pronoun, f"a little {answers.pronoun}")
        return f"{desc} with {answers.hair_color} hair and {answers.skin_tone} skin, big friendly eyes"

    # Non-human hero: the creature is the subject; skin_tone carries its colour.
    return (
        f"a cute friendly {ctype} character, {answers.skin_tone} colored, "
        f"big friendly eyes, adorable storybook character"
    )


def companion_desc(answers: "Answers") -> str:
    """Soft-consistent description of the favourite-animal companion."""
    return f"a friendly little {answers.favourite_animal} companion"


def page_image_prompt(beat: "Beat", answers: "Answers") -> str:
    """Full positive prompt for one page's illustration.

    Built from the beat's own place + activity (so every page differs and the
    image matches the poem). Hero and companion are listed plainly together —
    the framing that reliably renders both at SD 1.5 / 4 GB (see the scene
    prototype). Crowds of 4+ characters are not reliable at this model size.
    """
    from .questionnaire import ART_STYLE_FRAGMENT
    style = ART_STYLE_FRAGMENT.get(answers.art_style, answers.art_style)
    sheet = character_sheet(answers)
    companion = companion_desc(answers)
    return (
        f"{style}, in {beat.place}: {sheet} {beat.activity}, "
        f"together with {companion}, {beat.text}, "
        f"full lively scene, children's book illustration"
    )


def cover_image_prompt(title: str, answers: "Answers") -> str:
    """Prompt for the cover hero shot — child front-and-center, title framing."""
    from .questionnaire import ART_STYLE_FRAGMENT
    style = ART_STYLE_FRAGMENT.get(answers.art_style, answers.art_style)
    sheet = character_sheet(answers)
    return (
        f"{style}, {sheet}, hero portrait, front and center, "
        f"title page of a children's book called '{title}', "
        f"in a {answers.setting}, warm and inviting"
    )
