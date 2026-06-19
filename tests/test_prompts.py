"""Tests for prompts.py — pure string assembly, no models."""
import dataclasses
import pytest
from pipeline import prompts


# --- character_sheet ---

def test_character_sheet_contains_pronoun_desc(valid_answers):
    sheet = prompts.character_sheet(valid_answers)
    assert "little girl" in sheet


def test_character_sheet_contains_hair_color(valid_answers):
    sheet = prompts.character_sheet(valid_answers)
    assert valid_answers.hair_color in sheet


def test_character_sheet_contains_skin_tone(valid_answers):
    sheet = prompts.character_sheet(valid_answers)
    assert valid_answers.skin_tone in sheet


def test_character_sheet_boy_pronoun(valid_answers):
    a = dataclasses.replace(valid_answers, pronoun="boy")
    assert "little boy" in prompts.character_sheet(a)


def test_character_sheet_child_pronoun(valid_answers):
    a = dataclasses.replace(valid_answers, pronoun="child")
    assert "young child" in prompts.character_sheet(a)


# --- companion_desc ---

def test_companion_desc_contains_animal(valid_answers):
    desc = prompts.companion_desc(valid_answers)
    assert valid_answers.favourite_animal in desc


def test_companion_desc_different_animals(valid_answers):
    for animal in ["cat", "unicorn", "bear"]:
        a = dataclasses.replace(valid_answers, favourite_animal=animal)
        assert animal in prompts.companion_desc(a)


# --- page_image_prompt ---

def _beat(text="the hero finds a treasure", place="a sunny meadow", activity="picking flowers"):
    from pipeline.story import Beat
    return Beat(text=text, place=place, activity=activity)


def test_page_prompt_contains_art_style(valid_answers):
    p = prompts.page_image_prompt(_beat(), valid_answers)
    assert valid_answers.art_style in p


def test_page_prompt_contains_beat_place(valid_answers):
    p = prompts.page_image_prompt(_beat(place="a busy kitchen"), valid_answers)
    assert "a busy kitchen" in p


def test_page_prompt_contains_beat_activity(valid_answers):
    """Image must show the page's activity (matches the poem)."""
    p = prompts.page_image_prompt(_beat(activity="taking a bubble bath"), valid_answers)
    assert "taking a bubble bath" in p


def test_page_prompt_contains_beat_text(valid_answers):
    p = prompts.page_image_prompt(_beat(text="the hero discovers a glowing door"), valid_answers)
    assert "the hero discovers a glowing door" in p


def test_page_prompt_contains_character_sheet_fragments(valid_answers):
    p = prompts.page_image_prompt(_beat(), valid_answers)
    assert valid_answers.hair_color in p
    assert valid_answers.skin_tone in p


def test_page_prompt_contains_companion(valid_answers):
    p = prompts.page_image_prompt(_beat(), valid_answers)
    assert valid_answers.favourite_animal in p


# --- cover_image_prompt ---

def test_cover_prompt_contains_title(valid_answers):
    title = "Luna and the Dragon"
    p = prompts.cover_image_prompt(title, valid_answers)
    assert title in p


def test_cover_prompt_contains_art_style(valid_answers):
    p = prompts.cover_image_prompt("My Book", valid_answers)
    assert valid_answers.art_style in p


def test_cover_prompt_contains_setting(valid_answers):
    p = prompts.cover_image_prompt("My Book", valid_answers)
    assert valid_answers.setting in p


# --- custom (typed) values must not crash and must flow into the prompt ---

def test_custom_pronoun_and_art_style_do_not_crash(valid_answers):
    import dataclasses
    a = dataclasses.replace(valid_answers, pronoun="kiddo", art_style="neon comic")
    sheet = prompts.character_sheet(a)
    page = prompts.page_image_prompt(_beat(), a)
    assert "kiddo" in sheet
    assert "neon comic" in page
