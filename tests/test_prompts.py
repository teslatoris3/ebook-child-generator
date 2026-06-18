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

def test_page_prompt_contains_art_style(valid_answers):
    p = prompts.page_image_prompt("the hero finds a treasure", valid_answers)
    assert valid_answers.art_style in p


def test_page_prompt_contains_setting(valid_answers):
    p = prompts.page_image_prompt("the hero finds a treasure", valid_answers)
    assert valid_answers.setting in p


def test_page_prompt_contains_beat(valid_answers):
    beat = "the hero discovers a glowing door"
    p = prompts.page_image_prompt(beat, valid_answers)
    assert beat in p


def test_page_prompt_contains_character_sheet_fragments(valid_answers):
    p = prompts.page_image_prompt("a scene", valid_answers)
    assert valid_answers.hair_color in p
    assert valid_answers.skin_tone in p


def test_page_prompt_contains_companion(valid_answers):
    p = prompts.page_image_prompt("a scene", valid_answers)
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
