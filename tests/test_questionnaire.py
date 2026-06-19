"""Tests for Answers.validate() — pure logic, no models."""
import dataclasses
import pytest
from pipeline.questionnaire import Answers, PRONOUNS, HAIR_COLORS, SKIN_TONES, ANIMALS, LOVED_ONES, THEMES, SETTINGS, ART_STYLES


# --- tracer bullet: valid answers pass silently ---

def test_valid_answers_do_not_raise(valid_answers):
    valid_answers.validate()  # must not raise


# --- child_name guards ---

def test_empty_name_raises(valid_answers):
    a = dataclasses.replace(valid_answers, child_name="")
    with pytest.raises(ValueError, match="child_name"):
        a.validate()


def test_whitespace_only_name_raises(valid_answers):
    a = dataclasses.replace(valid_answers, child_name="   ")
    with pytest.raises(ValueError, match="child_name"):
        a.validate()


@pytest.mark.parametrize("bad_char", list(r'\/:*?"<>|'))
def test_unsafe_filesystem_chars_in_name_raise(valid_answers, bad_char):
    a = dataclasses.replace(valid_answers, child_name=f"kid{bad_char}name")
    with pytest.raises(ValueError, match="child_name"):
        a.validate()


# --- custom values are allowed (dropdowns accept typed input) ---

@pytest.mark.parametrize("field", [
    "pronoun", "hair_color", "skin_tone", "favourite_animal",
    "loved_one", "theme", "setting", "art_style",
])
def test_custom_value_is_allowed(valid_answers, field):
    """Fields accept values outside their catalog (user can type their own)."""
    a = dataclasses.replace(valid_answers, **{field: "something custom"})
    a.validate()  # must not raise


@pytest.mark.parametrize("field", [
    "pronoun", "hair_color", "skin_tone", "favourite_animal",
    "loved_one", "theme", "setting", "art_style",
])
def test_empty_required_field_raises(valid_answers, field):
    """Required art/story fields must still be non-empty."""
    a = dataclasses.replace(valid_answers, **{field: "   "})
    with pytest.raises(ValueError, match=field):
        a.validate()


@pytest.mark.parametrize("field,catalog", [
    ("pronoun", PRONOUNS),
    ("hair_color", HAIR_COLORS),
    ("skin_tone", SKIN_TONES),
    ("favourite_animal", ANIMALS),
    ("loved_one", LOVED_ONES),
    ("theme", THEMES),
    ("setting", SETTINGS),
    ("art_style", ART_STYLES),
])
def test_all_catalog_values_are_valid(valid_answers, field, catalog):
    for value in catalog:
        a = dataclasses.replace(valid_answers, **{field: value})
        a.validate()  # must not raise


# --- favourite_activities (free text, optional) ---

def test_favourite_activities_optional(valid_answers):
    a = dataclasses.replace(valid_answers, favourite_activities="")
    a.validate()  # empty is fine — LLM invents activities


def test_favourite_activities_free_text_ok(valid_answers):
    a = dataclasses.replace(valid_answers, favourite_activities="cooking, bathing, brushing teeth")
    a.validate()
