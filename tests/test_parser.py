"""Tests for pipeline/parser.py — mocks the LLM, no model needed."""
import json
from unittest.mock import MagicMock

import pytest

from pipeline.parser import ANSWER_DEFAULTS, extract_answers_dict, extraction_messages


# --- Helpers ---

def _mock_llm(payload: dict) -> MagicMock:
    m = MagicMock()
    m.create_chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps(payload)}}]
    }
    return m


# --- extraction_messages ---

def test_extraction_messages_embeds_description():
    text = "A brave little robot named Bolt"
    msgs = extraction_messages(text)
    combined = " ".join(m["content"] for m in msgs)
    assert text in combined


def test_extraction_messages_has_system_and_user():
    msgs = extraction_messages("anything")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_extraction_messages_asks_for_json():
    msgs = extraction_messages("anything")
    assert "JSON" in msgs[1]["content"]


def test_extraction_messages_lists_required_keys():
    msgs = extraction_messages("anything")
    user_text = msgs[1]["content"]
    for key in ("child_name", "character_type", "pronoun", "skin_tone", "favourite_animal"):
        assert key in user_text


# --- extract_answers_dict ---

def test_extract_returns_child_name():
    llm = _mock_llm({"child_name": "Bolt", "character_type": "robot"})
    result = extract_answers_dict("A robot named Bolt", llm)
    assert result["child_name"] == "Bolt"


def test_extract_returns_character_type():
    llm = _mock_llm({"child_name": "Bolt", "character_type": "robot"})
    result = extract_answers_dict("...", llm)
    assert result["character_type"] == "robot"


def test_extract_calls_llm_with_json_response_format():
    llm = _mock_llm({"child_name": "X"})
    extract_answers_dict("anything", llm)
    kw = llm.create_chat_completion.call_args.kwargs
    assert kw.get("response_format", {}).get("type") == "json_object"


def test_extract_uses_low_temperature():
    llm = _mock_llm({"child_name": "X"})
    extract_answers_dict("anything", llm)
    kw = llm.create_chat_completion.call_args.kwargs
    assert kw.get("temperature", 1.0) <= 0.2


def test_extract_passes_messages_through():
    llm = _mock_llm({"child_name": "Luna", "pronoun": "she/her"})
    extract_answers_dict("a girl named Luna", llm)
    msgs = llm.create_chat_completion.call_args.kwargs.get("messages", [])
    combined = " ".join(m["content"] for m in msgs)
    assert "a girl named Luna" in combined


def test_extract_returns_full_payload():
    payload = {
        "child_name": "Zara",
        "character_type": "alien",
        "pronoun": "she/her",
        "hair_color": "purple",
        "skin_tone": "purple",
        "favourite_animal": "space cat",
        "loved_one": "Grandma",
        "theme": "curiosity",
        "setting": "outer space",
        "art_style": "cartoon",
        "favourite_activities": "painting, reading",
    }
    llm = _mock_llm(payload)
    result = extract_answers_dict("...", llm)
    for key, val in payload.items():
        assert result[key] == val


# --- ANSWER_DEFAULTS ---

def test_answer_defaults_has_required_fields():
    for key in ("character_type", "pronoun", "skin_tone", "loved_one",
                "theme", "setting", "art_style"):
        assert key in ANSWER_DEFAULTS


def test_answer_defaults_character_type_is_child():
    assert ANSWER_DEFAULTS["character_type"] == "child"
