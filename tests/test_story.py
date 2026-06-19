"""Tests for pipeline/story.py — mocks llama_cpp, no model needed."""
import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from config import Config
from pipeline.device import DevicePolicy
from pipeline.story import StoryWriter


# --- Helpers ---

def _llm_resp(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _outline_json(n: int = 8) -> str:
    return json.dumps({"beats": [
        {
            "beat": f"Beat {i + 1}: the hero does something",
            "place": f"place {i + 1}",
            "activity": f"activity {i + 1}",
        }
        for i in range(n)
    ]})


def _stanza(lines: int = 4) -> str:
    return "\n".join(f"Line {i + 1} rhymes with something fine" for i in range(lines))


def _beats(n: int = 8):
    from pipeline.story import Beat
    return [Beat(text=f"beat {i}", place=f"place {i}", activity=f"activity {i}") for i in range(n)]


def _make_policy(llm_device="cpu", sd_device="cuda", vram_gb=4.0) -> DevicePolicy:
    return DevicePolicy(llm_device=llm_device, sd_device=sd_device, vram_gb=vram_gb, reason="test")


@pytest.fixture
def mock_llama_cls(monkeypatch):
    """Inject a fake llama_cpp module; yield the mock Llama class."""
    mod = ModuleType("llama_cpp")
    mock_cls = MagicMock()
    mod.Llama = mock_cls
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    return mock_cls


@pytest.fixture
def loaded_writer(mock_llama_cls):
    """StoryWriter with load() already called; returns (writer, mock_llm_instance)."""
    cfg = Config()
    w = StoryWriter(cfg, _make_policy())
    w.load()
    instance = mock_llama_cls.return_value
    return w, instance


# --- load / unload ---

def test_load_sets_llm(mock_llama_cls):
    w = StoryWriter(Config(), _make_policy())
    assert w._llm is None
    w.load()
    assert w._llm is not None


def test_load_passes_model_path(mock_llama_cls):
    cfg = Config()
    StoryWriter(cfg, _make_policy()).load()
    kw = mock_llama_cls.call_args.kwargs
    assert str(cfg.paths.llm) == kw.get("model_path")


def test_load_passes_n_ctx(mock_llama_cls):
    cfg = Config()
    StoryWriter(cfg, _make_policy()).load()
    assert mock_llama_cls.call_args.kwargs.get("n_ctx") == cfg.llm_ctx


def test_load_passes_zero_gpu_layers_for_cpu(mock_llama_cls):
    cfg = Config()
    StoryWriter(cfg, _make_policy(llm_device="cpu")).load()
    assert mock_llama_cls.call_args.kwargs.get("n_gpu_layers") == 0


def test_unload_clears_llm(mock_llama_cls):
    w = StoryWriter(Config(), _make_policy())
    w.load()
    w.unload()
    assert w._llm is None


# --- generate_outline ---

def test_outline_returns_list_of_8(loaded_writer, valid_answers):
    w, inst = loaded_writer
    inst.create_chat_completion.return_value = _llm_resp(_outline_json(8))
    beats = w.generate_outline(valid_answers)
    assert isinstance(beats, list)
    assert len(beats) == 8


def test_outline_items_are_beats_with_place_and_activity(loaded_writer, valid_answers):
    w, inst = loaded_writer
    inst.create_chat_completion.return_value = _llm_resp(_outline_json(8))
    beats = w.generate_outline(valid_answers)
    for b in beats:
        assert b.text.strip()
        assert b.place.strip()
        assert b.activity.strip()


def test_outline_seeds_prompt_with_favourite_activities(loaded_writer, valid_answers):
    w, inst = loaded_writer
    inst.create_chat_completion.return_value = _llm_resp(_outline_json(8))
    w.generate_outline(valid_answers)
    msgs = inst.create_chat_completion.call_args.kwargs.get("messages", [])
    text = " ".join(m.get("content", "") for m in msgs)
    assert valid_answers.favourite_activities in text


def test_outline_calls_llm_exactly_once(loaded_writer, valid_answers):
    w, inst = loaded_writer
    inst.create_chat_completion.return_value = _llm_resp(_outline_json(8))
    w.generate_outline(valid_answers)
    inst.create_chat_completion.assert_called_once()


def test_outline_truncates_to_num_pages(loaded_writer, valid_answers):
    w, inst = loaded_writer
    inst.create_chat_completion.return_value = _llm_resp(_outline_json(12))
    beats = w.generate_outline(valid_answers)
    assert len(beats) == w.config.num_pages


def test_outline_uses_json_response_format(loaded_writer, valid_answers):
    w, inst = loaded_writer
    inst.create_chat_completion.return_value = _llm_resp(_outline_json(8))
    w.generate_outline(valid_answers)
    call_kw = inst.create_chat_completion.call_args.kwargs
    fmt = call_kw.get("response_format", {})
    assert fmt.get("type") == "json_object"


# --- generate_stanzas ---

def test_stanzas_count_matches_outline(loaded_writer, valid_answers):
    w, inst = loaded_writer
    outline = _beats(8)
    inst.create_chat_completion.return_value = _llm_resp(_stanza(4))
    stanzas = w.generate_stanzas(outline, valid_answers)
    assert len(stanzas) == len(outline)


def test_stanzas_each_has_lines_per_page_lines(loaded_writer, valid_answers):
    w, inst = loaded_writer
    outline = _beats(8)
    inst.create_chat_completion.return_value = _llm_resp(_stanza(4))
    stanzas = w.generate_stanzas(outline, valid_answers)
    for s in stanzas:
        assert len(s.strip().splitlines()) == w.config.lines_per_page


def test_stanzas_llm_called_once_per_beat(loaded_writer, valid_answers):
    w, inst = loaded_writer
    outline = _beats(8)
    inst.create_chat_completion.return_value = _llm_resp(_stanza(4))
    w.generate_stanzas(outline, valid_answers)
    assert inst.create_chat_completion.call_count == len(outline)


def test_stanzas_are_nonempty_strings(loaded_writer, valid_answers):
    w, inst = loaded_writer
    inst.create_chat_completion.return_value = _llm_resp(_stanza(4))
    stanzas = w.generate_stanzas(_beats(1), valid_answers)
    assert all(isinstance(s, str) and s.strip() for s in stanzas)


def test_stanza_prompt_includes_beat_activity(loaded_writer, valid_answers):
    """The poem must be about the page's activity (poem<->image match)."""
    w, inst = loaded_writer
    from pipeline.story import Beat
    inst.create_chat_completion.return_value = _llm_resp(_stanza(4))
    w.generate_stanzas([Beat(text="b", place="the bath", activity="taking a bubble bath")], valid_answers)
    msgs = inst.create_chat_completion.call_args.kwargs.get("messages", [])
    text = " ".join(m.get("content", "") for m in msgs)
    assert "taking a bubble bath" in text


def test_stanzas_prior_beats_in_later_calls(loaded_writer, valid_answers):
    """Each stanza call should include previously generated stanzas in messages."""
    w, inst = loaded_writer
    outline = _beats(3)
    inst.create_chat_completion.return_value = _llm_resp(_stanza(4))
    w.generate_stanzas(outline, valid_answers)
    # The third call's messages should reference content from at least the first stanza
    third_call_msgs = inst.create_chat_completion.call_args_list[2].kwargs.get("messages", [])
    full_text = " ".join(m.get("content", "") for m in third_call_msgs)
    assert "Line 1 rhymes" in full_text  # text from first stanza mock


# --- generate_title ---

def test_title_returns_nonempty_string(loaded_writer, valid_answers):
    w, inst = loaded_writer
    inst.create_chat_completion.return_value = _llm_resp("Luna and the Dragon")
    title = w.generate_title(valid_answers, _beats(2))
    assert isinstance(title, str) and title.strip()


def test_title_calls_llm_once(loaded_writer, valid_answers):
    w, inst = loaded_writer
    inst.create_chat_completion.return_value = _llm_resp("A Brave Tale")
    w.generate_title(valid_answers, _beats(1))
    inst.create_chat_completion.assert_called_once()


def test_title_strips_whitespace(loaded_writer, valid_answers):
    w, inst = loaded_writer
    inst.create_chat_completion.return_value = _llm_resp("  My Book  \n")
    title = w.generate_title(valid_answers, _beats(1))
    assert title == "My Book"


def test_title_contains_child_name(loaded_writer, valid_answers):
    """Title prompt should mention the child's name."""
    w, inst = loaded_writer
    inst.create_chat_completion.return_value = _llm_resp("Luna's Brave Adventure")
    w.generate_title(valid_answers, _beats(1))
    all_content = " ".join(
        m.get("content", "")
        for m in inst.create_chat_completion.call_args.kwargs.get("messages", [])
    )
    assert valid_answers.child_name in all_content
