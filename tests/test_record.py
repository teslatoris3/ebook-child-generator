"""Tests for pipeline/record.py — Book Record: reproducible seeds + book.json."""
import pytest

from config import Config
from pipeline.record import BookRecord, PageRecord, seed_for


# --- seed_for: deterministic ---

import dataclasses

from pipeline.record import REFERENCE_INDEX


def test_seed_for_is_deterministic(valid_answers):
    cfg = Config()
    a = seed_for(valid_answers, 0, cfg)
    b = seed_for(valid_answers, 0, cfg)
    assert a == b


def test_seed_for_differs_by_page_index(valid_answers):
    cfg = Config()
    assert seed_for(valid_answers, 0, cfg) != seed_for(valid_answers, 1, cfg)


def test_seed_for_differs_by_child_name(valid_answers):
    cfg = Config()
    other = dataclasses.replace(valid_answers, child_name="Max")
    assert seed_for(valid_answers, 0, cfg) != seed_for(other, 0, cfg)


def test_seed_for_uses_base_seed_when_nonzero(valid_answers):
    cfg = Config(base_seed=1000)
    # base_seed dominates: name no longer affects the seed.
    other = dataclasses.replace(valid_answers, child_name="Max")
    assert seed_for(valid_answers, 0, cfg) == 1000
    assert seed_for(other, 0, cfg) == 1000


def test_seed_for_reference_index(valid_answers):
    cfg = Config()
    ref = seed_for(valid_answers, REFERENCE_INDEX, cfg)
    page0 = seed_for(valid_answers, 0, cfg)
    assert ref != page0


# --- for_answers: build up front ---

def test_for_answers_keeps_answers(valid_answers):
    rec = BookRecord.for_answers(valid_answers, Config())
    assert rec.answers == valid_answers


def test_for_answers_starts_with_no_pages(valid_answers):
    rec = BookRecord.for_answers(valid_answers, Config())
    assert rec.pages == []


def test_for_answers_sets_reference_seed(valid_answers):
    cfg = Config()
    rec = BookRecord.for_answers(valid_answers, cfg)
    assert rec.reference_seed == seed_for(valid_answers, REFERENCE_INDEX, cfg)


# --- save / load round-trip ---

def _filled_record(valid_answers) -> BookRecord:
    rec = BookRecord.for_answers(valid_answers, Config())
    rec.title = "Luna and the Dragon"
    rec.reference_path = "reference.png"
    rec.pages = [
        PageRecord(prompt="page one prompt", seed=11, image_path="page_01.png"),
        PageRecord(prompt="page two prompt", seed=12, image_path="page_02.png"),
    ]
    return rec


def test_save_writes_book_json(tmp_path, valid_answers):
    out = _filled_record(valid_answers).save(tmp_path)
    assert out == tmp_path / "book.json"
    assert out.exists()


def test_round_trip_preserves_record(tmp_path, valid_answers):
    original = _filled_record(valid_answers)
    original.save(tmp_path)
    loaded = BookRecord.load(tmp_path)
    assert loaded == original


def test_load_rejects_version_mismatch(tmp_path, valid_answers):
    import json
    rec = _filled_record(valid_answers)
    path = rec.save(tmp_path)
    data = json.loads(path.read_text())
    data["version"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        BookRecord.load(tmp_path)


# --- reroll (bump_seed) ---

def test_bump_seed_default_increments_stored(valid_answers):
    rec = _filled_record(valid_answers)
    new = rec.bump_seed(0)
    assert new == 12  # stored 11 + 1
    assert rec.pages[0].seed == 12


def test_bump_seed_explicit_seed(valid_answers):
    rec = _filled_record(valid_answers)
    new = rec.bump_seed(1, new_seed=500)
    assert new == 500
    assert rec.pages[1].seed == 500


def test_bump_seed_is_reproducible_after_save(tmp_path, valid_answers):
    rec = _filled_record(valid_answers)
    rec.bump_seed(0)
    rec.save(tmp_path)
    assert BookRecord.load(tmp_path).pages[0].seed == 12
