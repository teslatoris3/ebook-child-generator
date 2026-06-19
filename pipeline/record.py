"""The Book Record: the reproducible record of one generated book.

Single owner of reproducibility. Given the same ``Answers`` it derives the same
per-page **seeds** (``seed_for``), and re-rendering / rerolling a page goes
through it so the new state is itself reproducible. Persisted as ``book.json``
in the book's output folder. See CONTEXT.md ("Book Record").
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config

    from .questionnaire import Answers

BOOK_JSON_NAME = "book.json"

# Bump when the book.json shape changes; load() refuses mismatches.
BOOK_JSON_VERSION = 1

# Sentinel page index for the hero Reference image (it isn't a numbered page).
REFERENCE_INDEX = -1


def _name_hash(name: str) -> int:
    """Stable, cross-process hash of the child name (NOT Python's salted hash)."""
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)  # 32-bit, plenty of spread, fits any RNG


def seed_for(answers: "Answers", page_index: int, config: "Config") -> int:
    """Deterministic seed for one page (or the Reference at REFERENCE_INDEX).

    Uses ``config.base_seed`` when non-zero; otherwise a stable hash of the
    child name. Offset by ``page_index`` so pages within a book differ.
    """
    base = config.base_seed if config.base_seed else _name_hash(answers.child_name)
    return base + page_index


@dataclass
class PageRecord:
    """One page's reproducible state: prompt, seed, stanza, and (once rendered) path.

    The stanza is persisted so a single page can be re-rendered and recomposed
    (a reroll) without re-running the LLM.
    """

    prompt: str
    seed: int
    stanza: str = ""
    image_path: str | None = None


@dataclass
class BookRecord:
    """The reproducible record of one book; round-trips to ``book.json``."""

    answers: "Answers"
    title: str = ""
    reference_seed: int = 0
    pages: list[PageRecord] = field(default_factory=list)
    reference_path: str | None = None
    version: int = BOOK_JSON_VERSION

    @classmethod
    def for_answers(cls, answers: "Answers", config: "Config") -> "BookRecord":
        """Start an empty record for ``answers`` with the reference seed fixed.

        Page records are appended as each page is rendered (their seeds come from
        ``seed_for(answers, page_index, config)``).
        """
        return cls(
            answers=answers,
            reference_seed=seed_for(answers, REFERENCE_INDEX, config),
        )

    # -- reroll --------------------------------------------------------------
    def bump_seed(self, page_index: int, new_seed: int | None = None) -> int:
        """Set a new seed for one page and return it (the reroll operation).

        Defaults to the stored seed + 1 (a deterministic "next" variant); pass
        ``new_seed`` to jump to a specific seed. The change is written onto the
        page record so a subsequent ``save`` keeps the reroll reproducible.
        """
        page = self.pages[page_index]
        page.seed = page.seed + 1 if new_seed is None else new_seed
        return page.seed

    # -- persistence ---------------------------------------------------------
    def save(self, out_dir: Path) -> Path:
        """Write ``book.json`` into ``out_dir`` and return its path."""
        path = Path(out_dir) / BOOK_JSON_NAME
        payload = {
            "version": self.version,
            "answers": self.answers.to_dict(),
            "title": self.title,
            "reference_seed": self.reference_seed,
            "reference_path": self.reference_path,
            "pages": [asdict(p) for p in self.pages],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, out_dir: Path) -> "BookRecord":
        """Read ``book.json`` from ``out_dir``; raise ValueError on shape/version mismatch."""
        from .questionnaire import Answers

        path = Path(out_dir) / BOOK_JSON_NAME
        data = json.loads(path.read_text())

        version = data.get("version")
        if version != BOOK_JSON_VERSION:
            raise ValueError(
                f"book.json version {version!r} != supported {BOOK_JSON_VERSION}"
            )

        return cls(
            answers=Answers(**data["answers"]),
            title=data.get("title", ""),
            reference_seed=data.get("reference_seed", 0),
            reference_path=data.get("reference_path"),
            pages=[PageRecord(**p) for p in data.get("pages", [])],
            version=version,
        )
