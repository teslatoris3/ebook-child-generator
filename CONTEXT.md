# Domain Context — Kids Ebook Generator

Shared vocabulary for the project. Keep terms here consistent with the code so
both humans and agents name the same concepts the same way. Architectural
vocabulary (module / interface / seam / adapter / leverage / locality) is
defined by the improve-codebase-architecture skill, not here.

## Terms

### Answers
One filled-in 9-field questionnaire. The single input that defines a book.
(`pipeline/questionnaire.py`)

### Character Sheet
The frozen visual description of the hero child, reused verbatim on every page
so the child stays recognisable. (`pipeline/prompts.py:character_sheet`)

### Reference (image)
The one hero portrait generated up front (IP-Adapter scale 0). Every page is
then conditioned on it to lock the child's identity. (`pipeline/images.py:make_reference`)

### Book Record
The reproducible record of one generated book: the Answers, the title, and for
each page its prompt, its seed, and (once rendered) its saved image path, plus
the reference path. Persisted as `book.json` in the book's output folder.

It is the single owner of reproducibility: given the same Answers it derives the
same per-page **seeds** deterministically (`seed_for`), and re-rendering or
rerolling a page goes through it so the new state is itself reproducible. Both
`generate` and `regenerate_page` build/read it rather than tracking seeds and
prompts on their own. (`pipeline/record.py`)

- **seed_for(answers, page_index)** — deterministic seed. Uses
  `config.base_seed` when non-zero; otherwise a stable `hashlib`-derived hash of
  `child_name` (NOT Python's salted `hash()`), offset by `page_index`. The
  Reference uses a fixed sentinel index.
- **reroll** — producing a new variant of one page. Defaults to the stored seed
  + 1 (a deterministic "next" variant); an explicit seed may be passed. The new
  seed is written back to the Book Record so the reroll is reproducible too.

### Book Generator (orchestrator)
Owns the StoryWriter + ImageStudio for one session and runs the pipeline
(story → reference → pages → compose → PDF), building the Book Record as it
goes and emitting Progress. (`pipeline/orchestrator.py`)
