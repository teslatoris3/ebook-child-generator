# Domain Context — Kids Ebook Generator

Shared vocabulary for the project. Keep terms here consistent with the code so
both humans and agents name the same concepts the same way. Architectural
vocabulary (module / interface / seam / adapter / leverage / locality) is
defined by the improve-codebase-architecture skill, not here.

## Terms

### Answers
One filled-in questionnaire — the single input that defines a book.
(`pipeline/questionnaire.py`)

**Inputs accept custom values (relaxes CLAUDE.md's "dropdown-constrained" rule,
2026-06-19).** The dropdowns are *suggestions*: the UI uses
`allow_custom_value=True` and the user may type any value. `validate()` therefore
only guards the child name (non-empty + filesystem-safe) and requires the other
art/story fields be non-empty — it no longer rejects out-of-catalog values.
Consumers of catalog→fragment maps (`prompts.py`) fall back to the raw typed
value, so custom inputs flow straight into prompts. The decision: flexibility for
places/activities/animals was worth more than guaranteed prompt fragments.

- **favourite_activities** — optional free-text (e.g. "cooking, bathing,
  brushing"). Seeds the per-page activity planning so each page shows a different
  activity and its poem matches that activity.
- **character_type** — what the hero IS: a human "child" (default) OR any
  creature (dinosaur, alien, robot, dragon, …; custom values welcome). Humans
  are described with pronoun + hair + skin; non-humans are drawn as that creature
  with `skin_tone` as the body colour (the "hair/skin" phrasing is dropped). See
  `HUMAN_CHARACTER_TYPES` in `pipeline/questionnaire.py`.

### Character Sheet
The frozen visual description of the hero, reused verbatim on every page so the
hero stays recognisable. The hero may be a human child or any creature
(`character_type`); IP-Adapter locks identity the same way either way.
(`pipeline/prompts.py:character_sheet`)

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

### Model Stack
A heavy model that owns a device and follows a `load()` / `unload()` lifecycle:
the StoryWriter (LLM, CPU) and the ImageStudio (SD+IP-Adapter, GPU) are the two
**adapters** of this seam (a `typing.Protocol`). Naming the seam lets the
orchestrator depend on the contract, not the concrete stacks, so tests inject a
fake stack with no GPU/model. (`pipeline/stack.py:ModelStack`)

- **exclusive(stack)** — the one home for the "never two stacks resident on the
  4&nbsp;GB GPU at once" invariant. A context manager that loads a stack on
  enter and unloads it on exit, so the previous stack is always freed before the
  next loads. (`pipeline/stack.py:exclusive`)

### Book Generator (orchestrator)
Owns the StoryWriter + ImageStudio for one session and runs the pipeline
(story → reference → pages → compose → PDF), building the Book Record as it
goes and emitting Progress. (`pipeline/orchestrator.py`)
