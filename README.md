# Kids Ebook Generator

A fully **offline**, AI-powered personalized children's picture book generator.
Fill in a 9-field questionnaire — or just describe the book in plain English — and get
an 8-page rhyming storybook (age 3–6) with one illustration per page, exported as a PDF.
No cloud. No API keys. Everything runs on local models.

---

## What it produces

| Output | Details |
|--------|---------|
| **8-page storybook PDF** | Cover + 8 illustrated pages, each with a 4-line rhyming stanza |
| **Hero reference image** | One portrait used to lock the child's appearance across all pages |
| **Per-page PNGs** | Composited illustration + poem, ready to print |
| **`book.json`** | Seeds, prompts, and answers for reproducible page rerolls |

---

## Features

- **Two input modes** — guided 9-field form *or* free-text description (the LLM extracts the fields automatically via in-prompt few-shot; no vector store required)
- **Character consistency** — hero is locked across pages via IP-Adapter; animal companion is seed-frozen
- **Dynamic stories** — the LLM plans 2–3 pages *without* the hero (companion solo, setting shots, loved-one moments) so the story breathes instead of repeating one pose
- **Human *or* creature hero** — child, dinosaur, alien, robot, dragon, or any custom type
- **Favourite activities** — each page shows the child doing a *different* activity; the poem matches the activity on that page
- **Reproducible rerolls** — deterministic seeds mean any single page can be regenerated without touching the rest
- **Fully local** — LLM on CPU, SD on GPU; the two stacks are never co-resident (4 GB VRAM budget enforced by the `exclusive()` context manager)

---

## Architecture

```
app.py                       ← Gradio UI (form tab + description tab)
├── pipeline/
│   ├── orchestrator.py      ← BookGenerator: ties all stages together
│   ├── story.py             ← StoryWriter: outline → stanzas → title (LLM)
│   ├── images.py            ← ImageStudio: SD 1.5 + IP-Adapter + LCM-LoRA (GPU)
│   ├── prompts.py           ← character sheet + per-page image prompts
│   ├── compose.py           ← Pillow: composite illustration + poem → PDF
│   ├── questionnaire.py     ← Answers dataclass + dropdown catalogs
│   ├── record.py            ← BookRecord: reproducibility (seeds, prompts, book.json)
│   ├── stack.py             ← ModelStack protocol + exclusive() context manager
│   └── device.py            ← Adaptive CPU/GPU device policy
├── config.py                ← All paths + tunables in one place
├── tests/                   ← 174 unit tests (no models needed — all mocked)
└── scripts/
    ├── download_models.sh   ← Fetch all three image models (~4.8 GB)
    └── smoke_book.py        ← End-to-end CLI smoke test
```

### Pipeline stages (in order)

```
Questionnaire / description
        │
        ▼
1. LLM → 8-beat outline (place + activity per page, hero_present flag)
2. LLM → 4-line rhyming stanza per beat (sequential, prior stanzas in context)
3. LLM → book title
        │
        ▼  (exclusive: LLM unloaded before SD loads)
4. SD   → hero reference image (IP-Adapter scale = 0, text-only)
5. SD   → 8 page illustrations (IP-Adapter locks hero; hero-free pages skip lock)
        │
        ▼
6. Pillow → composite illustration + stanza per page
7. Pillow → cover page
8. PDF export + book.json saved
```

---

## Hardware requirements

| Component | Minimum | Tested on |
|-----------|---------|-----------|
| GPU VRAM  | 4 GB    | RTX 3050 Laptop 4 GB |
| RAM       | 12 GB   | 15 GB |
| CPU       | Any modern x86 | i5-11400H (12 threads) |
| CUDA      | 11.8+   | 12.4 |
| Disk      | ~10 GB  | (models ~8 GB + outputs) |

> **GPU budget**: LLM runs on CPU; SD stack runs on GPU via `enable_model_cpu_offload`.
> They never co-reside. Peak VRAM during image generation: ~1.9 GB / 4 GB.

---

## Software requirements

- **Conda** (Miniconda or Anaconda) with a Python 3.11 environment that has
  `torch 2.x + CUDA` already installed
- `git`, `wget` (for model downloads)
- A Hugging Face account (free) — models are gated

> The dependencies in `requirements.txt` are intentionally installed **into your
> existing conda env** so they share the same CUDA-enabled PyTorch. Do not install
> torch again via pip — it will clobber the CUDA build.

---

## Step-by-step setup

### 1 — Clone the repo

```bash
git clone https://github.com/teslatoris3/ebook-child-generator.git
cd ebook-child-generator
```

### 2 — Install Python dependencies

```bash
# activate the conda env that already has torch+CUDA
conda activate base          # or whatever env has torch 2.x with CUDA

pip install -r requirements.txt
```

### 3 — Download the LLM

The project uses **Gemma-4B (IQ4_NL GGUF)** via [LM Studio](https://lmstudio.ai/).

1. Install LM Studio and open it.
2. Search for: `bartowski/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-GGUF`
3. Download the `IQ4_NL` variant.

LM Studio saves it to `~/.lmstudio/models/bartowski/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/`.
The path is already set in `config.py`.

**Alternative LLM (smaller/faster):** The project also supports `qwen2.5-1.5b-instruct-q4_k_m.gguf`
placed at `~/models/llm/`. Update `config.py → Paths.llm` to switch.

### 4 — Download the image models (~4.8 GB)

You need a [Hugging Face](https://huggingface.co) account and must accept the model licences
on the HF website before downloading.

**Models required:**

| Model | HF repo | Size |
|-------|---------|------|
| DreamShaper V8 (SD 1.5) | `Lykon/dreamshaper-8` | ~3.4 GB (fp16) |
| IP-Adapter (SD 1.5) | `h94/IP-Adapter` | ~1.0 GB |
| LCM-LoRA (SD 1.5) | `latent-consistency/lcm-lora-sdv1-5` | ~390 MB |

**Download script (recommended):**

```bash
# Log in to Hugging Face CLI first
pip install huggingface_hub
huggingface-cli login          # paste your HF token when prompted

# Then run the download script
bash scripts/download_models.sh
```

The script saves everything under `~/models/sd/`:

```
~/models/sd/
├── dreamshaper-8/             ← SD 1.5 base (diffusers format)
├── ip-adapter/                ← IP-Adapter weights + CLIP encoder
└── lcm-lora-sdv1-5/          ← LCM-LoRA for fast sampling
```

**Manual download alternative:**
Each model can also be downloaded from `https://huggingface.co/<repo>/tree/main`
via browser. Place the files in the paths shown above.

---

## Running the app

```bash
conda activate base
python app.py
```

Open the URL printed to the terminal (usually `http://127.0.0.1:7860`).

### Tab 1 — Fill the form

Fill in the 9 fields:

| Field | Type | Notes |
|-------|------|-------|
| Hero's name | Text | Any name |
| Hero is a… | Dropdown | `child`, `dinosaur`, `alien`, `robot`, … or type anything |
| Pronoun | Dropdown | girl / boy / child |
| Hair color | Dropdown | Applies to human heroes |
| Skin / body color | Dropdown | For non-humans this is their body colour |
| Favourite animal | Dropdown | The soft-consistent companion |
| Include a loved one | Dropdown | Appears in the story |
| Story is about… | Dropdown | Theme / moral |
| Overall setting | Dropdown | World backdrop |
| Art style | Dropdown | Watercolor, pastel cartoon, crayon, storybook |
| Favourite activities | Free text | Comma-separated — each page shows a different activity |
| Character lock slider | Slider | 0 = varied scenes, 1 = tighter face lock |

### Tab 2 — Describe your book

Paste plain English and press **Generate from description**. The LLM extracts
all fields automatically.

**Example:**
```
Make a book about Rex, a green baby dinosaur who loves cooking in the jungle.
His dad comes along. Watercolor art style.
```

---

## CLI smoke test (no UI needed)

```bash
python scripts/smoke_book.py
# or with custom args:
python scripts/smoke_book.py --name Rex --character-type dinosaur --skin green
```

Runs the full pipeline end-to-end and saves output to `output/`.

---

## Running the tests

```bash
# All 174 tests — no models required (everything is mocked)
python -m pytest tests/ -q
```

The test suite covers every pipeline module individually:
`compose`, `device`, `images`, `orchestrator`, `prompts`, `questionnaire`,
`record`, `stack`, `story` — including the free-text extraction, `ModelStack`
protocol, `exclusive()` invariant, and `BookRecord` reproducibility.

---

## Key technical decisions

| Decision | Rationale |
|----------|-----------|
| **LLM on CPU, SD on GPU** | 4 GB VRAM cannot hold both; `exclusive()` enforces never-co-resident |
| **IP-Adapter for hero consistency** | Reliable single-subject lock at SD 1.5 / 4 GB; ControlNet and full fine-tuning require more VRAM |
| **LCM-LoRA** | 6-step sampling (~2.8 s/page) instead of 50-step DDIM |
| **`enable_model_cpu_offload`** | Safer than manual layer surgery; peaks at 1.9 GB / 4 GB |
| **In-prompt few-shot extraction** | 3 examples replace a FAISS/RAG pipeline for this corpus size; zero infrastructure cost |
| **`Beat.hero_present`** | LLM decides per-page whether the hero appears; hero-free pages set IP-Adapter scale to 0 and omit the character sheet from the prompt |
| **Deterministic seeds** | `SHA-256(child_name + config)` → reproducible rerolls without storing RNG state |
| **Structured generation** | Outline → stanzas (sequential, prior stanzas in context) → title; never one-shot the whole book |

---

## Project structure in depth

```
pipeline/stack.py         ModelStack protocol + exclusive() context manager
                          Enforces: LLM and SD never co-reside on GPU

pipeline/record.py        BookRecord: answers + seeds + prompts saved as book.json
                          Any page can be rerolled from disk without re-generating the rest

pipeline/story.py         StoryWriter wraps llama-cpp-python
                          generate_outline() → generate_stanzas() → generate_title()
                          extract_from_prompt() for free-text description mode

pipeline/images.py        ImageStudio wraps diffusers
                          make_reference() → IP-Adapter scale 0, text-only portrait
                          make_page()      → IP-Adapter scale 0.4, conditioned on reference

pipeline/prompts.py       Pure string assembly (no models)
                          character_sheet() · page_image_prompt() · cover_image_prompt()
                          Hero-free pages omit the character sheet entirely

pipeline/compose.py       Pillow: illustration + stanza → PNG, pages → PDF

pipeline/orchestrator.py  BookGenerator: runs all stages in order
                          generate(answers) and generate_from_description(text)
```

---

## Development methodology

This project was developed using **[Claude Code](https://claude.ai/code)** — Anthropic's
AI-powered CLI for software engineering — as the primary development tool.

The workflow followed strict **Test-Driven Development**:

- Every module was specced with failing tests before implementation
- 174 unit tests cover all pipeline modules with no real models (everything mocked)
- Architecture was reviewed and refactored via the `improve-codebase-architecture` skill
  (surfaced shallow modules, concentrated logic, eliminated pass-throughs)

Claude Code handled: module design, TDD cycles, architecture review and refactor,
debugging GPU memory issues, and writing this README — while the developer directed
decisions, validated outputs, and maintained full ownership of the codebase.

---

## License

MIT
