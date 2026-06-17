# Kids Ebook Generator — Project Context (v1)

> This file replaces TWO earlier plans: an abandoned Rust/Tauri/candle/mistral.rs
> design, and a stale Python design (topic-string input, RAG, VLM refine loop,
> single poem). The agreed v1 is below; rationale is in "Decisions" so we don't
> relitigate. See `PLAN.md` for the detailed build plan.

## What this project is
A desktop, fully-offline, AI-powered **personalized children's picture book**
generator. A guided **questionnaire (9 fields)** produces an **8-page rhyming
storybook** (age 3–6) with one illustration per page and exports a **PDF**.
No cloud, no API keys — local models only.

## Dev machine (the only v1 target)
Ubuntu 20.04, **RTX 3050 Laptop 4 GB VRAM**, i5-11400H (12 threads), 15 GB RAM,
nvcc 12.4. Conda `base` already has **Python 3.11.9 + torch 2.10.0+cu128 (CUDA OK)**.

## Cast & consistency (the central design constraint)
- We cannot reproduce a real child's likeness (no photo input). The "child" is a
  **stylized character** built from chosen traits (pronoun, hair color, skin tone).
- At SD 1.5 / 4 GB, IP-Adapter reliably locks **one** subject:
  - **Child = hard-locked hero** via IP-Adapter across all 8 pages (generate one
    reference image, then condition every page on it).
  - **Favourite animal = soft-consistent companion** (frozen description + seed,
    no IP-Adapter lock).
  - **Loved one ("Dad"/"Mom"/…) = in the text** + maybe 1–2 pages, soft-consistent.

## Stack
| Concern          | Choice |
|------------------|--------|
| Language/runtime | Python 3.11 (conda env cloned from `base`) |
| Text (LLM)       | `llama-cpp-python` + `~/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf`, **on CPU** |
| Images           | `diffusers` SD 1.5 = **DreamShaper V8** + **IP-Adapter** + **LCM-LoRA**, **on GPU** |
| Consistency      | IP-Adapter (child) + seed/character-sheet (animal); art style from dropdown → prompt |
| UI               | **Gradio** (form → progress → previews → PDF download) |
| Output           | **Pillow**-composited pages → **PDF** (+ generated title/cover) |
| Device policy    | adaptive; on 4 GB → LLM-CPU / SD-GPU, **no GPU swapping** |

## Questionnaire (9 fields — dropdowns for anything that affects the art)
`child_name`(text) · `pronoun` · `hair_color` · `skin_tone` · `favourite_animal`
· `loved_one` · `theme/lesson` · `setting` · `art_style`

## Pipeline (in order)
1. Collect questionnaire (9 fields) in Gradio.
2. Build **character sheet** (locked child description) + companion description.
3. LLM: generate **8-beat outline** (theme + setting + cast).
4. LLM: generate **4-line rhyming stanza per page** (sequential, prior lines carried).
5. Build **per-page image prompt** = style + charsheet + companion + scene + setting.
6. SD: generate the **child reference image** once.
7. SD + IP-Adapter (+LCM): generate **8 page illustrations** (child locked).
8. **Pillow**: composite illustration + stanza per page, add cover/title.
9. Combine → **PDF**; write `book.json` (answers, prompts, seeds) for reproducible reroll.

## Model file locations (`~/models/`)
- `llm/`   — GGUF LLM ✅ (`qwen2.5-1.5b-instruct-q4_k_m.gguf`, `smollm2-1.7b-…`)
- `sd/`    — **EMPTY — must download** DreamShaper V8 + IP-Adapter + LCM-LoRA
- `vlm/`   — Qwen2-VL / SmolVLM2 GGUF — **unused in v1**
- `embed/` — **EMPTY — unused in v1** (RAG cut)

### Downloads needed (HF, authenticated as `tesilator`)
- `Lykon/dreamshaper-8`  (SD 1.5 base)
- `h94/IP-Adapter`  (`models/ip-adapter_sd15.bin` + `models/image_encoder/`)
- `latent-consistency/lcm-lora-sdv1-5`  (LCM-LoRA for SD 1.5)

## Proposed module layout
```
app.py                # Gradio UI + progress callbacks
config.py             # paths, model ids, device policy defaults
pipeline/
  device.py           # adaptive GPU/CPU policy (4GB -> LLM CPU, SD GPU)
  questionnaire.py    # field defs + dropdown -> prompt-fragment maps
  story.py            # LLM: outline + per-page stanzas (llama-cpp-python)
  prompts.py          # character sheet + per-page image prompts
  images.py           # diffusers SD + IP-Adapter + LCM; reference + pages
  compose.py          # Pillow page compositing -> PDF
  orchestrator.py     # ties stages together, emits progress
assets/fonts/         # bundled kid-friendly fonts
output/<name>_<ts>/   # reference.png, page_01..08.png, book.pdf, book.json
```

## Coding rules
- v1 = this machine only. No cross-platform/packaging/Android complexity.
- Never keep the LLM and SD+IP-Adapter stacks on the GPU simultaneously (4 GB).
- Prefer dropdown-constrained inputs for anything that affects the art.
- Structured generation only (outline → stanzas); never one-shot the whole book.
- Persist seeds + prompts so any page can be regenerated reproducibly.

## Decisions (resolved — do not relitigate without new info)
- **Python, not Rust.** Rust/candle has no IP-Adapter/ControlNet; mistral.rs's
  diffusion loader is FLUX-only (~24 GB) and cannot run SD 1.5 / DreamShaper.
  Python/diffusers is the only stack that solves character consistency at 4 GB.
- **Android/Tauri dropped.** Local SD on a phone is infeasible; desktop-only v1.
- **Multi-page** chosen over single-spread, accepting the consistency work.
- **Personalized questionnaire** input, NOT a free-text topic string.
- **RAG cut.** No kids-poem corpus exists; few-shot examples in the prompt suffice.
- **VLM refine loop cut.** Extra 2 GB+ model + vague "drift" trigger; not worth v1.
- **DreamShaper V8 + style-via-prompt** over a baked-in cartoon checkpoint.
- **PDF via Pillow** over WeasyPrint/reportlab (no fragile system deps).
- **LLM on CPU / SD on GPU** — they can't co-reside in 4 GB; avoids swap churn.

## v1 feasibility risk to retire first
Confirm SD + IP-Adapter + LCM fit in 4 GB at usable speed via a small **spike**
(load stack, generate 1 reference + 1 page, measure VRAM/time). Decide
`enable_attention_slicing` + VAE tiling vs `enable_model_cpu_offload` from results.
