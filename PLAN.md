# Kids Ebook Generator — v1 Implementation Plan

Detailed build plan for the design agreed in `CLAUDE.md`. Phased so the biggest
risk (SD + IP-Adapter on 4 GB) is retired before the full app is built.

## Phase 0 — Environment

Clone `base` (already has torch 2.10+cu128) to keep it clean, then add libs:

```bash
conda create -n kidsbook --clone base
conda activate kidsbook
pip install diffusers transformers accelerate safetensors peft \
            llama-cpp-python pillow gradio huggingface_hub
# llama-cpp-python: if CPU build is slow, rebuild with CUDA:
#   CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

## Phase 1 — Model downloads (into ~/models/sd/)

```bash
huggingface-cli download Lykon/dreamshaper-8 --local-dir ~/models/sd/dreamshaper-8
huggingface-cli download h94/IP-Adapter \
  models/ip-adapter_sd15.bin models/image_encoder/* \
  --local-dir ~/models/sd/ip-adapter
huggingface-cli download latent-consistency/lcm-lora-sdv1-5 \
  --local-dir ~/models/sd/lcm-lora-sdv1-5
```

## Phase 2 — Feasibility spike (DO THIS BEFORE BUILDING THE APP)

Goal: prove SD 1.5 + IP-Adapter + LCM fit in 4 GB at usable speed.

- Load DreamShaper V8 via `StableDiffusionPipeline`, fuse LCM-LoRA, set
  `LCMScheduler`, `enable_attention_slicing()`, `enable_vae_tiling()`.
- Load IP-Adapter (`pipe.load_ip_adapter(...)`, `ip-adapter_sd15.bin`).
- Generate 1 reference image (no IP), then 1 page conditioned on it (IP scale ~0.6).
- Print peak VRAM (`torch.cuda.max_memory_allocated()`) and wall-clock per image.

Decision gate:

- Fits + fast (<~15 s/img): keep attention slicing, proceed.
- OOM: add `pipe.enable_model_cpu_offload()`; re-measure.
- Still OOM: drop IP-Adapter image encoder to fp16 / try `enable_sequential_cpu_offload()`
  (slower) — last resort.

## Phase 3 — Core pipeline modules

`config.py` — dataclass: model paths, default seeds, IP scale, LCM steps (4–8),
image size (512 or 512×768), output dir, device policy.

`pipeline/device.py` — `pick_devices()`: SD→cuda always; LLM→cpu on ≤4 GB,
cuda otherwise; env-var overrides. Returns a small policy object.

`pipeline/questionnaire.py` — field definitions + maps from dropdown values to
prompt fragments. Proposed catalogs (tunable):

- pronoun: girl→"a little girl, she" / boy→"a little boy, he" / child→"a child, they"
- hair_color: black, brown, blonde, red, dark curly
- skin_tone: light, medium, tan, dark
- favourite_animal: cat, dog, dragon, rabbit, lion, dinosaur, unicorn, bear, (other)
- loved_one: Mom, Dad, Grandma, Grandpa, big sister, big brother, (none)
- theme: making friends, being brave, bedtime, sharing, kindness, trying new things
- setting: enchanted forest, ocean, outer space, cozy home, magical castle, jungle
- art_style: watercolor children's book, soft pastel cartoon, crayon doodle,
  classic storybook illustration

`pipeline/story.py` — llama-cpp-python wrapper.

- `generate_outline(answers) -> list[str]` (8 one-line beats, JSON-constrained).
- `generate_stanzas(outline, answers) -> list[str]` (4 rhyming lines/page,
  generated sequentially with prior stanzas in context for rhyme/flow).
- Include 1–2 few-shot rhyming examples in the system prompt (replaces RAG).
- `generate_title(answers, outline) -> str`.

`pipeline/prompts.py`

- `character_sheet(answers) -> str` (frozen child description string).
- `companion_desc(answers) -> str`.
- `page_image_prompt(beat, answers) -> str` = style + charsheet + companion +
  scene(from beat) + setting. Plus a shared negative prompt.

`pipeline/images.py` — diffusers wrapper.

- `load()` once (pipeline + LCM + IP-Adapter, memory opts from Phase 2).
- `make_reference(answers, seed) -> PIL.Image` (child only).
- `make_page(prompt, reference, seed, ip_scale) -> PIL.Image`.

`pipeline/compose.py` — Pillow.

- `compose_page(image, stanza, font) -> PIL.Image` (text band below/over art).
- `make_cover(title, child_name, hero_image) -> PIL.Image`.
- `to_pdf(pages, out_path)`.

`pipeline/orchestrator.py` — runs stages, yields progress
("Writing story…", "Page 3/8…"), writes `output/<name>_<ts>/`
(`reference.png`, `page_01..08.png`, `book.pdf`, `book.json`).
`book.json` stores answers + every prompt + every seed → reproducible reroll.

## Phase 4 — Gradio app (app.py)

- `gr.Blocks`: 9 inputs (text + dropdowns), "Generate my book" button.
- Stream progress via a generator; show reference + page previews as they render.
- Per-page "🔄 regenerate" + "regenerate child reference" buttons (reuse stored
  seeds/prompts, only re-run the affected image, recompose PDF).
- "Download PDF" button.

## Build order / milestones

1. Phase 0–1 setup + downloads.
2. Phase 2 spike → confirm memory/speed, lock memory settings.
3. story.py end-to-end in a script (outline + 8 stanzas + title) — eyeball quality;
   if 1.5B rhyme is weak, try `smollm2-1.7b` (already downloaded).
4. images.py: reference + 8 pages with IP-Adapter consistency — eyeball consistency.
5. compose.py: PDF with cover.
6. orchestrator.py + book.json.
7. Gradio app + reroll.

## Known v1 risks to watch

- 1.5B rhyme quality/meter over 8 stanzas (mitigation: SmolLM2-1.7B; tighter few-shot).
- IP-Adapter identity drift when the child is small/far in a scene (mitigation:
  tune IP scale, keep child framed prominently).
- Animal soft-consistency will vary page to page (accepted for v1).
- Per-book wall time at 4 GB (mitigation: LCM low steps, attention slicing).
