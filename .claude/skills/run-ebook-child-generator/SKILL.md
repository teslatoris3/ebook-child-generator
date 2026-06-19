---
name: run-ebook-child-generator
description: >
  Launch, screenshot, and drive the Kids Ebook Generator Gradio web app.
  Use when asked to run, start, screenshot, or verify the app UI.
---

# Run: Kids Ebook Generator

Gradio web app at `http://127.0.0.1:7860`. Driven by snap `chromium-browser` in headless mode.
Screenshots land in `~/snap/chromium/common/` (snap sandbox constraint).

## Prerequisites

No extra packages needed. Required tools are already installed:
- `conda base` env with `gradio` installed
- `/usr/bin/chromium-browser` (snap, Chromium 149)

## Launch

```bash
# From the project root
conda run -n base python app.py &
sleep 5   # wait for Gradio to bind to 127.0.0.1:7860
```

Verify it's up:
```bash
curl -s http://127.0.0.1:7860/ | head -1
# should return: <!doctype html>
```

## Screenshot (agent path)

Chromium snap is sandboxed — `--screenshot` must be run from `~/snap/chromium/common/` and screenshots write there:

```bash
cd ~/snap/chromium/common && /usr/bin/chromium-browser \
  --headless --no-sandbox --disable-gpu \
  --screenshot=gradio_ui.png \
  --window-size=1280,900 \
  --virtual-time-budget=5000 \
  http://127.0.0.1:7860
# Screenshot at: ~/snap/chromium/common/gradio_ui.png
```

## Check the UI via curl (no browser needed)

```bash
# Confirm all 9 form fields are present in the rendered HTML
curl -s http://127.0.0.1:7860/ | grep -c 'gradio-app'
```

## Stop the app

```bash
pkill -f "python app.py"
```

## Gotchas

- **`--screenshot` path is sandboxed.** The snap chromium writes `--screenshot=foo.png` relative to `~/snap/chromium/common/`, not your working directory. Running from any other directory silently produces no file. Always `cd ~/snap/chromium/common/` first.
- **`--virtual-time-budget=5000` is required.** Gradio loads its UI via JavaScript. Without this flag, the screenshot captures "Loading..." instead of the rendered form. 5000 ms is sufficient.
- **Gradio `generate` button will raise `NotImplementedError`** until `orchestrator.BookGenerator.generate()` is implemented. The form renders fine; only generation fails.
- **App must already be running.** The skill does not start the app; if `7860` is not listening, chromium silently screenshots an error page.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Screenshot shows "Loading..." | Add `--virtual-time-budget=5000` |
| Screenshot file not found after command | Must `cd ~/snap/chromium/common/` before running |
| `curl` returns connection refused | App not started; run `conda run -n base python app.py &` first |
| Gradio `OSError: port 7860 already in use` | `pkill -f "python app.py"` then retry |
