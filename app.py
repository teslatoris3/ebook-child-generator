"""Gradio UI: guided questionnaire OR free-text description -> generate -> PDF.

Wiring layer only. Two input modes share the same output panel:
  • "Fill the form" tab — the 9-field questionnaire (existing behaviour).
  • "Describe your book" tab — paste a plain-English description; the LLM
    extracts the questionnaire fields automatically (no vector store needed for
    the current example library size — in-prompt few-shot suffices).
All real generation logic lives in the pipeline modules.
"""
from __future__ import annotations

from typing import Iterator

import gradio as gr

from config import Config
from pipeline.orchestrator import BookGenerator, BookResult, Progress
from pipeline.questionnaire import (
    ANIMALS,
    ART_STYLES,
    Answers,
    CHARACTER_TYPES,
    HAIR_COLORS,
    LOVED_ONES,
    PRONOUNS,
    SETTINGS,
    SKIN_TONES,
    THEMES,
)

CONFIG = Config()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _on_generate(
    child_name: str,
    character_type: str,
    pronoun: str,
    hair_color: str,
    skin_tone: str,
    favourite_animal: str,
    loved_one: str,
    theme: str,
    setting: str,
    art_style: str,
    favourite_activities: str,
    ip_scale: float,
) -> Iterator[tuple]:
    """Form-mode handler: build Answers, run the generator, stream UI updates."""
    answers = Answers(
        child_name=child_name,
        character_type=character_type,
        pronoun=pronoun,
        hair_color=hair_color,
        skin_tone=skin_tone,
        favourite_animal=favourite_animal,
        loved_one=loved_one,
        theme=theme,
        setting=setting,
        art_style=art_style,
        favourite_activities=favourite_activities,
    )
    CONFIG.ip_scale = float(ip_scale)
    generator = BookGenerator(CONFIG)
    gallery: list[str] = []
    for event in generator.generate(answers):
        if isinstance(event, Progress):
            if event.preview_path is not None:
                gallery = gallery + [str(event.preview_path)]
            status = f"**{event.stage}** {event.current}/{event.total} — {event.message}"
            yield status, gallery, None
        elif isinstance(event, BookResult):
            yield f"✅ Done: *{event.title}*", [str(p) for p in event.page_image_paths], str(event.pdf_path)


def _on_generate_from_description(description: str, ip_scale: float) -> Iterator[tuple]:
    """Description-mode handler: extract Answers via LLM, then generate book."""
    if not description.strip():
        yield "⚠️ Please enter a description first.", [], None
        return
    CONFIG.ip_scale = float(ip_scale)
    generator = BookGenerator(CONFIG)
    gallery: list[str] = []
    for event in generator.generate_from_description(description):
        if isinstance(event, Progress):
            if event.preview_path is not None:
                gallery = gallery + [str(event.preview_path)]
            status = f"**{event.stage}** {event.current}/{event.total} — {event.message}"
            yield status, gallery, None
        elif isinstance(event, BookResult):
            yield f"✅ Done: *{event.title}*", [str(p) for p in event.page_image_paths], str(event.pdf_path)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_ui() -> "gr.Blocks":
    with gr.Blocks(title="Kids Ebook Generator") as demo:
        gr.Markdown("# 📖 Kids Ebook Generator\nMake a personalized rhyming picture book.")

        with gr.Row():
            # ── Left column: input (two tabs) ───────────────────────────────
            with gr.Column(scale=1):
                with gr.Tabs():

                    # ── Tab 1: form ─────────────────────────────────────────
                    with gr.TabItem("📝 Fill the form"):
                        child_name = gr.Textbox(label="Hero's name")
                        character_type = gr.Dropdown(
                            CHARACTER_TYPES, label="Hero is a…", value=CHARACTER_TYPES[0],
                            allow_custom_value=True,
                            info="Child, or any creature — dinosaur, alien, robot, dragon…",
                        )
                        pronoun = gr.Dropdown(PRONOUNS, label="Pronoun", value=PRONOUNS[0], allow_custom_value=True)
                        hair_color = gr.Dropdown(HAIR_COLORS, label="Hair color (if human)", value=HAIR_COLORS[0], allow_custom_value=True)
                        skin_tone = gr.Dropdown(SKIN_TONES, label="Skin / body color", value=SKIN_TONES[0], allow_custom_value=True)
                        favourite_animal = gr.Dropdown(ANIMALS, label="Favourite animal", value=ANIMALS[0], allow_custom_value=True)
                        loved_one = gr.Dropdown(LOVED_ONES, label="Include a loved one", value=LOVED_ONES[0], allow_custom_value=True)
                        theme = gr.Dropdown(THEMES, label="Story is about…", value=THEMES[0], allow_custom_value=True)
                        setting = gr.Dropdown(SETTINGS, label="Overall setting", value=SETTINGS[0], allow_custom_value=True)
                        art_style = gr.Dropdown(ART_STYLES, label="Art style", value=ART_STYLES[0], allow_custom_value=True)
                        favourite_activities = gr.Textbox(
                            label="Favourite activities (comma-separated)",
                            placeholder="e.g. cooking, bathing, painting, dancing",
                            info="Each page shows a different activity; the poem matches it.",
                        )
                        ip_scale_form = gr.Slider(
                            0.0, 1.0, value=CONFIG.ip_scale, step=0.05,
                            label="Character lock (lower = more varied scenes, looser face)",
                        )
                        generate_btn = gr.Button("Generate my book", variant="primary")

                    # ── Tab 2: free-text description ─────────────────────────
                    with gr.TabItem("✍️ Describe your book"):
                        gr.Markdown(
                            "Paste a plain-English description and the AI will figure out the details.\n\n"
                            "**Example:** *Make a book about Rex, a green baby dinosaur who loves cooking "
                            "in the jungle. His dad comes along. Watercolor art style.*"
                        )
                        description_box = gr.Textbox(
                            label="Describe your book",
                            placeholder=(
                                'e.g. "A book about Luna, a girl with blonde hair who loves dancing '
                                'and cooking. Include her mom. Theme: being brave. Enchanted forest setting."'
                            ),
                            lines=6,
                        )
                        ip_scale_desc = gr.Slider(
                            0.0, 1.0, value=CONFIG.ip_scale, step=0.05,
                            label="Character lock (lower = more varied scenes, looser face)",
                        )
                        generate_desc_btn = gr.Button("Generate from description", variant="primary")

            # ── Right column: shared output ──────────────────────────────────
            with gr.Column(scale=2):
                status = gr.Markdown("Fill in the form or describe your book, then press generate.")
                gallery = gr.Gallery(label="Pages", columns=4, height="auto")
                pdf_file = gr.File(label="Download PDF")

        # Wire form tab
        generate_btn.click(
            _on_generate,
            inputs=[child_name, character_type, pronoun, hair_color, skin_tone,
                    favourite_animal, loved_one, theme, setting, art_style,
                    favourite_activities, ip_scale_form],
            outputs=[status, gallery, pdf_file],
        )

        # Wire description tab
        generate_desc_btn.click(
            _on_generate_from_description,
            inputs=[description_box, ip_scale_desc],
            outputs=[status, gallery, pdf_file],
        )

    return demo


def main() -> None:
    CONFIG.ensure_dirs()
    build_ui().launch()


if __name__ == "__main__":
    main()
