"""Gradio UI: guided 9-field questionnaire -> generate -> previews -> PDF.

This file is the *wiring* layer only. It builds the form, streams progress from
``BookGenerator.generate`` into the UI, and exposes per-page regeneration + PDF
download. All real generation logic lives in the (stubbed) pipeline modules.
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
    HAIR_COLORS,
    LOVED_ONES,
    PRONOUNS,
    SETTINGS,
    SKIN_TONES,
    THEMES,
)

CONFIG = Config()


def _on_generate(
    child_name: str,
    pronoun: str,
    hair_color: str,
    skin_tone: str,
    favourite_animal: str,
    loved_one: str,
    theme: str,
    setting: str,
    art_style: str,
) -> Iterator[tuple]:
    """Gradio streaming handler: build Answers, run the generator, yield UI updates.

    Yields ``(status_markdown, gallery_images, pdf_file)`` tuples. The heavy work
    is delegated to the (currently stubbed) ``BookGenerator``.
    """
    answers = Answers(
        child_name=child_name,
        pronoun=pronoun,
        hair_color=hair_color,
        skin_tone=skin_tone,
        favourite_animal=favourite_animal,
        loved_one=loved_one,
        theme=theme,
        setting=setting,
        art_style=art_style,
    )
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


def build_ui() -> "gr.Blocks":
    """Construct the Gradio Blocks app (layout only; handler wired to pipeline)."""
    with gr.Blocks(title="Kids Ebook Generator") as demo:
        gr.Markdown("# 📖 Kids Ebook Generator\nMake a personalized rhyming picture book.")
        with gr.Row():
            with gr.Column(scale=1):
                child_name = gr.Textbox(label="Child's name")
                pronoun = gr.Dropdown(PRONOUNS, label="Child is a…", value=PRONOUNS[0])
                hair_color = gr.Dropdown(HAIR_COLORS, label="Hair color", value=HAIR_COLORS[0])
                skin_tone = gr.Dropdown(SKIN_TONES, label="Skin tone", value=SKIN_TONES[0])
                favourite_animal = gr.Dropdown(ANIMALS, label="Favourite animal", value=ANIMALS[0])
                loved_one = gr.Dropdown(LOVED_ONES, label="Include a loved one", value=LOVED_ONES[0])
                theme = gr.Dropdown(THEMES, label="Story is about…", value=THEMES[0])
                setting = gr.Dropdown(SETTINGS, label="Setting", value=SETTINGS[0])
                art_style = gr.Dropdown(ART_STYLES, label="Art style", value=ART_STYLES[0])
                generate_btn = gr.Button("Generate my book", variant="primary")
            with gr.Column(scale=2):
                status = gr.Markdown("Fill in the form and press generate.")
                gallery = gr.Gallery(label="Pages", columns=4, height="auto")
                pdf_file = gr.File(label="Download PDF")

        generate_btn.click(
            _on_generate,
            inputs=[child_name, pronoun, hair_color, skin_tone, favourite_animal,
                    loved_one, theme, setting, art_style],
            outputs=[status, gallery, pdf_file],
        )
    return demo


def main() -> None:
    CONFIG.ensure_dirs()
    build_ui().launch()


if __name__ == "__main__":
    main()
