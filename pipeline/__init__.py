"""Kids Ebook Generator pipeline package.

Importing this package is intentionally *light*: no torch / diffusers / llama_cpp
imports happen at module load. Heavy frameworks are imported lazily inside the
``load()`` methods of ``StoryWriter`` / ``ImageStudio`` so the package can be
imported (and the Gradio UI built) without paying model-load cost.
"""
from __future__ import annotations

from .device import DevicePolicy, pick_devices
from .questionnaire import Answers
from .orchestrator import BookGenerator, BookResult, Progress

__all__ = [
    "DevicePolicy",
    "pick_devices",
    "Answers",
    "BookGenerator",
    "BookResult",
    "Progress",
]
