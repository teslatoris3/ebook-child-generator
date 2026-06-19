"""Tests for pipeline/images.py — mocks diffusers, no GPU/model needed."""
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from config import Config
from pipeline.device import DevicePolicy
from pipeline.images import ImageStudio


# --- Helpers ---

def _make_policy(llm_device="cpu", sd_device="cuda", vram_gb=4.0) -> DevicePolicy:
    return DevicePolicy(llm_device=llm_device, sd_device=sd_device, vram_gb=vram_gb, reason="test")


@pytest.fixture
def mock_diffusers(monkeypatch):
    """Inject a fake diffusers module; yield (StableDiffusionPipeline mock, LCMScheduler mock)."""
    mod = ModuleType("diffusers")
    sd_pipe_cls = MagicMock(name="StableDiffusionPipeline")
    lcm_sched_cls = MagicMock(name="LCMScheduler")
    mod.StableDiffusionPipeline = sd_pipe_cls
    mod.LCMScheduler = lcm_sched_cls
    monkeypatch.setitem(sys.modules, "diffusers", mod)
    return sd_pipe_cls, lcm_sched_cls


@pytest.fixture
def loaded_studio(mock_diffusers):
    """ImageStudio with load() called; returns (studio, mock_pipe_instance)."""
    sd_pipe_cls, _ = mock_diffusers
    studio = ImageStudio(Config(), _make_policy())
    studio.load()
    pipe = sd_pipe_cls.from_pretrained.return_value
    return studio, pipe


# --- load ---

def test_load_sets_pipe(mock_diffusers):
    studio = ImageStudio(Config(), _make_policy())
    assert studio._pipe is None
    studio.load()
    assert studio._pipe is not None


def test_load_passes_fp16_variant(mock_diffusers):
    sd_pipe_cls, _ = mock_diffusers
    ImageStudio(Config(), _make_policy()).load()
    assert sd_pipe_cls.from_pretrained.call_args.kwargs.get("variant") == "fp16"


def test_load_passes_sd_base_path(mock_diffusers):
    sd_pipe_cls, _ = mock_diffusers
    cfg = Config()
    ImageStudio(cfg, _make_policy()).load()
    assert sd_pipe_cls.from_pretrained.call_args.args[0] == str(cfg.paths.sd_base)


def test_load_sets_lcm_scheduler(loaded_studio, mock_diffusers):
    _, lcm_sched_cls = mock_diffusers
    _, pipe = loaded_studio
    lcm_sched_cls.from_config.assert_called_once()
    assert pipe.scheduler is lcm_sched_cls.from_config.return_value


def test_load_loads_lcm_lora(loaded_studio):
    studio, pipe = loaded_studio
    pipe.load_lora_weights.assert_called_once_with(str(studio.config.paths.lcm_lora_dir))


def test_load_loads_ip_adapter(loaded_studio):
    studio, pipe = loaded_studio
    pipe.load_ip_adapter.assert_called_once()
    kw = pipe.load_ip_adapter.call_args.kwargs
    assert kw.get("subfolder") == studio.config.paths.ip_adapter_subfolder
    assert kw.get("weight_name") == studio.config.paths.ip_adapter_weight


def test_load_enables_cpu_offload(loaded_studio):
    _, pipe = loaded_studio
    pipe.enable_model_cpu_offload.assert_called_once()


# --- unload ---

def test_unload_clears_pipe(loaded_studio):
    studio, _ = loaded_studio
    studio.unload()
    assert studio._pipe is None


def test_unload_empties_cuda_cache(loaded_studio, monkeypatch):
    studio, _ = loaded_studio
    import torch
    empty = MagicMock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty)
    studio.unload()
    empty.assert_called_once()


# --- make_reference ---

def test_make_reference_returns_image(loaded_studio, valid_answers):
    studio, pipe = loaded_studio
    img = studio.make_reference(valid_answers, seed=42)
    assert img is pipe.return_value.images[0]


def test_make_reference_sets_ip_scale_zero(loaded_studio, valid_answers):
    studio, pipe = loaded_studio
    studio.make_reference(valid_answers, seed=42)
    # last set_ip_adapter_scale call before generating should be 0
    assert pipe.set_ip_adapter_scale.call_args.args[0] == 0.0


def test_make_reference_prompt_uses_character_sheet(loaded_studio, valid_answers):
    from pipeline.prompts import character_sheet
    studio, pipe = loaded_studio
    studio.make_reference(valid_answers, seed=42)
    prompt = pipe.call_args.kwargs.get("prompt", "")
    assert character_sheet(valid_answers) in prompt


def test_make_reference_passes_ip_adapter_image(loaded_studio, valid_answers):
    studio, pipe = loaded_studio
    studio.make_reference(valid_answers, seed=42)
    assert pipe.call_args.kwargs.get("ip_adapter_image") is not None


def test_make_reference_uses_lcm_steps(loaded_studio, valid_answers):
    studio, pipe = loaded_studio
    studio.make_reference(valid_answers, seed=42)
    assert pipe.call_args.kwargs.get("num_inference_steps") == studio.config.lcm_steps


# --- make_page ---

def _ref_image():
    return MagicMock(name="reference_image")


def test_make_page_returns_image(loaded_studio):
    studio, pipe = loaded_studio
    img = studio.make_page("a scene", _ref_image(), seed=7)
    assert img is pipe.return_value.images[0]


def test_make_page_passes_prompt(loaded_studio):
    studio, pipe = loaded_studio
    studio.make_page("a magical scene in the woods", _ref_image(), seed=7)
    assert pipe.call_args.kwargs.get("prompt") == "a magical scene in the woods"


def test_make_page_conditions_on_reference(loaded_studio):
    studio, pipe = loaded_studio
    ref = _ref_image()
    studio.make_page("a scene", ref, seed=7)
    assert pipe.call_args.kwargs.get("ip_adapter_image") is ref


def test_make_page_uses_config_ip_scale_by_default(loaded_studio):
    studio, pipe = loaded_studio
    studio.make_page("a scene", _ref_image(), seed=7)
    assert pipe.set_ip_adapter_scale.call_args.args[0] == studio.config.ip_scale


def test_make_page_ip_scale_override(loaded_studio):
    studio, pipe = loaded_studio
    studio.make_page("a scene", _ref_image(), seed=7, ip_scale=0.9)
    assert pipe.set_ip_adapter_scale.call_args.args[0] == 0.9


def test_make_page_uses_lcm_steps_and_guidance(loaded_studio):
    studio, pipe = loaded_studio
    studio.make_page("a scene", _ref_image(), seed=7)
    kw = pipe.call_args.kwargs
    assert kw.get("num_inference_steps") == studio.config.lcm_steps
    assert kw.get("guidance_scale") == studio.config.guidance_scale
