"""Tests for device.pick_devices() — monkeypatching VRAM detection."""
import pytest
from pipeline import device as device_mod
from pipeline.device import pick_devices, VRAM_GPU_LLM_THRESHOLD_GB


@pytest.fixture(autouse=True)
def patch_vram(monkeypatch):
    """Each test sets VRAM via this fixture's param; default overridden per test."""
    return monkeypatch


# --- tracer bullet: no GPU -> both on CPU ---

def test_no_gpu_puts_both_on_cpu(monkeypatch):
    monkeypatch.setattr(device_mod, "_detect_vram_gb", lambda: 0.0)
    policy = pick_devices()
    assert policy.llm_device == "cpu"
    assert policy.sd_device == "cpu"


# --- 4 GB GPU (target machine): LLM CPU, SD CUDA ---

def test_4gb_gpu_puts_llm_on_cpu_sd_on_cuda(monkeypatch):
    monkeypatch.setattr(device_mod, "_detect_vram_gb", lambda: 4.0)
    policy = pick_devices()
    assert policy.llm_device == "cpu"
    assert policy.sd_device == "cuda"


# --- large GPU (≥ threshold): both on CUDA ---

def test_large_gpu_puts_both_on_cuda(monkeypatch):
    monkeypatch.setattr(device_mod, "_detect_vram_gb", lambda: VRAM_GPU_LLM_THRESHOLD_GB + 1)
    policy = pick_devices()
    assert policy.llm_device == "cuda"
    assert policy.sd_device == "cuda"


# --- env-var overrides ---

def test_force_llm_overrides_auto(monkeypatch):
    monkeypatch.setattr(device_mod, "_detect_vram_gb", lambda: 4.0)
    policy = pick_devices(force_llm="cuda")
    assert policy.llm_device == "cuda"


def test_force_sd_overrides_auto(monkeypatch):
    monkeypatch.setattr(device_mod, "_detect_vram_gb", lambda: 4.0)
    policy = pick_devices(force_sd="cpu")
    assert policy.sd_device == "cpu"


# --- policy carries metadata ---

def test_policy_carries_vram(monkeypatch):
    monkeypatch.setattr(device_mod, "_detect_vram_gb", lambda: 3.7)
    policy = pick_devices()
    assert abs(policy.vram_gb - 3.7) < 0.01


def test_policy_reason_is_non_empty(monkeypatch):
    monkeypatch.setattr(device_mod, "_detect_vram_gb", lambda: 4.0)
    policy = pick_devices()
    assert policy.reason
