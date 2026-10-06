"""Automatic CPU/CUDA selection and concise accelerator reporting."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    name: str
    cuda_available: bool
    memory_bytes: int | None


def get_device(requested: str = "auto") -> torch.device:
    """Choose CUDA when available, or validate an explicit user selection."""
    requested = requested.lower().strip()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    if requested not in {"cpu", "cuda"} and not requested.startswith("cuda:"):
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
    return torch.device(requested)


def get_device_info(device: torch.device | None = None) -> DeviceInfo:
    selected = device or get_device()
    if selected.type == "cuda" and torch.cuda.is_available():
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        return DeviceInfo(selected, properties.name, True, properties.total_memory)
    return DeviceInfo(selected, "CPU", torch.cuda.is_available(), None)


def format_device_info(info: DeviceInfo) -> str:
    if info.memory_bytes is None:
        return f"CPU (CUDA {'available' if info.cuda_available else 'not available'})"
    return f"{info.name} ({info.memory_bytes / 1024**3:.1f} GiB VRAM)"
