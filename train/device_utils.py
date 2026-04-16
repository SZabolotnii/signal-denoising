"""Centralized device selection and platform-aware utilities.

Supports CUDA, Apple MPS (Metal), and CPU backends transparently.
All trainers and inference scripts import from here instead of
inlining torch.cuda.* calls.
"""

import platform
import subprocess
import torch

__all__ = [
    'get_device',
    'empty_cache',
    'reset_peak_memory',
    'get_peak_memory_gb',
    'format_vram_str',
    'get_dataloader_kwargs',
    'get_batch_size_multiplier',
]


def get_device(preferred: str | None = None) -> torch.device:
    """Auto-detect the best available device.

    Priority: preferred arg → CUDA → MPS → CPU.
    """
    if preferred is not None and preferred != 'auto':
        dev = torch.device(preferred)
        if dev.type == 'cuda' and not torch.cuda.is_available():
            print(f"[device] CUDA requested but not available, falling back to CPU")
            return torch.device('cpu')
        if dev.type == 'mps' and not torch.backends.mps.is_available():
            print(f"[device] MPS requested but not available, falling back to CPU")
            return torch.device('cpu')
        return dev

    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def empty_cache(device: torch.device | str) -> None:
    """Release cached memory for the given device backend."""
    dtype = device.type if isinstance(device, torch.device) else device
    if dtype == 'cuda':
        torch.cuda.empty_cache()
    elif dtype == 'mps':
        torch.mps.empty_cache()


def reset_peak_memory(device: torch.device | str) -> None:
    """Reset peak memory tracking (CUDA only; no-op on MPS/CPU)."""
    dtype = device.type if isinstance(device, torch.device) else device
    if dtype == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_gb(device: torch.device | str) -> float | None:
    """Return peak GPU memory in GB (CUDA only)."""
    dtype = device.type if isinstance(device, torch.device) else device
    if dtype == 'cuda':
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return None


def format_vram_str(device: torch.device | str) -> str:
    """Format a suffix string for epoch logging."""
    peak = get_peak_memory_gb(device)
    if peak is not None:
        return f" | vram={peak:.2f}GB"
    dtype = device.type if isinstance(device, torch.device) else device
    if dtype == 'mps':
        return " | mps"
    return ""


def get_dataloader_kwargs(device: torch.device | str) -> dict:
    """Return pin_memory / num_workers / persistent_workers for DataLoader."""
    dtype = device.type if isinstance(device, torch.device) else device
    if dtype == 'cuda':
        return dict(num_workers=4, pin_memory=True, persistent_workers=True)
    if dtype == 'mps':
        # MPS doesn't support pin_memory; multiprocess workers can cause
        # serialisation issues with MPS tensors on macOS.
        return dict(num_workers=0, pin_memory=False, persistent_workers=False)
    # CPU
    return dict(num_workers=4, pin_memory=False, persistent_workers=True)


def _get_system_memory_gb() -> float | None:
    """Return total system RAM in GB (macOS only)."""
    if platform.system() != 'Darwin':
        return None
    try:
        out = subprocess.check_output(['sysctl', '-n', 'hw.memsize'],
                                      text=True, timeout=5)
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        return None


def get_batch_size_multiplier(device: torch.device | str) -> float:
    """Scale factor for batch sizes relative to the 8 GB NVIDIA baseline.

    On Apple Silicon the GPU shares unified memory, so we can safely
    increase batch sizes proportionally to available RAM.
    """
    dtype = device.type if isinstance(device, torch.device) else device
    if dtype != 'mps':
        return 1.0

    mem_gb = _get_system_memory_gb()
    if mem_gb is None:
        return 1.0

    # Conservative scaling: reserve ~12 GB for OS + other processes,
    # then scale relative to the 8 GB NVIDIA baseline.
    usable = max(mem_gb - 12, 8)
    multiplier = usable / 8.0
    # Cap at 8× to avoid diminishing returns from very large batches.
    return min(multiplier, 8.0)
