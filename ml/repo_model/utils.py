"""
Applied upgrades in this file:
  - A100/CUDA runtime optimization helpers
  - H100/Hopper runtime optimization helpers

Shared utilities for the Voice Anti-Spoofing pipeline.
Fixed random seed: 42
"""

import os
import random
import numpy as np
import torch
import platform

# ---- FIXED SEED (report this in the report) ----
SEED = 42


def set_seed(seed: int = SEED):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_torch_runtime(
    device: torch.device | None = None,
    enable_tf32: bool = True,
    benchmark: bool = True,
    enable_sdp_kernels: bool = True,
) -> torch.device:
    """
    Configure PyTorch runtime for high-throughput CUDA inference/training.

    On A100-class GPUs this enables TF32 and cudnn benchmark mode by default.
    """
    device = device or get_device()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        torch.backends.cudnn.benchmark = benchmark
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        if enable_sdp_kernels:
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        if benchmark:
            torch.backends.cudnn.deterministic = False
    return device


def get_autocast_dtype(device: torch.device | None = None):
    """Prefer bf16 on Hopper/Ampere when supported; fall back to fp16."""
    device = device or get_device()
    if device.type != "cuda":
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_autocast_kwargs(device: torch.device | None = None) -> dict:
    device = device or get_device()
    kwargs = {
        "device_type": "cuda" if device.type == "cuda" else "cpu",
        "enabled": device.type == "cuda",
    }
    dtype = get_autocast_dtype(device)
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def dataloader_kwargs(
    num_workers: int,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> dict:
    """Create DataLoader kwargs tuned for GPU training."""
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def get_hardware_info() -> dict:
    """Return hardware specification for the benchmarking section."""
    info = {
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
        info["cuda_capability"] = f"{capability[0]}.{capability[1]}"
        info["autocast_dtype"] = str(get_autocast_dtype(torch.device("cuda")))
        info["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
        info["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    return info


def print_header(title: str):
    """Print a formatted section header."""
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")
