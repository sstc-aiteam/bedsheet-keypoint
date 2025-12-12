"""
PyTorch memory helpers.

Why this exists:
- Some training scripts load a second model for evaluation ("reload from checkpoint")
  while the trained model is still referenced. That can temporarily hold two full
  model copies on GPU and trigger OOM.
"""

from __future__ import annotations

import gc
from typing import Any

import torch


def cleanup_torch_memory() -> None:
    """
    Best-effort cleanup of Python + PyTorch allocator state.

    Notes:
    - This does NOT magically free memory if you still hold references to tensors/models.
      You must drop references (e.g., `model = None`) before calling this.
    """
    gc.collect()

    if torch.cuda.is_available():
        # Releases cached blocks back to the CUDA caching allocator.
        torch.cuda.empty_cache()
        # Helps clean up IPC memory (can matter in some multiprocess / dataloader cases).
        torch.cuda.ipc_collect()


def free_torch_memory(*objs: Any, move_modules_to_cpu: bool = False) -> None:
    """
    Best-effort cleanup helper intended to be called right after you drop references.

    Example:
        trained_model = None
        optimizer = None
        free_torch_memory(move_modules_to_cpu=True)

    Args:
        objs: Optional objects; if `move_modules_to_cpu=True`, any torch.nn.Module-like
              objects with `.to()` will be moved to CPU before cleanup (best-effort).
        move_modules_to_cpu: Move modules to CPU to release GPU memory sooner.
    """
    if move_modules_to_cpu:
        for obj in objs:
            try:
                if isinstance(obj, torch.nn.Module):
                    obj.to("cpu")
            except Exception:
                # Best-effort only; never fail training because a cleanup attempt failed.
                pass

    cleanup_torch_memory()


