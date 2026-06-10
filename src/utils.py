"""
utils.py

Shared utilities for SIGNAL-Preop reproducibility code.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducible demonstration runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Return selected device.

    Args:
        device: "cuda", "cpu", or None. If None, CUDA is used when available.
    """
    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """
    Save dictionary as JSON.
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: Optional[str | Path]) -> Optional[Path]:
    """
    Convert path-like object to Path, keeping None unchanged.
    """
    if path is None:
        return None

    return Path(path)


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Load a PyTorch checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return torch.load(checkpoint_path, map_location=map_location)


def find_checkpoint(checkpoint_dir: str | Path) -> Path:
    """
    Find the default multitask Longformer checkpoint in a directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "multitask_longformer.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Expected checkpoint not found: {checkpoint_path}"
        )

    return checkpoint_path


def print_config(config: Dict[str, Any]) -> None:
    """
    Pretty-print configuration.
    """
    print(json.dumps(config, indent=2, ensure_ascii=False))


def set_huggingface_cache(cache_dir: Optional[str] = None) -> None:
    """
    Optionally set HuggingFace cache directory.
    """
    if cache_dir:
        os.environ["HF_HOME"] = str(cache_dir)