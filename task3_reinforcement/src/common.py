from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict

import numpy as np


def resolve_task3_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_root() -> Path:
    return resolve_task3_root().parent


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def now_stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def software_versions() -> Dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "torch": _package_version("torch"),
        "gymnasium": _package_version("gymnasium"),
        "stable-baselines3": _package_version("stable-baselines3"),
        "matplotlib": _package_version("matplotlib"),
        "scikit-learn": _package_version("scikit-learn"),
    }


def select_torch_device(prefer_mps: bool = True) -> str:
    try:
        import torch

        if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"
