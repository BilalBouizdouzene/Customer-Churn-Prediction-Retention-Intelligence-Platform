from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(name: str) -> dict[str, Any]:
    path = ROOT / "configs" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_output_dirs() -> None:
    for relative in [
        "data/interim",
        "data/processed",
        "models",
        "results/metrics",
        "results/figures",
        "results/explainability",
        "results/scores",
        "results/reports",
    ]:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
