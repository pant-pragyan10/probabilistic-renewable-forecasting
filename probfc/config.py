from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML config (defaults to configs/default.yaml)."""
    with open(path or DEFAULT_CONFIG, "r") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve cfg['paths'][key] against the project root unless it is already absolute."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p
