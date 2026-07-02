"""Configuration loading and dataframe IO helpers.

All paths in the config are relative to the project root, which is the
directory containing ``config/`` and ``src/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def get_project_root() -> Path:
    """Return the project root (the directory containing ``src/``)."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML file. Defaults to ``<root>/config/config.yaml``.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return get_project_root() / path


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a dataframe to CSV, creating parent directories if needed."""
    full_path = resolve_path(path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=True)
    return full_path


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV saved by :func:`save_dataframe` with a datetime index."""
    full_path = resolve_path(path)
    df = pd.read_csv(full_path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df
