from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} must contain a YAML mapping at top level.")
    return data


def save_yaml(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def parse_scalar(value: str) -> Any:
    """Parse a command-line override value using YAML scalar/list syntax."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def set_by_dotted_key(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str] | None) -> Dict[str, Any]:
    merged = copy.deepcopy(config)
    if not overrides:
        return merged
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override '{override}'. Use dotted.key=value, e.g. train.epochs=1."
            )
        key, raw_value = override.split("=", 1)
        set_by_dotted_key(merged, key, parse_scalar(raw_value))
    return merged


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, path))
        else:
            flat[path] = value
    return flat


def to_jsonable(data: Any) -> Any:
    """Convert common objects into JSON-serializable values for logging."""
    try:
        json.dumps(data)
        return data
    except TypeError:
        if isinstance(data, dict):
            return {str(k): to_jsonable(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [to_jsonable(v) for v in data]
        return str(data)
