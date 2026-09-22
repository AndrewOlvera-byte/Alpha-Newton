from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge mappings; scalar and list values replace their base."""
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def interpolate_variables(config: dict) -> dict:
    """Resolve ``${section.key}`` references against the merged configuration."""
    pattern = re.compile(r"\$\{([^}]+)\}")

    def resolve(value: Any) -> Any:
        if isinstance(value, str):
            def replace(match: re.Match[str]) -> str:
                current: Any = config
                for key in match.group(1).split("."):
                    if not isinstance(current, dict) or key not in current:
                        return match.group(0)
                    current = current[key]
                return str(current)

            return pattern.sub(replace, value)
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    return resolve(config)


def _load_with_extends(path: Path, seen: set[Path] | None = None) -> dict:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Circular config inheritance involving {path}")

    data = yaml.safe_load(path.read_text()) or {}
    parents = data.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]

    merged: dict = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = PROJECT_ROOT / parent_path
        merged = deep_merge(merged, _load_with_extends(parent_path, seen | {path}))
    return deep_merge(merged, data)


@dataclass
class Config:
    run: dict[str, Any]
    model: dict[str, Any]
    tokenizer: dict[str, Any]
    data: dict[str, Any]
    training: dict[str, Any]
    wandb: dict[str, Any]
    grpo: dict[str, Any]
    topk_eval: dict[str, Any]
    peft: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: str | Path, base_configs: list[str | Path] | None = None) -> "Config":
        merged: dict = {}
        for base_path in base_configs or []:
            merged = deep_merge(merged, _load_with_extends(Path(base_path)))
        merged = deep_merge(merged, _load_with_extends(Path(path)))
        merged = interpolate_variables(merged)
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in merged.items() if key in fields})

    @classmethod
    def from_experiment(cls, experiment: str) -> "Config":
        experiment_path = PROJECT_ROOT / "configs" / "exp" / f"{experiment}.yaml"
        preview = _load_with_extends(experiment_path)
        if preview.get("run", {}).get("mode") != "rlvr":
            raise ValueError(f"Only RLVR experiments are supported: {experiment}")

        return cls.load(
            experiment_path,
            base_configs=[
                PROJECT_ROOT / "configs" / "base" / "common.yaml",
                PROJECT_ROOT / "configs" / "base" / "rlvr.yaml",
            ],
        )
