from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path
import yaml
import re


def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def interpolate_variables(cfg: dict) -> dict:
    pattern = r'\$\{([^}]+)\}'

    def resolve_value(value, cfg_root):
        if isinstance(value, str):
            def replace_var(match):
                path = match.group(1)
                keys = path.split('.')
                result = cfg_root
                try:
                    for key in keys:
                        result = result[key]
                    return str(result)
                except (KeyError, TypeError):
                    return match.group(0)

            return re.sub(pattern, replace_var, value)
        elif isinstance(value, dict):
            return {k: resolve_value(v, cfg_root) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item, cfg_root) for item in value]
        else:
            return value

    return resolve_value(cfg, cfg)


@dataclass
class Config:
    run: Dict[str, Any]
    model: Dict[str, Any]
    tokenizer: Dict[str, Any]
    data: Dict[str, Any]
    training: Dict[str, Any]
    wandb: Dict[str, Any]
    grpo: Dict[str, Any] = None  # RLVR/GRPO specific config

    @classmethod
    def load(cls, path: str | Path, base_configs: List[str | Path] = None):
        merged = {}

        if base_configs:
            for base_path in base_configs:
                if Path(base_path).exists():
                    with open(base_path, "r") as f:
                        base = yaml.safe_load(f)
                    merged = deep_merge(merged, base)

        with open(path, "r") as f:
            main = yaml.safe_load(f)

        merged = deep_merge(merged, main)

        merged = interpolate_variables(merged)

        known_fields = {'run', 'model', 'tokenizer', 'data', 'training', 'wandb', 'grpo'}
        filtered = {k: v for k, v in merged.items() if k in known_fields}

        return cls(**filtered)

    @classmethod
    def from_experiment(cls, exp_name: str):
        common_path = Path("configs/base/common.yaml")
        rlvr_path = Path("configs/base/rlvr.yaml")
        exp_path = Path("configs/exp") / f"{exp_name}.yaml"

        with open(exp_path, "r") as f:
            exp_cfg = yaml.safe_load(f)

        mode = exp_cfg.get("run", {}).get("mode", "")

        base_configs = [common_path]
        if mode == "rlvr" and rlvr_path.exists():
            base_configs.append(rlvr_path)

        return cls.load(exp_path, base_configs=base_configs)
