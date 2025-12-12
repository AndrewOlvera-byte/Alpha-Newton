from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path
import yaml
import re


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def interpolate_variables(cfg: dict) -> dict:
    """
    Resolve ${var.subvar} style variable references

    Works recursively on dict values to preserve types
    """
    pattern = r'\$\{([^}]+)\}'

    def resolve_value(value, cfg_root):
        """Recursively resolve variables in a value"""
        if isinstance(value, str):
            # Only process strings that contain ${...}
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
            # Preserve types for numbers, bools, None, etc.
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
        """
        Load config with optional base configs to merge

        Args:
            path: Path to main config file
            base_configs: List of base config paths to merge (in order)
        """
        # Start with empty or load base configs
        merged = {}

        if base_configs:
            for base_path in base_configs:
                if Path(base_path).exists():
                    with open(base_path, "r") as f:
                        base = yaml.safe_load(f)
                    merged = deep_merge(merged, base)

        # Load and merge main config
        with open(path, "r") as f:
            main = yaml.safe_load(f)

        merged = deep_merge(merged, main)

        # Interpolate variables like ${run.name}
        merged = interpolate_variables(merged)

        # Filter to only known fields (handles extra sections gracefully)
        known_fields = {'run', 'model', 'tokenizer', 'data', 'training', 'wandb', 'grpo'}
        filtered = {k: v for k, v in merged.items() if k in known_fields}

        return cls(**filtered)

    @classmethod
    def from_experiment(cls, exp_name: str):
        """
        Load config for an experiment

        Merges based on mode:
        - rlvr: common.yaml -> rlvr.yaml -> exp/{exp_name}.yaml
        - default: common.yaml -> exp/{exp_name}.yaml
        """
        common_path = Path("configs/base/common.yaml")
        rlvr_path = Path("configs/base/rlvr.yaml")
        exp_path = Path("configs/exp") / f"{exp_name}.yaml"

        # Peek at exp config to check mode
        with open(exp_path, "r") as f:
            exp_cfg = yaml.safe_load(f)

        mode = exp_cfg.get("run", {}).get("mode", "")

        # Build base config list based on mode
        base_configs = [common_path]
        if mode == "rlvr" and rlvr_path.exists():
            base_configs.append(rlvr_path)

        return cls.load(exp_path, base_configs=base_configs)
