"""Alpha-Newton math evaluation entrypoint.

Runs ecosystem-standard math benchmarks via EleutherAI lm-evaluation-harness.

This mirrors the training workflow:
- Training:  python src/entrypoints/train_*.py --exp <exp_name>
- Eval:      python src/entrypoints/eval_math.py --exp <exp_name> --suite <suite_name>

Suites live in: configs/eval/*.yaml

Notes on prompts:
- We default to lm-eval's `--apply_chat_template`, which wraps each benchmark prompt
  into an OpenAI-style messages list (user role) and uses the tokenizer's
  `apply_chat_template(...)` for correct chat formatting.

Example:
  python src/entrypoints/eval_math.py --exp qwen3_600M_rlvr_math --suite math_small
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.core.config import Config, deep_merge, interpolate_variables


_TASK_ALIASES = {
    # aliases -> lm-eval task IDs
    "gsm8k": "gsm8k",
    # lm-eval uses hendrycks_math for the MATH benchmark family
    "math": "hendrycks_math",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _gen_kwargs_to_str(gen_kwargs: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k, v in gen_kwargs.items():
        if isinstance(v, bool):
            v_str = "true" if v else "false"
        else:
            v_str = str(v)
        parts.append(f"{k}={v_str}")
    return ",".join(parts)


def _resolve_tasks(tasks: List[str]) -> List[str]:
    resolved = []
    for t in tasks:
        t_norm = str(t).strip()
        resolved.append(_TASK_ALIASES.get(t_norm, t_norm))
    return resolved


def _default_output_dir(training_output_dir: str, suite_name: str) -> Path:
    return Path(training_output_dir) / "eval" / suite_name


def _pick_results_json(output_path: Path) -> Path | None:
    # lm-eval can write to a file path or a directory.
    if output_path.suffix.lower() == ".json" and output_path.exists():
        return output_path

    if output_path.is_dir():
        preferred = output_path / "results.json"
        if preferred.exists():
            return preferred

        # Fallback: pick the newest json file in the directory.
        json_files = [p for p in output_path.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
        if not json_files:
            return None
        return max(json_files, key=lambda p: p.stat().st_mtime)

    # If output_path is a file that doesn't exist, nothing to read.
    return None


def _print_summary(results: Dict[str, Any]) -> None:
    task_results = results.get("results", {})
    if not isinstance(task_results, dict) or not task_results:
        print("[Eval] No task results found in lm-eval output.")
        return

    print("\n[Eval] Summary")
    print("=" * 60)
    for task, metrics in task_results.items():
        print(f"- {task}:")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"    {metrics}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate math benchmarks (GSM8K/MATH) via lm-eval")
    parser.add_argument("--exp", type=str, default=None, help="Training experiment name (configs/exp/{exp}.yaml)")
    parser.add_argument("--suite", type=str, required=True, help="Eval suite name (configs/eval/{suite}.yaml)")

    parser.add_argument("--model", type=str, default=None, help="Override pretrained model path or HF ID")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint folder under output_dir (e.g. checkpoint-2000)")
    parser.add_argument("--output", type=str, default=None, help="Override output path (dir or .json)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation samples (smoke testing)")

    parser.add_argument("--wandb", action="store_true", help="Log summary metrics to WandB")

    args = parser.parse_args()

    suite_path = Path("configs/eval") / f"{args.suite}.yaml"
    if not suite_path.exists():
        raise FileNotFoundError(f"Eval suite not found: {suite_path}")

    suite_cfg = _load_yaml(suite_path)

    # Load training exp config (optional, but preferred for consistent output dirs)
    training_cfg = None
    if args.exp:
        training_cfg = Config.from_experiment(args.exp)

    # Determine pretrained model path
    if args.model:
        pretrained = args.model
        training_output_dir = None
    elif training_cfg is not None:
        pretrained = training_cfg.training["output_dir"]
        training_output_dir = training_cfg.training["output_dir"]
    else:
        raise ValueError("Provide either --exp or --model")

    # If evaluating a specific checkpoint, resolve it under output_dir
    if args.checkpoint:
        if training_output_dir is None:
            raise ValueError("--checkpoint requires --exp (so we know output_dir)")
        pretrained = str(Path(training_output_dir) / args.checkpoint)

    # Runner settings
    runner = suite_cfg.get("runner", {})
    tasks = _resolve_tasks(runner.get("tasks", []))
    if not tasks:
        raise ValueError(f"No tasks configured in {suite_path}")

    num_fewshot = int(runner.get("num_fewshot", 0))
    batch_size = runner.get("batch_size", 4)
    apply_chat_template = bool(runner.get("apply_chat_template", True))
    gen_kwargs = runner.get("gen_kwargs", {}) or {}

    # Model args: prefer training config if available
    trust_remote_code = False
    dtype = None
    revision = None
    if training_cfg is not None:
        trust_remote_code = bool(training_cfg.model.get("trust_remote_code", False))
        dtype = training_cfg.model.get("dtype")
        revision = training_cfg.model.get("revision")

    model_args_parts = [f"pretrained={pretrained}"]
    if trust_remote_code:
        model_args_parts.append("trust_remote_code=True")
    if dtype:
        model_args_parts.append(f"dtype={dtype}")
    if revision:
        model_args_parts.append(f"revision={revision}")

    # If local checkpoint includes tokenizer files, letting lm-eval load tokenizer from pretrained is fine.
    model_args = ",".join(model_args_parts)

    # Output path
    suite_name = (suite_cfg.get("suite", {}) or {}).get("name", args.suite)

    output_cfg = suite_cfg.get("output", {}) or {}
    output_path_raw = args.output if args.output is not None else output_cfg.get("path")

    if output_path_raw:
        # Allow interpolation (e.g., outputs/${run.name}/eval/...)
        merged_for_interp = {
            "run": (training_cfg.run if training_cfg is not None else {"name": "eval"}),
        }
        output_path = Path(interpolate_variables(deep_merge(merged_for_interp, {"output": {"path": output_path_raw}}))["output"]["path"])
    else:
        if training_cfg is None:
            # If user provided --model only, default to a local eval folder
            safe_name = pretrained.replace("/", "_").replace("\\", "_")
            output_path = Path("outputs") / safe_name / "eval" / suite_name
        else:
            output_path = _default_output_dir(training_cfg.training["output_dir"], suite_name)

    # Ensure output directory exists if path is a directory
    if output_path.suffix.lower() != ".json":
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build lm-eval command
    # Use python -m lm_eval to avoid PATH issues.
    cmd: List[str] = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        str(runner.get("model", "hf")),
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--num_fewshot",
        str(num_fewshot),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output_path),
    ]

    if apply_chat_template:
        cmd.append("--apply_chat_template")

    if gen_kwargs:
        cmd.extend(["--gen_kwargs", _gen_kwargs_to_str(gen_kwargs)])

    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    print("[Eval] Running lm-eval")
    print(f"[Eval] Suite: {suite_name} ({suite_path})")
    print(f"[Eval] Pretrained: {pretrained}")
    print(f"[Eval] Tasks: {tasks}")
    print(f"[Eval] Output: {output_path}")
    print()

    # Run
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)

    # Read results
    results_path = _pick_results_json(output_path)
    if results_path is None:
        print("[Eval] Completed, but could not find results JSON.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    _print_summary(results)

    # Optional WandB
    if args.wandb and training_cfg is not None:
        try:
            import wandb

            wandb.init(
                project=training_cfg.wandb["project"],
                entity=training_cfg.wandb["entity"],
                name=f"{training_cfg.run['name']}-eval-{suite_name}",
                tags=(training_cfg.wandb.get("tags", []) + ["eval", suite_name]),
            )

            # Log flattened metrics
            task_results = results.get("results", {}) or {}
            flat = {}
            for task, metrics in task_results.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        flat[f"{suite_name}/{task}/{k}"] = v

            wandb.log(flat)
            wandb.finish()
        except Exception as e:
            print(f"[Eval] WandB logging skipped due to error: {e}")


if __name__ == "__main__":
    main()
