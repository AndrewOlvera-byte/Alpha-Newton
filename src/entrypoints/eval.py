"""Alpha-Newton evaluation entrypoint.

Runs ecosystem-standard benchmarks via EleutherAI lm-evaluation-harness.

This mirrors the training workflow:
- Training:  python src/entrypoints/train_*.py --exp <exp_name>
- Eval:      python src/entrypoints/eval.py --exp <exp_name> --suite <suite_name>

Available suites (configs/eval/*.yaml):
- quick:     Fast sanity check (gsm8k, arc_challenge, hellaswag)
- standard:  Balanced coverage (adds winogrande, truthfulqa)
- full:      Comprehensive (adds mmlu, hendrycks_math, ifeval)
- reasoning: Focused on reasoning tasks for thinking models
- math_small: GSM8K only
- math_full:  GSM8K + MATH

Notes on prompts:
- We default to lm-eval's `--apply_chat_template`, which wraps each benchmark prompt
  into an OpenAI-style messages list (user role) and uses the tokenizer's
  `apply_chat_template(...)` for correct chat formatting.

Examples:
  # Quick sanity check
  python src/entrypoints/eval.py --exp qwen3_600M_sft_mixed_chat_12ksteps_2048seq --suite quick

  # Full evaluation with WandB logging  
  python src/entrypoints/eval.py --exp qwen3_600M_sft_mixed_chat_12ksteps_2048seq --suite full --wandb

  # Evaluate specific checkpoint
  python src/entrypoints/eval.py --exp qwen3_600M_rlvr_math --suite reasoning --checkpoint checkpoint-5000

  # Evaluate any HuggingFace model directly
  python src/entrypoints/eval.py --model Qwen/Qwen3-0.6B --suite quick
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.core.config import Config, deep_merge, interpolate_variables


# Task aliases -> lm-eval task IDs
# Allows friendly names in configs while mapping to official task names
_TASK_ALIASES: Dict[str, str] = {
    # Math reasoning
    "gsm8k": "gsm8k",
    "math": "hendrycks_math",
    "hendrycks_math": "hendrycks_math",
    
    # Commonsense & reasoning
    "arc": "arc_challenge",
    "arc_challenge": "arc_challenge",
    "arc_easy": "arc_easy",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    
    # Knowledge
    "mmlu": "mmlu",
    
    # Truthfulness
    "truthfulqa": "truthfulqa_mc2",
    "truthfulqa_mc2": "truthfulqa_mc2",
    
    # Instruction following
    "ifeval": "ifeval",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _gen_kwargs_to_str(gen_kwargs: Dict[str, Any]) -> str:
    """Convert gen_kwargs dict to lm-eval CLI format."""
    parts: List[str] = []
    for k, v in gen_kwargs.items():
        if isinstance(v, bool):
            v_str = "true" if v else "false"
        else:
            v_str = str(v)
        parts.append(f"{k}={v_str}")
    return ",".join(parts)


def _resolve_tasks(tasks: List[str]) -> List[str]:
    """Resolve task aliases to lm-eval task IDs."""
    resolved = []
    for t in tasks:
        t_norm = str(t).strip()
        resolved.append(_TASK_ALIASES.get(t_norm, t_norm))
    return resolved


def _default_output_dir(training_output_dir: str, suite_name: str) -> Path:
    """Default output path: outputs/<run_name>/eval/<suite>"""
    return Path(training_output_dir) / "eval" / suite_name


def _pick_results_json(output_path: Path) -> Optional[Path]:
    """Find the results JSON file in lm-eval output."""
    # lm-eval can write to a file path or a directory
    if output_path.suffix.lower() == ".json" and output_path.exists():
        return output_path

    if output_path.is_dir():
        preferred = output_path / "results.json"
        if preferred.exists():
            return preferred

        # Fallback: pick the newest json file in the directory
        json_files = [
            p for p in output_path.iterdir() 
            if p.is_file() and p.suffix.lower() == ".json"
        ]
        if not json_files:
            return None
        return max(json_files, key=lambda p: p.stat().st_mtime)

    return None


def _print_summary(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of evaluation results."""
    task_results = results.get("results", {})
    if not isinstance(task_results, dict) or not task_results:
        print("[Eval] No task results found in lm-eval output.")
        return

    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS")
    print("=" * 70)
    
    for task, metrics in sorted(task_results.items()):
        print(f"\n📊 {task}")
        print("-" * 40)
        if isinstance(metrics, dict):
            # Filter to main metrics (skip stderr, alias, etc.)
            main_metrics = {
                k: v for k, v in metrics.items() 
                if isinstance(v, (int, float)) and not k.endswith("_stderr")
            }
            for k, v in sorted(main_metrics.items()):
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"    {metrics}")
    
    print("\n" + "=" * 70)


def _build_model_args(
    pretrained: str,
    training_cfg: Optional[Config] = None,
) -> str:
    """Build lm-eval model_args string."""
    parts = [f"pretrained={pretrained}"]
    
    if training_cfg is not None:
        if training_cfg.model.get("trust_remote_code", False):
            parts.append("trust_remote_code=True")
        if dtype := training_cfg.model.get("dtype"):
            parts.append(f"dtype={dtype}")
        if revision := training_cfg.model.get("revision"):
            parts.append(f"revision={revision}")
    
    return ",".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate models on standard benchmarks via lm-eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/entrypoints/eval.py --exp qwen3_600M_sft_mixed_chat_12ksteps_2048seq --suite quick
  python src/entrypoints/eval.py --exp qwen3_600M_rlvr_math --suite reasoning --checkpoint checkpoint-5000
  python src/entrypoints/eval.py --model Qwen/Qwen3-0.6B --suite standard
        """
    )
    
    # Model source (one of these required)
    model_group = parser.add_argument_group("Model Selection (pick one)")
    model_group.add_argument(
        "--exp", type=str, default=None,
        help="Training experiment name (configs/exp/{exp}.yaml)"
    )
    model_group.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model ID or local path (alternative to --exp)"
    )
    model_group.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint folder under output_dir (e.g. checkpoint-2000)"
    )
    
    # Eval configuration
    eval_group = parser.add_argument_group("Evaluation Settings")
    eval_group.add_argument(
        "--suite", type=str, required=True,
        help="Eval suite name (configs/eval/{suite}.yaml)"
    )
    eval_group.add_argument(
        "--output", type=str, default=None,
        help="Override output path (dir or .json)"
    )
    eval_group.add_argument(
        "--limit", type=int, default=None,
        help="Limit samples per task (for quick testing)"
    )
    eval_group.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size from config"
    )
    
    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--wandb", action="store_true",
        help="Log summary metrics to WandB"
    )

    args = parser.parse_args()

    # Load eval suite config
    suite_path = Path("configs/eval") / f"{args.suite}.yaml"
    if not suite_path.exists():
        available = [p.stem for p in Path("configs/eval").glob("*.yaml")]
        raise FileNotFoundError(
            f"Eval suite not found: {suite_path}\n"
            f"Available suites: {', '.join(sorted(available))}"
        )

    suite_cfg = _load_yaml(suite_path)

    # Load training experiment config (optional)
    training_cfg: Optional[Config] = None
    if args.exp:
        training_cfg = Config.from_experiment(args.exp)

    # Determine model path
    if args.model:
        pretrained = args.model
        training_output_dir = None
    elif training_cfg is not None:
        pretrained = training_cfg.training["output_dir"]
        training_output_dir = training_cfg.training["output_dir"]
    else:
        raise ValueError("Provide either --exp or --model")

    # Handle checkpoint
    if args.checkpoint:
        if training_output_dir is None:
            raise ValueError("--checkpoint requires --exp (so we know output_dir)")
        pretrained = str(Path(training_output_dir) / args.checkpoint)

    # Parse runner settings from suite config
    runner = suite_cfg.get("runner", {})
    tasks = _resolve_tasks(runner.get("tasks", []))
    if not tasks:
        raise ValueError(f"No tasks configured in {suite_path}")

    num_fewshot = int(runner.get("num_fewshot", 0))
    batch_size = args.batch_size or runner.get("batch_size", 4)
    apply_chat_template = bool(runner.get("apply_chat_template", True))
    gen_kwargs = runner.get("gen_kwargs", {}) or {}

    # Build model args
    model_args = _build_model_args(pretrained, training_cfg)

    # Determine output path
    suite_name = (suite_cfg.get("suite", {}) or {}).get("name", args.suite)
    output_cfg = suite_cfg.get("output", {}) or {}
    output_path_raw = args.output if args.output is not None else output_cfg.get("path")

    if output_path_raw:
        merged_for_interp = {
            "run": (training_cfg.run if training_cfg is not None else {"name": "eval"}),
        }
        output_path = Path(
            interpolate_variables(
                deep_merge(merged_for_interp, {"output": {"path": output_path_raw}})
            )["output"]["path"]
        )
    else:
        if training_cfg is None:
            safe_name = pretrained.replace("/", "_").replace("\\", "_")
            output_path = Path("outputs") / safe_name / "eval" / suite_name
        else:
            output_path = _default_output_dir(
                training_cfg.training["output_dir"], suite_name
            )

    # Ensure output directory exists
    if output_path.suffix.lower() != ".json":
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build lm-eval command
    cmd: List[str] = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model", str(runner.get("model", "hf")),
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--output_path", str(output_path),
    ]

    if apply_chat_template:
        cmd.append("--apply_chat_template")

    if gen_kwargs:
        cmd.extend(["--gen_kwargs", _gen_kwargs_to_str(gen_kwargs)])

    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    # Print run info
    print("\n" + "=" * 70)
    print(" ALPHA-NEWTON EVALUATION")
    print("=" * 70)
    print(f"  Suite:      {suite_name}")
    print(f"  Tasks:      {', '.join(tasks)}")
    print(f"  Model:      {pretrained}")
    print(f"  Batch size: {batch_size}")
    print(f"  Few-shot:   {num_fewshot}")
    print(f"  Output:     {output_path}")
    if args.limit:
        print(f"  Limit:      {args.limit} samples/task")
    print("=" * 70 + "\n")

    # Run lm-eval
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)

    # Read and display results
    results_path = _pick_results_json(output_path)
    if results_path is None:
        print("[Eval] Completed, but could not find results JSON.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    _print_summary(results)

    # Optional WandB logging
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
                        if isinstance(v, (int, float)):
                            flat[f"eval/{suite_name}/{task}/{k}"] = v

            wandb.log(flat)
            wandb.finish()
            print(f"\n[Eval] Results logged to WandB: {training_cfg.wandb['project']}")
        except Exception as e:
            print(f"[Eval] WandB logging skipped due to error: {e}")


if __name__ == "__main__":
    main()
