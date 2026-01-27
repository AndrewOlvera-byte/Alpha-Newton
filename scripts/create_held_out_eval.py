#!/usr/bin/env python3
"""
Create held-out evaluation dataset for Top-K checkpoint management

Usage:
    python scripts/create_held_out_eval.py --dataset gsm8k --output held_out/gsm_eval.jsonl --n_samples 200
    python scripts/create_held_out_eval.py --dataset MATH --output held_out/math_eval.jsonl --n_samples 500
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter


def create_gsm8k_eval(tokenizer, n_samples=200, seed=42):
    print("Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")

    dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

    print(f"Processing {len(dataset)} samples...")
    samples = []

    for item in dataset:
        question = item["question"]
        answer = item["answer"].split("####")[-1].strip()

        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        samples.append({
            "prompt": prompt,
            "answer": answer,
            "difficulty": "level1",
            "problem": question,
            "dataset_source": "gsm8k"
        })

    return samples


def create_math_eval(tokenizer, n_samples=500, seed=42):
    print(f"Loading MATH test set...")
    dataset = load_dataset("hendrycks/competition_math", split="test")

    dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

    print(f"Processing {len(dataset)} samples...")
    samples = []

    level_map = {
        "Level 1": "level1",
        "Level 2": "level2",
        "Level 3": "level2",
        "Level 4": "level3",
        "Level 5": "level3",
    }

    for item in dataset:
        problem = item["problem"]
        solution = item["solution"]
        level = item.get("level", "Level 1")

        import re
        boxed_match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', solution)
        if boxed_match:
            answer = boxed_match.group(1)
        else:
            answer = solution.strip().split('\n')[-1]

        messages = [{"role": "user", "content": problem}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        difficulty = level_map.get(level, "level2")

        samples.append({
            "prompt": prompt,
            "answer": answer,
            "difficulty": difficulty,
            "problem": problem,
            "dataset_source": "MATH",
            "math_level": level,
            "math_type": item.get("type", "unknown")
        })

    return samples


def create_rlvr_gsm_eval(tokenizer, n_samples=200, seed=42):
    print(f"Loading RLVR-GSM dataset...")
    dataset = load_dataset("allenai/RLVR-GSM", split="train")

    dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

    print(f"Processing {len(dataset)} samples...")
    samples = []

    for item in dataset:
        messages = item.get("messages", [])
        if not messages:
            continue

        user_message = messages[0]
        content = user_message.get("content", "")

        if not content:
            continue

        formatted_messages = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        ground_truth = str(item.get("ground_truth", "")).strip()
        dataset_source = item.get("dataset", "gsm8k")

        samples.append({
            "prompt": prompt,
            "answer": ground_truth,
            "difficulty": "level1" if dataset_source == "gsm8k" else "level2",
            "problem": content,
            "dataset_source": dataset_source
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Create held-out evaluation dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gsm8k", "MATH", "rlvr-gsm"],
        help="Source dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path (e.g., held_out/gsm_eval.jsonl)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of samples to include"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-0.6B-Base",
        help="Tokenizer to use for chat template"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.dataset == "gsm8k":
        samples = create_gsm8k_eval(tokenizer, args.n_samples, args.seed)
    elif args.dataset == "MATH":
        samples = create_math_eval(tokenizer, args.n_samples, args.seed)
    elif args.dataset == "rlvr-gsm":
        samples = create_rlvr_gsm_eval(tokenizer, args.n_samples, args.seed)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Save to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(samples)} samples to {output_path}")
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f" Done! Created {output_path}")

    if samples:
        print(f"\nSample entry:")
        print(json.dumps(samples[0], indent=2))

        diff_counts = Counter(s.get("difficulty", "unknown") for s in samples)
        print(f"\nDifficulty distribution:")
        for diff, count in sorted(diff_counts.items()):
            pct = 100 * count / len(samples)
            print(f"  {diff}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
