#!/usr/bin/env python3
"""Build the deterministic GSM checkpoint-selection set used by the study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def build_eval_set(tokenizer, sample_count: int, seed: int) -> list[dict]:
    dataset = load_dataset("allenai/RLVR-GSM", split="train")
    dataset = dataset.shuffle(seed=seed).select(range(min(sample_count, len(dataset))))

    samples = []
    for item in dataset:
        messages = item.get("messages") or []
        question = messages[0].get("content", "") if messages else ""
        if not question:
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        samples.append(
            {
                "prompt": prompt,
                "answer": str(item.get("ground_truth", "")).strip(),
                "difficulty": "level1",
                "problem": question,
                "dataset_source": item.get("dataset", "gsm8k"),
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="held_out/gsm_eval_v2.jsonl")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    samples = build_eval_set(tokenizer, args.n_samples, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for sample in samples:
            stream.write(json.dumps(sample) + "\n")
    print(f"Wrote {len(samples)} samples to {output}")


if __name__ == "__main__":
    main()
