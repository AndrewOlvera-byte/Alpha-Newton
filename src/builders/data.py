from __future__ import annotations

from collections import Counter
from typing import Any

from datasets import load_dataset

from src.core.registry import register
from src.rlvr.math_verifier import _classify_answer_type


@register("data", "rlvr_allenai")
def build_rlvr_allenai_dataset(
    source: str,
    train_path: str,
    max_prompt_len: int,
    num_proc: int,
    tokenizer,
    eval_path: str | None = None,
    cache_dir: str | None = None,
    subset_pct: float = 100.0,
    name: str | None = None,
    classify_answer_types: bool = True,
    exclude_datasets: list[str] | None = None,
    **_: Any,
):
    """Load an AllenAI RLVR dataset and format it for TRL's GRPO trainer."""
    if source == "hf":
        dataset = load_dataset(train_path, name=name, split="train", cache_dir=cache_dir)
    elif source == "local":
        dataset = load_dataset("json", data_files=train_path, split="train", cache_dir=cache_dir)
    else:
        raise ValueError(f"Unsupported dataset source: {source}")

    if exclude_datasets:
        excluded = set(exclude_datasets)
        dataset = dataset.filter(
            lambda sample: sample.get("dataset", "") not in excluded,
            num_proc=num_proc,
            desc="Filtering datasets",
        )

    if not 0 < subset_pct <= 100:
        raise ValueError("subset_pct must be in the interval (0, 100]")
    if subset_pct < 100:
        dataset = dataset.select(range(int(len(dataset) * subset_pct / 100)))

    split = dataset.train_test_split(test_size=0.05, seed=42)

    def format_sample(sample: dict[str, Any]) -> dict[str, Any]:
        messages = sample.get("messages") or []
        content = messages[0].get("content", "") if messages else ""
        if not content:
            raise ValueError("RLVR samples must contain a non-empty user message")

        answer = str(sample.get("ground_truth", "")).strip()
        source_name = sample.get("dataset", "unknown")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        result = {
            "prompt": prompt,
            "answer": answer,
            "dataset_source": source_name,
            "metadata": str(
                {
                    "dataset": source_name,
                    "constraint_type": sample.get("constraint_type"),
                    "constraint": sample.get("constraint"),
                }
            ),
        }
        if classify_answer_types:
            result["answer_type"] = _classify_answer_type(answer)
        return result

    keep = ["prompt", "answer", "dataset_source", "metadata"]
    if classify_answer_types:
        keep.append("answer_type")

    formatted = {}
    for split_name, split_dataset in split.items():
        formatted[split_name] = split_dataset.map(
            format_sample,
            num_proc=num_proc,
            desc=f"Formatting {split_name}",
        ).select_columns(keep)

    distribution = Counter(formatted["train"]["dataset_source"])
    print(f"[RLVR] Train: {len(formatted['train']):,} | Eval: {len(formatted['test']):,}")
    print(f"[RLVR] Dataset distribution: {dict(sorted(distribution.items()))}")
    return {"train": formatted["train"], "eval": formatted["test"]}
