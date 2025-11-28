from src.core.registry import register
from datasets import load_dataset, interleave_datasets

@register("data", "sft")
def build_sft_dataset(source: str, train_path: str, eval_path: str, max_seq_len: int, num_proc: int, packing: bool, tokenizer, cache_dir: str = None):

    assert source in ("hf", "local", "jsonl", "json"), \
        f"Invalid source: {source}"

    # === 1. Load raw dataset === #

    if source == "hf":
        # Load dataset
        dataset = load_dataset(train_path, split="train", cache_dir=cache_dir)

        # If train_path == eval_path, split it ourselves
        if train_path == eval_path:
            # 95% train, 5% eval split
            split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
            train = split_dataset["train"]
            eval = split_dataset["test"]
        else:
            # Separate datasets specified
            train = dataset
            eval = load_dataset(eval_path, split="train", cache_dir=cache_dir)

    elif source == "local":
        train = load_dataset(
            "json",
            data_files=train_path,
            split="train",
            cache_dir=cache_dir,
        )

        eval = load_dataset(
            "json",
            data_files=eval_path,
            split="train",
            cache_dir=cache_dir,
        )

    # === 2. Format chat with tokenizer's chat template === #

    def format_chat(sample):
        """
        Convert various dataset formats to chat messages format.

        Supports:
        1. Standard "messages" format: [{"role": "user", "content": "..."}]
        2. Capybara "conversation" format: [{"input": "...", "output": "..."}]
        """
        # Check if already in messages format
        if "messages" in sample:
            messages = sample["messages"]

        # Convert Capybara conversation format
        elif "conversation" in sample:
            messages = []
            for turn in sample["conversation"]:
                messages.append({"role": "user", "content": turn["input"]})
                messages.append({"role": "assistant", "content": turn["output"]})

        else:
            raise ValueError(f"Unknown dataset format. Expected 'messages' or 'conversation', got: {list(sample.keys())}")

        # Apply tokenizer's chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train = train.map(format_chat, num_proc=num_proc)
    eval = eval.map(format_chat, num_proc=num_proc)

    # === 3. Tokenize (fast batched tokenization) === #

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            max_length=max_seq_len,
            truncation=True,
            padding=False,
        )

    train = train.map(tokenize, batched=True, num_proc=num_proc)
    eval = eval.map(tokenize, batched=True, num_proc=num_proc)

    # === 4. Optional packing === #

    if packing:
        # Sort by sequence length for efficient packing
        # Can't sort on list column directly, so add length column
        def add_length(sample):
            return {"length": len(sample["input_ids"])}

        train = train.map(add_length)
        train = train.sort("length")
        train = train.remove_columns(["length"])  # Clean up

    return {"train": train, "eval": eval}


@register("data", "dpo")
def build_dpo_dataset(source: str, train_path: str, eval_path: str, max_seq_len: int, num_proc: int, tokenizer, cache_dir: str = None, **kwargs):
    """
    Build DPO dataset from preference pairs

    Supports datasets with (chosen, rejected) pairs:
        - Anthropic/hh-rlhf
        - HuggingFaceH4/ultrafeedback_binarized
        - Intel/orca_dpo_pairs
        - argilla/ultrafeedback-binarized-preferences-cleaned

    Expected format:
        - chosen: preferred response
        - rejected: dis-preferred response
        - (optional) prompt: the input prompt
    """
    assert source in ("hf", "local", "jsonl"), f"Invalid source: {source}"

    # === 1. Load raw dataset === #

    if source == "hf":
        dataset = load_dataset(train_path, split="train", cache_dir=cache_dir)

        if train_path == eval_path:
            split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
            train = split_dataset["train"]
            eval = split_dataset["test"]
        else:
            train = dataset
            eval = load_dataset(eval_path, split="train", cache_dir=cache_dir)

    elif source == "local":
        train = load_dataset("json", data_files=train_path, split="train", cache_dir=cache_dir)
        eval = load_dataset("json", data_files=eval_path, split="train", cache_dir=cache_dir)

    # === 2. Format for DPO === #

    def format_dpo_sample(sample):
        """
        Convert to DPO format expected by TRL DPOTrainer

        TRL expects: {"prompt": str, "chosen": str, "rejected": str}
        """
        # Anthropic HH-RLHF format: "\n\nH: ...\n\nA: ..."
        if "chosen" in sample and "rejected" in sample and "prompt" not in sample:
            chosen = sample["chosen"]
            rejected = sample["rejected"]

            # Extract prompt from conversation (everything before last assistant response)
            if "\n\nH:" in chosen and "\n\nA:" in chosen:
                parts = chosen.split("\n\nA:")
                if len(parts) > 1:
                    prompt = parts[0] + "\n\nA:"  # Include "A:" prefix for completion
                    chosen_response = parts[-1].strip()

                    rejected_parts = rejected.split("\n\nA:")
                    rejected_response = rejected_parts[-1].strip() if len(rejected_parts) > 1 else rejected

                    return {
                        "prompt": prompt,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                    }

            # Fallback: use full conversations
            return {
                "prompt": "",
                "chosen": chosen,
                "rejected": rejected,
            }

        # Already in correct format
        elif all(k in sample for k in ["prompt", "chosen", "rejected"]):
            return sample

        else:
            raise ValueError(
                f"Unknown DPO format. Expected 'chosen/rejected' or 'prompt/chosen/rejected', "
                f"got: {list(sample.keys())}"
            )

    train = train.map(format_dpo_sample, num_proc=num_proc)
    eval = eval.map(format_dpo_sample, num_proc=num_proc)

    # Keep only required columns
    required_cols = ["prompt", "chosen", "rejected"]
    train = train.select_columns(required_cols)
    eval = eval.select_columns(required_cols)

    return {"train": train, "eval": eval}


@register("data", "sft_mixed")
def build_mixed_sft_dataset(
    datasets_config: list,
    max_seq_len: int,
    num_proc: int,
    packing: bool,
    tokenizer,
    cache_dir: str = None,
    **kwargs
):
    """
    Build mixed SFT dataset from multiple sources with sampling weights

    Example config:
        data:
          type: "sft_mixed"
          datasets_config:
            - path: "teknium/OpenHermes-2.5"
              weight: 0.7
            - path: "glaiveai/glaive-function-calling-v2"
              weight: 0.3
    """
    train_datasets = []
    eval_datasets = []
    probabilities = []

    print(f"\n[Mixed Dataset] Loading {len(datasets_config)} datasets:")

    for i, ds_config in enumerate(datasets_config):
        path = ds_config["path"]
        weight = ds_config.get("weight", 1.0)

        print(f"  [{i+1}] {path} (weight: {weight})")

        # Load dataset
        dataset = load_dataset(path, split="train", cache_dir=cache_dir)

        # Split into train/eval (95/5)
        split_dataset = dataset.train_test_split(test_size=0.05, seed=42)

        train_datasets.append(split_dataset["train"])
        eval_datasets.append(split_dataset["test"])
        probabilities.append(weight)

    # Normalize probabilities
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    print(f"[Mixed Dataset] Sampling: {[f'{p:.1%}' for p in probabilities]}\n")

    # Interleave with probabilities
    train = interleave_datasets(
        train_datasets,
        probabilities=probabilities,
        seed=42,
        stopping_strategy="all_exhausted",
    )

    eval = interleave_datasets(
        eval_datasets,
        probabilities=probabilities,
        seed=42,
        stopping_strategy="first_exhausted",
    )

    # === Format and tokenize === #

    def format_chat(sample):
        """Handle multiple dataset formats"""
        if "messages" in sample:
            messages = sample["messages"]
        elif "conversation" in sample:
            messages = []
            for turn in sample["conversation"]:
                messages.append({"role": "user", "content": turn["input"]})
                messages.append({"role": "assistant", "content": turn["output"]})
        elif "conversations" in sample:
            messages = sample["conversations"]
        else:
            raise ValueError(f"Unknown format: {list(sample.keys())}")

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train = train.map(format_chat, num_proc=num_proc)
    eval = eval.map(format_chat, num_proc=num_proc)

    # Tokenize
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            max_length=max_seq_len,
            truncation=True,
            padding=False,
        )

    train = train.map(tokenize, batched=True, num_proc=num_proc)
    eval = eval.map(tokenize, batched=True, num_proc=num_proc)

    # Packing
    if packing:
        def add_length(sample):
            return {"length": len(sample["input_ids"])}

        train = train.map(add_length)
        train = train.sort("length")
        train = train.remove_columns(["length"])

    print(f"[Mixed Dataset] ✓ Train: {len(train):,} | Eval: {len(eval):,}\n")

    return {"train": train, "eval": eval}
