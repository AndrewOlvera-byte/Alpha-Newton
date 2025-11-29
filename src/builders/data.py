from src.core.registry import register
from datasets import load_dataset, interleave_datasets


def _pack_tokenized_dataset(dataset, max_seq_len: int):
    """
    Concatenate and chunk tokenized sequences into fixed-length blocks.
    Ensures every sample is max_seq_len to better utilize the GPU.
    """

    def group_texts(examples):
        # Flatten then chunk to max_seq_len
        concatenated = sum(examples["input_ids"], [])
        total_length = (len(concatenated) // max_seq_len) * max_seq_len
        concatenated = concatenated[:total_length]

        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        input_ids = [
            concatenated[i : i + max_seq_len]
            for i in range(0, total_length, max_seq_len)
        ]
        attention_mask = [[1] * max_seq_len for _ in input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }

    packed = dataset.map(
        group_texts,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return packed


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
        Convert various dataset formats to universal OpenAI messages format,
        then apply tokenizer's chat template.

        Supports:
        - Standard "messages": [{"role": "user", "content": "..."}]
        - Capybara "conversation": [{"input": "...", "output": "..."}]
        - OpenHermes "conversations": [{"from": "human/gpt", "value": "..."}]
        """
        messages = []

        # Standard OpenAI format
        if "messages" in sample:
            messages = sample["messages"]

        # Capybara conversation format
        elif "conversation" in sample:
            for turn in sample["conversation"]:
                messages.append({"role": "user", "content": turn["input"]})
                messages.append({"role": "assistant", "content": turn["output"]})

        # OpenHermes-2.5 format
        elif "conversations" in sample:
            # Add system prompt if exists
            if sample.get("system_prompt"):
                messages.append({"role": "system", "content": sample["system_prompt"]})

            # Convert conversations
            for turn in sample["conversations"]:
                role = "user" if turn["from"] in ["human", "user"] else "assistant"
                messages.append({"role": role, "content": turn["value"]})

        else:
            raise ValueError(f"Unknown format. Expected 'messages', 'conversation', or 'conversations', got: {list(sample.keys())}")

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

    # Remove raw text to keep only tensor-ready columns
    keep_cols = {"input_ids", "attention_mask"}
    train = train.remove_columns([c for c in train.column_names if c not in keep_cols])
    eval = eval.remove_columns([c for c in eval.column_names if c not in keep_cols])

    # === 4. Optional packing === #

    if packing:
        train = _pack_tokenized_dataset(train, max_seq_len)
        eval = _pack_tokenized_dataset(eval, max_seq_len)
    else:
        # Add labels to mirror input_ids for trainer/collator
        def add_labels(batch):
            return {"labels": batch["input_ids"]}

        train = train.map(add_labels, batched=True)
        eval = eval.map(add_labels, batched=True)

    # Ensure PyTorch format for efficient collation
    columns = ["input_ids", "attention_mask", "labels"]
    train = train.with_format("torch", columns=columns)
    eval = eval.with_format("torch", columns=columns)

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
        """
        Convert various dataset formats to universal OpenAI messages format.

        Uses smart detection to handle any dataset format without hardcoding checks.
        """
        messages = []

        # Strategy: Try each parser in order, return first successful parse

        # Parser 1: Standard OpenAI format (already correct)
        if sample.get("messages") and isinstance(sample["messages"], list):
            messages = sample["messages"]

        # Parser 2: Glaive (system + chat text) - must check before conversations
        elif sample.get("system") and sample.get("chat"):
            system_text = sample["system"]
            if system_text and "SYSTEM:" in system_text:
                system_content = system_text.split("SYSTEM:", 1)[1].strip()
                messages.append({"role": "system", "content": system_content})

            chat_text = sample["chat"]
            current_role = None
            current_content = []

            for line in chat_text.split("\n\n"):
                line = line.strip()
                if not line or line == "<|endoftext|>":
                    continue

                if line.startswith("USER:"):
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": "\n".join(current_content).strip()
                        })
                    current_role = "user"
                    current_content = [line.replace("USER:", "").strip()]

                elif line.startswith("ASSISTANT:"):
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": "\n".join(current_content).strip()
                        })
                    current_role = "assistant"
                    current_content = [line.replace("ASSISTANT:", "").strip()]

                else:
                    if current_role:
                        current_content.append(line)

            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip()
                })

        # Parser 3: OpenHermes (conversations list with from/value)
        elif sample.get("conversations") and isinstance(sample.get("conversations"), list):
            if sample.get("system_prompt"):
                messages.append({"role": "system", "content": sample["system_prompt"]})

            for turn in sample["conversations"]:
                if isinstance(turn, dict) and "from" in turn and "value" in turn:
                    role = "user" if turn["from"] in ["human", "user"] else "assistant"
                    messages.append({"role": role, "content": turn["value"]})

        # Parser 4: Capybara (conversation list with input/output)
        elif sample.get("conversation") and isinstance(sample.get("conversation"), list):
            for turn in sample["conversation"]:
                if isinstance(turn, dict) and "input" in turn and "output" in turn:
                    messages.append({"role": "user", "content": turn["input"]})
                    messages.append({"role": "assistant", "content": turn["output"]})

        # If no parser worked, skip this sample (don't crash)
        if not messages:
            return {"text": ""}

        # Apply chat template
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            # Fallback: skip samples that fail chat template
            return {"text": ""}

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

    # Keep only token columns
    keep_cols = {"input_ids", "attention_mask"}
    train = train.remove_columns([c for c in train.column_names if c not in keep_cols])
    eval = eval.remove_columns([c for c in eval.column_names if c not in keep_cols])

    # Packing to fixed blocks for better GPU utilization
    if packing:
        train = _pack_tokenized_dataset(train, max_seq_len)
        eval = _pack_tokenized_dataset(eval, max_seq_len)
    else:
        def add_labels(batch):
            return {"labels": batch["input_ids"]}

        train = train.map(add_labels, batched=True)
        eval = eval.map(add_labels, batched=True)

    print(f"[Mixed Dataset] ✓ Train: {len(train):,} | Eval: {len(eval):,}\n")

    # Set torch format for efficient collation
    columns = ["input_ids", "attention_mask", "labels"]
    train = train.with_format("torch", columns=columns)
    eval = eval.with_format("torch", columns=columns)

    return {"train": train, "eval": eval}
