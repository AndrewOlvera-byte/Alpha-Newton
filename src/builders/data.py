from src.core.registry import register
from datasets import load_dataset, interleave_datasets
from typing import Any, Dict, List


def _pack_tokenized_dataset(dataset, max_seq_len: int):
    """
    Concatenate and chunk tokenized sequences into fixed-length blocks.
    Ensures every sample is max_seq_len to better utilize the GPU.
    """

    def group_texts(examples):
        # Flatten then chunk to max_seq_len (keeps label masking intact)
        concatenated_ids = sum(examples["input_ids"], [])
        concatenated_labels = sum(examples["labels"], [])

        total_length = (len(concatenated_ids) // max_seq_len) * max_seq_len
        concatenated_ids = concatenated_ids[:total_length]
        concatenated_labels = concatenated_labels[:total_length]

        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        input_ids = [
            concatenated_ids[i : i + max_seq_len]
            for i in range(0, total_length, max_seq_len)
        ]
        labels = [
            concatenated_labels[i : i + max_seq_len]
            for i in range(0, total_length, max_seq_len)
        ]
        attention_mask = [[1] * max_seq_len for _ in input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    packed = dataset.map(
        group_texts,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return packed


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Normalize/clean message list into the minimal schema expected by chat templates:
      [{"role": "system"|"user"|"assistant", "content": str}, ...]
    """
    cleaned: List[Dict[str, str]] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if content is None:
            continue
        content = str(content).strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def _messages_from_sample(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert a dataset row (various formats) into OpenAI-style messages.
    Returns [] if unrecognized.
    """
    messages: List[Dict[str, Any]] = []

    # Parser 1: Standard OpenAI format (already correct)
    if sample.get("messages") and isinstance(sample["messages"], list):
        messages = sample["messages"]

    # Parser 2: Glaive (system + chat text) - must check before conversations
    elif sample.get("system") and sample.get("chat"):
        system_text = sample["system"]
        if system_text and "SYSTEM:" in system_text:
            system_content = system_text.split("SYSTEM:", 1)[1].strip()
            if system_content:
                messages.append({"role": "system", "content": system_content})

        chat_text = sample["chat"]
        current_role = None
        current_content: List[str] = []

        for chunk in str(chat_text).split("\n\n"):
            line = chunk.strip()
            if not line or line == "<|endoftext|>":
                continue

            if line.startswith("USER:"):
                if current_role and current_content:
                    messages.append(
                        {"role": current_role, "content": "\n".join(current_content).strip()}
                    )
                current_role = "user"
                current_content = [line.replace("USER:", "", 1).strip()]

            elif line.startswith("ASSISTANT:"):
                if current_role and current_content:
                    messages.append(
                        {"role": current_role, "content": "\n".join(current_content).strip()}
                    )
                current_role = "assistant"
                current_content = [line.replace("ASSISTANT:", "", 1).strip()]

            else:
                if current_role:
                    current_content.append(line)

        if current_role and current_content:
            messages.append({"role": current_role, "content": "\n".join(current_content).strip()})

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

    return _normalize_messages(messages)


def _tokenize_prompt_and_response(
    messages: List[Dict[str, str]],
    tokenizer,
    max_seq_len: int,
):
    """
    Turn a multi-turn chat into many SFT examples (one per assistant turn),
    where labels are masked (-100) for the prompt and active for the assistant response.

    Uses chat template tokenization to avoid double special-token insertion.
    """
    examples: List[Dict[str, Any]] = []

    if not messages:
        return examples

    trunc_side = getattr(tokenizer, "truncation_side", "right")

    def _extract_input_ids(tokenized):
        # `apply_chat_template(..., tokenize=True)` can return List[int] or an encoding/dict.
        if isinstance(tokenized, list):
            return tokenized
        if isinstance(tokenized, dict) and "input_ids" in tokenized:
            return tokenized["input_ids"]
        if hasattr(tokenized, "input_ids"):
            return getattr(tokenized, "input_ids")
        return None

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if i == 0:
            continue

        prefix = messages[:i]
        full = messages[: i + 1]

        try:
            prompt_ids = tokenizer.apply_chat_template(
                prefix,
                tokenize=True,
                add_generation_prompt=True,
            )
            full_ids = tokenizer.apply_chat_template(
                full,
                tokenize=True,
                add_generation_prompt=False,
            )
        except Exception:
            continue

        prompt_ids = _extract_input_ids(prompt_ids)
        full_ids = _extract_input_ids(full_ids)

        if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
            continue
        if len(full_ids) == 0:
            continue

        # Safety: ensure prompt is a prefix of full (expected for chat templates)
        if len(prompt_ids) > len(full_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
            continue

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

        # Truncate consistently (keep labels aligned)
        if len(full_ids) > max_seq_len:
            if trunc_side == "left":
                full_ids = full_ids[-max_seq_len:]
                labels = labels[-max_seq_len:]
            else:
                full_ids = full_ids[:max_seq_len]
                labels = labels[:max_seq_len]

        attention_mask = [1] * len(full_ids)
        if len(full_ids) != len(labels):
            continue

        examples.append(
            {
                "input_ids": full_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    return examples


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

    # === 2. Convert rows -> (prompt + single assistant response) tokenized examples === #

    def to_sft_features(batch):
        input_ids, attention_mask, labels = [], [], []

        batch_size = len(next(iter(batch.values()))) if batch else 0
        for idx in range(batch_size):
            sample = {k: v[idx] for k, v in batch.items()}
            messages = _messages_from_sample(sample)
            exs = _tokenize_prompt_and_response(messages, tokenizer=tokenizer, max_seq_len=max_seq_len)
            for ex in exs:
                input_ids.append(ex["input_ids"])
                attention_mask.append(ex["attention_mask"])
                labels.append(ex["labels"])

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train = train.map(
        to_sft_features,
        batched=True,
        num_proc=num_proc,
        remove_columns=train.column_names,
    )
    eval = eval.map(
        to_sft_features,
        batched=True,
        num_proc=num_proc,
        remove_columns=eval.column_names,
    )

    # === 4. Optional packing === #

    if packing:
        train = _pack_tokenized_dataset(train, max_seq_len)
        eval = _pack_tokenized_dataset(eval, max_seq_len)

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

    def to_sft_features(batch):
        input_ids, attention_mask, labels = [], [], []

        batch_size = len(next(iter(batch.values()))) if batch else 0
        for idx in range(batch_size):
            sample = {k: v[idx] for k, v in batch.items()}
            messages = _messages_from_sample(sample)
            exs = _tokenize_prompt_and_response(messages, tokenizer=tokenizer, max_seq_len=max_seq_len)
            for ex in exs:
                input_ids.append(ex["input_ids"])
                attention_mask.append(ex["attention_mask"])
                labels.append(ex["labels"])

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train = train.map(
        to_sft_features,
        batched=True,
        num_proc=num_proc,
        remove_columns=train.column_names,
    )
    eval = eval.map(
        to_sft_features,
        batched=True,
        num_proc=num_proc,
        remove_columns=eval.column_names,
    )

    # Packing to fixed blocks for better GPU utilization
    if packing:
        train = _pack_tokenized_dataset(train, max_seq_len)
        eval = _pack_tokenized_dataset(eval, max_seq_len)

    print(f"[Mixed Dataset] ✓ Train: {len(train):,} | Eval: {len(eval):,}\n")

    # Set torch format for efficient collation
    columns = ["input_ids", "attention_mask", "labels"]
    train = train.with_format("torch", columns=columns)
    eval = eval.with_format("torch", columns=columns)

    return {"train": train, "eval": eval}


@register("data", "rlvr_math")
def build_rlvr_math_dataset(
    source: str,
    train_path: str,
    max_prompt_len: int,
    num_proc: int,
    tokenizer,
    eval_path: str = None,
    cache_dir: str = None,
    subset_pct: float = 100.0,
    **kwargs
):
    """
    Build math dataset for RLVR (online GRPO training).
    
    Returns dataset with:
    - prompt: Formatted prompt string for generation
    - answer: Ground truth answer for reward verification
    
    Supports:
    - DeepMath-103K: question, final_answer columns
    - GSM8K: question, answer columns (extracts number after ####)
    - Custom: expects 'question' and 'answer' columns
    """
    assert source in ("hf", "local"), f"Invalid source: {source}"
    
    # === 1. Load dataset === #
    if source == "hf":
        dataset = load_dataset(train_path, split="train", cache_dir=cache_dir)
    else:
        dataset = load_dataset("json", data_files=train_path, split="train", cache_dir=cache_dir)
    
    # Optional subset for testing
    if subset_pct < 100.0:
        n_samples = int(len(dataset) * subset_pct / 100)
        dataset = dataset.select(range(n_samples))
        print(f"[RLVR Math] Using {subset_pct}% subset: {n_samples:,} samples")
    
    # Split train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train = split["train"]
    eval_ds = split["test"]
    
    print(f"[RLVR Math] Dataset: {train_path}")
    print(f"[RLVR Math] Train: {len(train):,} | Eval: {len(eval_ds):,}")
    
    # === 2. Detect format and extract question/answer === #
    
    def format_sample(sample):
        """
        Convert various math dataset formats to GRPO format.
        Returns: {prompt: str, answer: str}
        """
        # Detect question field
        if "question" in sample:
            question = sample["question"]
        elif "problem" in sample:
            question = sample["problem"]
        else:
            raise ValueError(f"No question field found. Keys: {list(sample.keys())}")
        
        # Detect answer field
        if "final_answer" in sample:
            # DeepMath-103K format
            answer = str(sample["final_answer"]).strip()
        elif "answer" in sample:
            raw_answer = sample["answer"]
            # GSM8K format: "... #### 42"
            if "####" in str(raw_answer):
                answer = str(raw_answer).split("####")[-1].strip()
            else:
                answer = str(raw_answer).strip()
        else:
            raise ValueError(f"No answer field found. Keys: {list(sample.keys())}")
        
        # Format as chat prompt
        messages = [
            {"role": "user", "content": question}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # Add assistant prefix for generation
        )
        
        return {"prompt": prompt, "answer": answer}
    
    train = train.map(format_sample, num_proc=num_proc)
    eval_ds = eval_ds.map(format_sample, num_proc=num_proc)
    
    # Keep only required columns for GRPO
    train = train.select_columns(["prompt", "answer"])
    eval_ds = eval_ds.select_columns(["prompt", "answer"])
    
    print(f"[RLVR Math] ✓ Formatted for GRPO (prompt + answer columns)")
    
    return {"train": train, "eval": eval_ds}
