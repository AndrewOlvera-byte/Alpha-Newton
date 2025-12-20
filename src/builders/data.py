import re
from typing import Any, Dict, List

from datasets import load_dataset, interleave_datasets

from src.core.registry import register

def _strip_think_tokens(text: str) -> str:
      text = re.sub(r'<think>\s*', '', text, flags=re.IGNORECASE)
      text = re.sub(r'\s*</think>', '', text, flags=re.IGNORECASE)
      return text.strip()


def _pack_tokenized_dataset(dataset, max_seq_len: int, eos_token_id: int):

    def group_texts(examples):
        concatenated_ids = []
        concatenated_labels = []

        for ids, labs in zip(examples["input_ids"], examples["labels"]):
            concatenated_ids.extend(ids)
            concatenated_labels.extend(labs)
            
            concatenated_ids.append(eos_token_id)
            concatenated_labels.append(eos_token_id)

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
        content = _strip_think_tokens(str(content))
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def _parse_glaive_chat(system_text: str, chat_text: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    
    if system_text:
        sys_match = re.match(r"(?:SYSTEM:\s*)?(.+)", str(system_text), re.IGNORECASE | re.DOTALL)
        if sys_match:
            sys_content = sys_match.group(1).strip()
            if sys_content:
                messages.append({"role": "system", "content": sys_content})
    
    if not chat_text:
        return messages
    
    pattern = r'\n*(?=(?:USER|ASSISTANT|FUNCTION\s*RESPONSE)\s*:)'
    chunks = re.split(pattern, str(chat_text), flags=re.IGNORECASE)
    
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or chunk == "<|endoftext|>":
            continue
        
        chunk_upper = chunk.upper()
        
        if chunk_upper.startswith("USER:"):
            content = chunk[5:].strip()
            if content:
                messages.append({"role": "user", "content": content})
        
        elif chunk_upper.startswith("ASSISTANT:"):
            content = chunk[10:].strip()
            if content:
                messages.append({"role": "assistant", "content": content})
        
        elif chunk_upper.startswith("FUNCTION RESPONSE:") or chunk_upper.startswith("FUNCTION_RESPONSE:"):
            colon_idx = chunk.find(":")
            content = chunk[colon_idx + 1:].strip() if colon_idx != -1 else chunk.strip()
            if content:
                messages.append({"role": "user", "content": f"[Function Response]\n{content}"})
    
    return messages


def _messages_from_sample(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert a dataset row (various formats) into OpenAI-style messages.
    Returns [] if unrecognized.
    
    Supported formats:
        1. OpenAI format: {"messages": [{"role": "...", "content": "..."}]}
           Used by: NuminaMath-CoT, big-reasoning-traces, general chat datasets
        
        2. ShareGPT format: {"conversations": [{"from": "...", "value": "..."}]}
           Used by: OpenHermes-2.5, OpenThoughts3-1.2M
        
        3. CoT-Collection format: {"source": "...", "rationale": "...", "target": "..."}
           Used by: kaist-ai/CoT-Collection
        
        4. Glaive format: {"system": "SYSTEM: ...", "chat": "USER: ... ASSISTANT: ..."}
           Used by: glaiveai/glaive-function-calling-v2
        
        5. Capybara format: {"conversation": [{"input": "...", "output": "..."}]}
           Used by: LDJnr/Capybara
        
        6. Problem/Solution format: {"problem": "...", "solution": "..."}
           Used by: Various math datasets as fallback
    """
    messages: List[Dict[str, Any]] = []

    # === Parser 1: OpenAI format (messages with role/content) ===
    # Used by: NuminaMath-CoT, big-reasoning-traces, most modern datasets
    if sample.get("messages") and isinstance(sample["messages"], list):
        messages = sample["messages"]

    # === Parser 2: Glaive format (system + chat plain text) ===
    # Check before conversations since Glaive samples may have empty conversations field
    elif sample.get("chat") and isinstance(sample.get("chat"), str):
        system_text = sample.get("system", "")
        messages = _parse_glaive_chat(system_text, sample["chat"])

    # === Parser 3: ShareGPT format (conversations with from/value) ===
    # Used by: OpenHermes-2.5, OpenThoughts3-1.2M
    elif sample.get("conversations") and isinstance(sample["conversations"], list):
        # Check for external system_prompt field first
        if sample.get("system_prompt"):
            messages.append({"role": "system", "content": sample["system_prompt"]})

        for turn in sample["conversations"]:
            if not isinstance(turn, dict):
                continue
            
            # Get role: normalize "human"/"user" -> "user", "gpt"/"assistant" -> "assistant"
            from_field = str(turn.get("from", "")).lower()
            value = turn.get("value")
            
            if not value:
                continue
            
            if from_field in ("human", "user"):
                messages.append({"role": "user", "content": value})
            elif from_field in ("gpt", "assistant"):
                messages.append({"role": "assistant", "content": value})
            elif from_field == "system":
                # System message embedded in conversations (OpenThoughts3 style)
                messages.append({"role": "system", "content": value})

    # === Parser 4: CoT-Collection format (source/rationale/target) ===
    # Used by: kaist-ai/CoT-Collection
    elif sample.get("source") and sample.get("target"):
        source = str(sample["source"]).strip()
        target = str(sample["target"]).strip()
        rationale = sample.get("rationale", "")
        
        if source:
            messages.append({"role": "user", "content": source})
        
        if rationale and target:
            # Combine rationale (CoT) with final answer
            rationale = str(rationale).strip()
            response = f"{rationale}\n\nThe answer is: {target}" if rationale else target
            messages.append({"role": "assistant", "content": response})
        elif target:
            messages.append({"role": "assistant", "content": target})

    # === Parser 5: Capybara format (conversation with input/output) ===
    # Used by: LDJnr/Capybara
    elif sample.get("conversation") and isinstance(sample["conversation"], list):
        for turn in sample["conversation"]:
            if isinstance(turn, dict) and "input" in turn and "output" in turn:
                inp = str(turn["input"]).strip()
                out = str(turn["output"]).strip()
                if inp:
                    messages.append({"role": "user", "content": inp})
                if out:
                    messages.append({"role": "assistant", "content": out})

    # === Parser 6: Problem/Solution fallback ===
    # Used by: Various math datasets
    elif sample.get("problem") and sample.get("solution"):
        problem = str(sample["problem"]).strip()
        solution = str(sample["solution"]).strip()
        if problem:
            messages.append({"role": "user", "content": problem})
        if solution:
            messages.append({"role": "assistant", "content": solution})

    return _normalize_messages(messages)


def _tokenize_prompt_and_response(
    messages: List[Dict[str, str]],
    tokenizer,
    max_seq_len: int,
):
    examples: List[Dict[str, Any]] = []

    if not messages:
        return examples

    trunc_side = getattr(tokenizer, "truncation_side", "right")

    def _extract_input_ids(tokenized):
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

        if len(prompt_ids) > len(full_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
            continue

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

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
        dataset = load_dataset(train_path, split="train", cache_dir=cache_dir)

        if train_path == eval_path:
            split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
            train = split_dataset["train"]
            eval = split_dataset["test"]
        else:
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
        train = _pack_tokenized_dataset(train, max_seq_len, tokenizer.eos_token_id)
        eval = _pack_tokenized_dataset(eval, max_seq_len, tokenizer.eos_token_id)

    # Ensure PyTorch format for efficient collation
    columns = ["input_ids", "attention_mask", "labels"]
    train = train.with_format("torch", columns=columns)
    eval = eval.with_format("torch", columns=columns)

    return {"train": train, "eval": eval}


def _parse_anthropic_hh(sample):
    """
    Parse Anthropic HH-RLHF format into prompt/chosen/rejected.
    
    Format: "\n\nHuman: ...\n\nAssistant: ..."
    Returns dict with prompt, chosen, rejected strings.
    """
    chosen = sample["chosen"]
    rejected = sample["rejected"]

    if "\n\nHuman:" in chosen and "\n\nAssistant:" in chosen:
        parts = chosen.split("\n\nAssistant:")
        if len(parts) > 1:
            prompt = "\n\nAssistant:".join(parts[:-1]) + "\n\nAssistant:"
            chosen_response = parts[-1].strip()

            rejected_parts = rejected.split("\n\nAssistant:")
            rejected_response = rejected_parts[-1].strip() if len(rejected_parts) > 1 else rejected

            return {
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
            }

    return {
        "prompt": "",
        "chosen": chosen,
        "rejected": rejected,
    }


@register("data", "dpo")
def build_dpo_dataset(
    source: str,
    train_path: str,
    eval_path: str,
    max_seq_len: int,
    num_proc: int,
    tokenizer,
    cache_dir: str = None,
    **kwargs
):
    """
    Build DPO dataset from preference pairs.

    Supports datasets with (chosen, rejected) pairs:
        - Anthropic/hh-rlhf
        - HuggingFaceH4/ultrafeedback_binarized
        - Intel/orca_dpo_pairs
        - argilla/ultrafeedback-binarized-preferences-cleaned

    Args:
        source: "hf" or "local"
        train_path: HuggingFace dataset name or local path
        eval_path: HuggingFace dataset name or local path
        max_seq_len: Maximum sequence length (used by trainer, not here)
        num_proc: Number of processes for parallel processing
        tokenizer: Tokenizer instance
        cache_dir: Cache directory for datasets

    Returns:
        Dict with 'train' and 'eval' datasets containing prompt/chosen/rejected text columns
        TRL DPOTrainer handles tokenization internally
    """
    assert source in ("hf", "local", "jsonl"), f"Invalid source: {source}"

    # === 1. Load raw dataset === #

    if source == "hf":
        dataset = load_dataset(train_path, split="train", cache_dir=cache_dir)

        if train_path == eval_path:
            split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
            train = split_dataset["train"]
            eval_ds = split_dataset["test"]
        else:
            train = dataset
            eval_ds = load_dataset(eval_path, split="train", cache_dir=cache_dir)

    elif source == "local":
        train = load_dataset("json", data_files=train_path, split="train", cache_dir=cache_dir)
        eval_ds = load_dataset("json", data_files=eval_path, split="train", cache_dir=cache_dir)

    print(f"[DPO Data] Loaded {len(train):,} train, {len(eval_ds):,} eval samples")

    # === 2. Format for DPO === #

    def format_dpo_sample(sample):
        """Convert to DPO format: {"prompt": str, "chosen": str, "rejected": str}"""
        if "chosen" in sample and "rejected" in sample and "prompt" not in sample:
            return _parse_anthropic_hh(sample)

        elif all(k in sample for k in ["prompt", "chosen", "rejected"]):
            return {
                "prompt": sample["prompt"],
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
            }

        else:
            raise ValueError(
                f"Unknown DPO format. Expected 'chosen/rejected' or 'prompt/chosen/rejected', "
                f"got: {list(sample.keys())}"
            )

    train = train.map(format_dpo_sample, num_proc=num_proc, desc="Formatting train")
    eval_ds = eval_ds.map(format_dpo_sample, num_proc=num_proc, desc="Formatting eval")

    required_cols = ["prompt", "chosen", "rejected"]
    train = train.select_columns(required_cols)
    eval_ds = eval_ds.select_columns(required_cols)

    print(f"[DPO Data] Formatted: {len(train):,} train, {len(eval_ds):,} eval")

    return {"train": train, "eval": eval_ds}


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
        name = ds_config.get("name", None)  # Config name (e.g., "DeepSeek")
        trust_remote_code = ds_config.get("trust_remote_code", False)

        print(f"  [{i+1}] {path} (weight: {weight})")

        dataset = load_dataset(
            path,
            name=name,
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )

        split_dataset = dataset.train_test_split(test_size=0.05, seed=42)

        train_datasets.append(split_dataset["train"])
        eval_datasets.append(split_dataset["test"])
        probabilities.append(weight)

    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    print(f"[Mixed Dataset] Sampling: {[f'{p:.1%}' for p in probabilities]}\n")

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

    if packing:
        train = _pack_tokenized_dataset(train, max_seq_len, tokenizer.eos_token_id)
        eval = _pack_tokenized_dataset(eval, max_seq_len, tokenizer.eos_token_id)

    print(f"[Mixed Dataset] Train: {len(train):,} | Eval: {len(eval):,}\n")

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
    
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train = split["train"]
    eval_ds = split["test"]
    
    print(f"[RLVR Math] Dataset: {train_path}")
    print(f"[RLVR Math] Train: {len(train):,} | Eval: {len(eval_ds):,}")
    
    # === 2. Detect format and extract question/answer === #
    
    def format_sample(sample):
        if "question" in sample:
            question = sample["question"]
        elif "problem" in sample:
            question = sample["problem"]
        else:
            raise ValueError(f"No question field found. Keys: {list(sample.keys())}")
        
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
        
        messages = [
            {"role": "user", "content": question}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        return {"prompt": prompt, "answer": answer}
    
    train = train.map(format_sample, num_proc=num_proc)
    eval_ds = eval_ds.map(format_sample, num_proc=num_proc)
    
    train = train.select_columns(["prompt", "answer"])
    eval_ds = eval_ds.select_columns(["prompt", "answer"])
    
    print(f"[RLVR Math] Formatted for GRPO (prompt + answer columns)")
    
    return {"train": train, "eval": eval_ds}
