"""
Debug script to visualize the data pipeline before training.

Displays:
1. Raw samples from each dataset
2. Parsed OpenAI-style messages
3. Tokenized sequences with label masking
4. Packed batches with EOS separators

Usage:
    python -m src.entrypoints.debug_data --exp qwen3_600M_sft_thinking
    python -m src.entrypoints.debug_data --exp qwen3_600M_sft_mixed_chat_12ksteps_2048seq --samples 3
"""

import argparse
from typing import Any, Dict, List

from datasets import load_dataset

from src.core.config import Config
from src.core.registry import build
from src.builders.data import _messages_from_sample, _normalize_messages

import src.builders.tokenizer
from src.builders.data import _tokenize_prompt_and_response

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    DIM = '\033[2m'


def print_header(text: str, char: str = "="):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{char * 70}")
    print(f" {text}")
    print(f"{char * 70}{Colors.END}\n")


def print_subheader(text: str):
    print(f"\n{Colors.BOLD}{Colors.YELLOW}>>> {text}{Colors.END}")


def truncate_text(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"{Colors.DIM}... [{len(text) - max_len} more chars]{Colors.END}"


def visualize_raw_sample(sample: Dict[str, Any], idx: int):
    print(f"\n{Colors.BLUE}[Raw Sample {idx + 1}]{Colors.END}")
    print(f"{Colors.DIM}Keys: {list(sample.keys())}{Colors.END}")
    
    if "messages" in sample:
        print(f"  Format: {Colors.GREEN}OpenAI (messages){Colors.END}")
        for i, msg in enumerate(sample["messages"][:3]):
            role = msg.get("role", "?")
            content = truncate_text(str(msg.get("content", "")), 150)
            print(f"    [{i}] {Colors.CYAN}{role}{Colors.END}: {content}")
        if len(sample["messages"]) > 3:
            print(f"    {Colors.DIM}... {len(sample['messages']) - 3} more messages{Colors.END}")
    
    elif "conversations" in sample:
        print(f"  Format: {Colors.GREEN}ShareGPT (conversations){Colors.END}")
        for i, turn in enumerate(sample["conversations"][:3]):
            from_field = turn.get("from", "?")
            value = truncate_text(str(turn.get("value", "")), 150)
            print(f"    [{i}] {Colors.CYAN}{from_field}{Colors.END}: {value}")
        if len(sample["conversations"]) > 3:
            print(f"    {Colors.DIM}... {len(sample['conversations']) - 3} more turns{Colors.END}")
    
    elif "chat" in sample:
        print(f"  Format: {Colors.GREEN}Glaive (system + chat){Colors.END}")
        if sample.get("system"):
            print(f"    system: {truncate_text(str(sample['system']), 100)}")
        print(f"    chat: {truncate_text(str(sample.get('chat', '')), 200)}")
    
    elif "source" in sample and "target" in sample:
        print(f"  Format: {Colors.GREEN}CoT-Collection (source/rationale/target){Colors.END}")
        print(f"    source: {truncate_text(str(sample.get('source', '')), 150)}")
        if sample.get("rationale"):
            print(f"    rationale: {truncate_text(str(sample['rationale']), 150)}")
        print(f"    target: {truncate_text(str(sample.get('target', '')), 100)}")
    
    elif "problem" in sample and "solution" in sample:
        print(f"  Format: {Colors.GREEN}Problem/Solution{Colors.END}")
        print(f"    problem: {truncate_text(str(sample.get('problem', '')), 150)}")
        print(f"    solution: {truncate_text(str(sample.get('solution', '')), 150)}")
    
    else:
        print(f"  Format: {Colors.RED}Unknown{Colors.END}")
        for k, v in list(sample.items())[:5]:
            print(f"    {k}: {truncate_text(str(v), 100)}")


def visualize_parsed_messages(messages: List[Dict[str, str]], idx: int):
    print(f"\n{Colors.GREEN}[Parsed Messages {idx + 1}]{Colors.END}")
    
    if not messages:
        print(f"  {Colors.RED}(empty - parsing failed or no valid messages){Colors.END}")
        return
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        
        # Color-code by role
        if role == "system":
            role_color = Colors.YELLOW
        elif role == "user":
            role_color = Colors.BLUE
        else:  # assistant
            role_color = Colors.GREEN
        
        content_preview = truncate_text(content, 200)
        print(f"  [{i}] {role_color}{role:>10}{Colors.END}: {content_preview}")


def visualize_tokenized(
    input_ids: List[int],
    labels: List[int],
    tokenizer,
    idx: int,
    show_tokens: int = 50
):
    print(f"\n{Colors.CYAN}[Tokenized Sequence {idx + 1}]{Colors.END}")
    print(f"  Length: {len(input_ids)} tokens")
    
    masked = sum(1 for l in labels if l == -100)
    active = len(labels) - masked
    print(f"  Labels: {Colors.DIM}{masked} masked (-100){Colors.END}, {Colors.GREEN}{active} active (trained){Colors.END}")
    
    print(f"\n  {Colors.BOLD}First {show_tokens} tokens:{Colors.END}")
    print(f"  {'Token':<20} {'ID':>8} {'Label':>8}")
    print(f"  {'-' * 40}")
    
    for i in range(min(show_tokens, len(input_ids))):
        token_id = input_ids[i]
        label = labels[i]
        
        try:
            token_str = tokenizer.decode([token_id])
            token_str = repr(token_str)[1:-1]
        except:
            token_str = f"<{token_id}>"
        
        if len(token_str) > 18:
            token_str = token_str[:15] + "..."
        
        if label == -100:
            label_str = f"{Colors.DIM}-100{Colors.END}"
        else:
            label_str = f"{Colors.GREEN}{label}{Colors.END}"
        
        print(f"  {token_str:<20} {token_id:>8} {label_str:>8}")
    
    if len(input_ids) > show_tokens:
        print(f"  {Colors.DIM}... {len(input_ids) - show_tokens} more tokens{Colors.END}")


def visualize_packed_batch(
    batch: Dict[str, Any],
    tokenizer,
    batch_idx: int,
    show_samples: int = 2
):
    print_subheader(f"Packed Batch {batch_idx + 1}")
    
    input_ids_batch = batch["input_ids"]
    labels_batch = batch["labels"]
    
    n_samples = len(input_ids_batch)
    seq_len = len(input_ids_batch[0]) if n_samples > 0 else 0
    
    print(f"  Batch size: {n_samples}")
    print(f"  Sequence length: {seq_len}")
    
    eos_token_id = tokenizer.eos_token_id
    eos_token_str = tokenizer.decode([eos_token_id]) if eos_token_id else "<EOS>"
    print(f"  EOS token: {repr(eos_token_str)} (id={eos_token_id})")
    
    for sample_idx in range(min(show_samples, n_samples)):
        input_ids = input_ids_batch[sample_idx]
        labels = labels_batch[sample_idx]
        
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        
        print(f"\n  {Colors.BOLD}[Sample {sample_idx + 1} in batch]{Colors.END}")
        
        eos_positions = [i for i, tid in enumerate(input_ids) if tid == eos_token_id]
        print(f"  EOS positions: {eos_positions[:10]}{'...' if len(eos_positions) > 10 else ''}")
        print(f"  Packed sequences: ~{len(eos_positions)} original samples")
        
        print(f"\n  {Colors.BOLD}Sequence boundaries (around EOS):{Colors.END}")
        
        shown = 0
        for eos_pos in eos_positions[:3]:
            start = max(0, eos_pos - 5)
            end = min(len(input_ids), eos_pos + 6)
            
            segment_ids = input_ids[start:end]
            segment_labels = labels[start:end]
            
            print(f"\n    Position {eos_pos}:")
            for i, (tid, lab) in enumerate(zip(segment_ids, segment_labels)):
                pos = start + i
                try:
                    token_str = tokenizer.decode([tid])
                    token_str = repr(token_str)[1:-1]
                except:
                    token_str = f"<{tid}>"
                
                if len(token_str) > 15:
                    token_str = token_str[:12] + "..."
                
                if tid == eos_token_id:
                    marker = f"{Colors.RED}◀ EOS{Colors.END}"
                else:
                    marker = ""
                
                label_str = f"{Colors.DIM}-100{Colors.END}" if lab == -100 else f"{Colors.GREEN}{lab}{Colors.END}"
                
                print(f"      [{pos:4d}] {token_str:<18} label={label_str:<15} {marker}")
            
            shown += 1
        
        if len(eos_positions) > 3:
            print(f"\n    {Colors.DIM}... {len(eos_positions) - 3} more boundaries{Colors.END}")


def main(exp_name: str, num_samples: int = 2, num_batches: int = 2):
    print_header(f"Data Pipeline Debug: {exp_name}")
    
    cfg = Config.from_experiment(exp_name)
    
    print(f"Run: {cfg.run['name']}")
    print(f"Mode: {cfg.run['mode']}")
    print(f"Data type: {cfg.data.get('type', 'N/A')}")
    print(f"Max seq len: {cfg.data.get('max_seq_len', 'N/A')}")
    print(f"Packing: {cfg.data.get('packing', False)}")
    
    print_subheader("Building Tokenizer")
    tokenizer = build("tokenizer", **cfg.tokenizer)
    print(f"Tokenizer: {cfg.tokenizer.get('id', 'N/A')}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
    print(f"Pad token: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
    
    # =========================================================================
    # PART 1: Show raw samples and parsing for each dataset
    # =========================================================================
    print_header("PART 1: Raw Samples → Parsed Messages", "=")
    
    datasets_config = cfg.data.get("datasets_config", [])
    if not datasets_config:
        # Single dataset mode
        train_path = cfg.data.get("train_path")
        if train_path:
            datasets_config = [{"path": train_path, "weight": 1.0}]
    
    for ds_cfg in datasets_config:
        path = ds_cfg["path"]
        weight = ds_cfg.get("weight", 1.0)
        name = ds_cfg.get("name", None)
        trust_remote_code = ds_cfg.get("trust_remote_code", False)

        print_subheader(f"Dataset: {path} (weight={weight})")

        try:
            # Load small sample with config parameters
            ds = load_dataset(
                path,
                name=name,
                split="train",
                streaming=True,
                trust_remote_code=trust_remote_code
            )
            samples = list(ds.take(num_samples))
            
            for idx, sample in enumerate(samples):
                # Show raw
                visualize_raw_sample(sample, idx)
                
                # Parse to messages
                messages = _messages_from_sample(sample)
                visualize_parsed_messages(messages, idx)
                
                print(f"\n{Colors.DIM}{'─' * 60}{Colors.END}")
        
        except Exception as e:
            print(f"  {Colors.RED}Error loading dataset: {e}{Colors.END}")
    
    # =========================================================================
    # PART 2: Show tokenization with label masking
    # =========================================================================
    print_header("PART 2: Tokenized Sequences with Label Masking", "=")
    

    
    if datasets_config:
        ds_cfg = datasets_config[0]
        path = ds_cfg["path"]
        name = ds_cfg.get("name", None)
        trust_remote_code = ds_cfg.get("trust_remote_code", False)

        print(f"Using dataset: {path}\n")

        try:
            ds = load_dataset(
                path,
                name=name,
                split="train",
                streaming=True,
                trust_remote_code=trust_remote_code
            )
            samples = list(ds.take(num_samples))
            
            max_seq_len = cfg.data.get("max_seq_len", 2048)
            
            for idx, sample in enumerate(samples):
                messages = _messages_from_sample(sample)
                
                if not messages:
                    print(f"{Colors.RED}[Sample {idx + 1}] No messages parsed, skipping{Colors.END}")
                    continue
                
                # Tokenize
                tokenized_examples = _tokenize_prompt_and_response(
                    messages, tokenizer, max_seq_len
                )
                
                if not tokenized_examples:
                    print(f"{Colors.RED}[Sample {idx + 1}] No tokenized output, skipping{Colors.END}")
                    continue
                
                # Show first tokenized example
                ex = tokenized_examples[0]
                visualize_tokenized(
                    ex["input_ids"], 
                    ex["labels"], 
                    tokenizer, 
                    idx,
                    show_tokens=40
                )
                
                print(f"\n{Colors.DIM}{'─' * 60}{Colors.END}")
        
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # PART 3: Show packed batches with EOS boundaries
    # =========================================================================
    if cfg.data.get("packing", False):
        print_header("PART 3: Packed Batches with EOS Boundaries", "=")
        
        print("Building full dataset with packing...")
        print(f"{Colors.DIM}(This may take a moment for large datasets){Colors.END}\n")
        
        try:
            import src.builders.data
            dataset = build("data", tokenizer=tokenizer, **cfg.data)
            
            train_ds = dataset["train"]
            print(f"Train dataset size: {len(train_ds)} packed sequences")
            
            for batch_idx in range(min(num_batches, len(train_ds))):
                batch = {
                    "input_ids": [train_ds[batch_idx]["input_ids"]],
                    "labels": [train_ds[batch_idx]["labels"]],
                }
                visualize_packed_batch(batch, tokenizer, batch_idx, show_samples=1)
        
        except Exception as e:
            print(f"{Colors.RED}Error building dataset: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
    else:
        print_header("PART 3: Packing Disabled", "=")
        print("Packing is disabled for this experiment.")
        print("Sequences will be padded individually during collation.")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Summary", "=")
    
    print(f"✓ Experiment: {exp_name}")
    print(f"✓ Datasets: {len(datasets_config)}")
    for ds_cfg in datasets_config:
        print(f"    - {ds_cfg['path']} (weight={ds_cfg.get('weight', 1.0)})")
    print(f"✓ Max sequence length: {cfg.data.get('max_seq_len', 'N/A')}")
    print(f"✓ Packing: {cfg.data.get('packing', False)}")
    print(f"✓ EOS token ID: {tokenizer.eos_token_id}")
    
    print(f"\n{Colors.GREEN}Data pipeline validation complete!{Colors.END}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug data pipeline before training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.entrypoints.debug_data --exp qwen3_600M_sft_thinking
    python -m src.entrypoints.debug_data --exp qwen3_600M_sft_mixed_chat_12ksteps_2048seq --samples 3
    python -m src.entrypoints.debug_data --exp qwen3_600M_sft_thinking --batches 5
        """
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name (loads from configs/exp/{exp}.yaml)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of raw samples to show per dataset (default: 2)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=2,
        help="Number of packed batches to visualize (default: 2)",
    )
    
    args = parser.parse_args()
    main(exp_name=args.exp, num_samples=args.samples, num_batches=args.batches)
