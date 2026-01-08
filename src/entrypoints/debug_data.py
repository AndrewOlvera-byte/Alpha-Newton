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
import src.builders.data
import src.builders.data_thinking
from src.builders.data import _tokenize_prompt_and_response
from src.builders.data_thinking import get_adapter

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

    elif "question" in sample and "generated_solution" in sample and "expected_answer" in sample:
        print(f"  Format: {Colors.GREEN}OpenMathInstruct (question/generated_solution/expected_answer){Colors.END}")
        print(f"    question: {truncate_text(str(sample.get('question', '')), 150)}")
        print(f"    generated_solution: {truncate_text(str(sample.get('generated_solution', '')), 200)}")
        print(f"    expected_answer: {truncate_text(str(sample.get('expected_answer', '')), 50)}")
        print(f"    is_correct: {sample.get('is_correct', 'N/A')}")
        print(f"    error_message: {truncate_text(str(sample.get('error_message', '')), 100) or '(empty - no code execution)'}")

    elif "question" in sample and "answer" in sample:
        print(f"  Format: {Colors.GREEN}Question/Answer (e.g., Orca Math){Colors.END}")
        print(f"    question: {truncate_text(str(sample.get('question', '')), 150)}")
        print(f"    answer: {truncate_text(str(sample.get('answer', '')), 200)}")

    elif "query" in sample and "resp" in sample and "ans_correct" in sample:
        print(f"  Format: {Colors.GREEN}DART-Math (query/resp/ans_correct){Colors.END}")
        print(f"    query: {truncate_text(str(sample.get('query', '')), 150)}")
        print(f"    resp: {truncate_text(str(sample.get('resp', '')), 300)}")
        print(f"    gt_ans: {sample.get('gt_ans', 'N/A')}")
        print(f"    ans_correct: {sample.get('ans_correct', 'N/A')}")
        if sample.get('query_metadata'):
            print(f"    query_metadata: {sample.get('query_metadata')}")

    elif "problem_text" in sample and "solution" in sample:
        print(f"  Format: {Colors.GREEN}SciBench (problem_text/solution){Colors.END}")
        print(f"    problem_text: {truncate_text(str(sample.get('problem_text', '')), 150)}")
        print(f"    solution: {truncate_text(str(sample.get('solution', '')), 200)}")
        print(f"    answer_number: {sample.get('answer_number', 'N/A')}")
        print(f"    subject: {sample.get('subject', 'N/A')}")

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
        
        # Show more content for assistant to see <think> tags and \\boxed{}
        max_len = 400 if role == "assistant" else 200
        content_preview = truncate_text(content, max_len)
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
    
    # Get system prompt and data type from config
    system_prompt = cfg.data.get("system_prompt", "")
    data_type = cfg.data.get("type", "")
    
    # Check if we should use adapters (for math_thinking_sft or similar)
    use_adapters = data_type in ["math_thinking_sft"] and system_prompt
    
    for ds_cfg in datasets_config:
        path = ds_cfg["path"]
        weight = ds_cfg.get("weight", 1.0)
        name = ds_cfg.get("name", None)
        split = ds_cfg.get("split", "train")
        trust_remote_code = ds_cfg.get("trust_remote_code", False)
        adapter_name = ds_cfg.get("adapter", None)

        print_subheader(f"Dataset: {path} (weight={weight})")
        if adapter_name:
            print(f"Adapter: {adapter_name}")
        if split != "train":
            print(f"Split: {split}")

        try:
            # Load small sample with config parameters
            ds = load_dataset(
                path,
                name=name,
                split=split,
                streaming=True,
                trust_remote_code=trust_remote_code
            )
            samples = list(ds.take(num_samples))
            
            for idx, sample in enumerate(samples):
                # Show raw
                visualize_raw_sample(sample, idx)

                # Parse to messages using adapter if available
                if use_adapters and adapter_name:
                    try:
                        adapter_fn = get_adapter(adapter_name)
                        messages = adapter_fn(sample, system_prompt)

                        # Diagnostic: Show reasoning length for all adapters
                        if messages:
                            assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
                            if assistant_msg:
                                content = assistant_msg.get("content", "")
                                # Extract just the reasoning part (inside <think> tags)
                                import re
                                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                                if think_match:
                                    reasoning = think_match.group(1).strip()
                                    print(f"  {Colors.DIM}📊 Reasoning length: {len(reasoning)} chars{Colors.END}")

                        if not messages and adapter_name == "openmath1":
                            # Debug why OpenMathInstruct-1 is failing
                            print(f"  {Colors.YELLOW}Debug: generated_solution present: {bool(sample.get('generated_solution'))}{Colors.END}")
                            if sample.get('generated_solution'):
                                print(f"  {Colors.YELLOW}Debug: solution length: {len(str(sample.get('generated_solution')))}{Colors.END}")
                    except Exception as e:
                        print(f"  {Colors.RED}Adapter error: {e}{Colors.END}")
                        import traceback
                        traceback.print_exc()
                        messages = []
                else:
                    # Fallback to generic parser
                    messages = _messages_from_sample(sample)

                visualize_parsed_messages(messages, idx)
                
                print(f"\n{Colors.DIM}{'─' * 60}{Colors.END}")
        
        except Exception as e:
            error_msg = str(e)
            print(f"  {Colors.RED}Error loading dataset: {error_msg}{Colors.END}")
            if "gated dataset" in error_msg.lower() or "authenticated" in error_msg.lower():
                print(f"  {Colors.YELLOW}💡 Tip: This is a gated dataset. You may need to:{Colors.END}")
                print(f"  {Colors.YELLOW}     1. Request access at https://huggingface.co/{path}{Colors.END}")
                print(f"  {Colors.YELLOW}     2. Login with: huggingface-cli login{Colors.END}")
                print(f"  {Colors.YELLOW}     3. Or set HF_TOKEN environment variable{Colors.END}")
    
    # =========================================================================
    # PART 2: Show tokenization with label masking
    # =========================================================================
    print_header("PART 2: Tokenized Sequences with Label Masking", "=")
    

    
    if datasets_config:
        ds_cfg = datasets_config[0]
        path = ds_cfg["path"]
        name = ds_cfg.get("name", None)
        split = ds_cfg.get("split", "train")
        trust_remote_code = ds_cfg.get("trust_remote_code", False)
        adapter_name = ds_cfg.get("adapter", None)

        print(f"Using dataset: {path}")
        if adapter_name:
            print(f"Adapter: {adapter_name}")
        if split != "train":
            print(f"Split: {split}")
        print()

        try:
            ds = load_dataset(
                path,
                name=name,
                split=split,
                streaming=True,
                trust_remote_code=trust_remote_code
            )
            samples = list(ds.take(num_samples))
            
            max_seq_len = cfg.data.get("max_seq_len", 2048)
            
            for idx, sample in enumerate(samples):
                # Parse using adapter if available
                if use_adapters and adapter_name:
                    try:
                        adapter_fn = get_adapter(adapter_name)
                        messages = adapter_fn(sample, system_prompt)
                    except Exception as e:
                        print(f"{Colors.RED}[Sample {idx + 1}] Adapter error: {e}{Colors.END}")
                        messages = []
                else:
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
    # PART 3: Show packed batches with EOS boundaries (SKIPPED FOR SPEED)
    # =========================================================================
    print_header("PART 3: Packed Batches (Skipped for Speed)", "=")

    if cfg.data.get("packing", False):
        print(f"{Colors.YELLOW}⚠ Skipping full dataset build{Colors.END}")
        print(f"\nPacking is ENABLED in your config.")
    else:
        print("Packing is disabled for this experiment.")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Summary", "=")

    print(f" Experiment: {exp_name}")
    print(f" Datasets: {len(datasets_config)}")
    for ds_cfg in datasets_config:
        print(f"    - {ds_cfg['path']} (weight={ds_cfg.get('weight', 1.0)})")
    print(f" Max sequence length: {cfg.data.get('max_seq_len', 'N/A')}")
    print(f" Packing: {cfg.data.get('packing', False)}")
    print(f" EOS token ID: {tokenizer.eos_token_id}")

    print(f"\n{Colors.BOLD}📝 Notes:{Colors.END}")
    print(f" • DART-Math datasets use rejection sampling for high-quality long reasoning")
    print(f" • Check the '📊 Reasoning length' diagnostics above for each dataset")
    print(f" • Target: 500-2000+ chars per reasoning chain to fill 4096 seq length")
    print(f" • Gated datasets require HuggingFace authentication")
    print(f" • Adapters apply dataset-specific formatting and quality filters")
    print(f" • All code blocks/tool traces are stripped for code-free reasoning")

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
