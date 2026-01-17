"""
Math SFT Dataset Builder with <think> tags and \\boxed{} formatting.

A flexible, multi-dataset builder for Chain-of-Thought math SFT training
Supports various math datasets with unified formatting and deduplication

Output format:
    <think>
    {step-by-step reasoning}
    </think>

    \\boxed{final_answer}

Supported adapters:
    - dart_math: hkust-nlp/dart-math-pool-* (optimal for long reasoning)
    - openmath_reasoning: nvidia/OpenMathReasoning (competition math)
    - acemath: AceMath-Instruct (frontier model distillation)
    - scibench: xw27/scibench (college-level science/math)
    - openmath1: nvidia/OpenMathInstruct-1 (code-free solutions only)
    - openmath2: nvidia/OpenMathInstruct-2
    - bigmath_rl: SynthLabsAI/Big-Math-RL-Verified (no reasoning, answer only)
    - numina_cot: AI-MO/NuminaMath-CoT
    - orca_math: microsoft/orca-math-word-problems-200k
    - metamath: meta-math/MetaMathQA
    - generic_qa: Generic question/answer format
    - generic_ps: Generic problem/solution format
    - generic_messages: Generic OpenAI messages format
"""

import re
import hashlib
from typing import Any, Callable, Dict, List, Optional, Set

from datasets import load_dataset, interleave_datasets

from src.core.registry import register
from src.builders.data import _tokenize_prompt_and_response, _pack_tokenized_dataset

_THINK_OPEN_RE = re.compile(r"<think>\s*", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"\s*</think>", flags=re.IGNORECASE)

_CODE_FENCE_RE = re.compile(r"```[^`]*```", flags=re.DOTALL)
_LLM_CODE_RE = re.compile(r"<llm-code>.*?</llm-code>", flags=re.DOTALL | re.IGNORECASE)
_LLM_CODE_OUT_RE = re.compile(
    r"<llm-code-output>.*?</llm-code-output>", flags=re.DOTALL | re.IGNORECASE
)
_EXECUTION_RE = re.compile(
    r"<\|execution\|>.*?<\|/execution\|>", flags=re.DOTALL | re.IGNORECASE
)

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_BOXED_NESTED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def _strip_think_tokens(text: str) -> str:
    text = _THINK_OPEN_RE.sub("", text)
    text = _THINK_CLOSE_RE.sub("", text)
    return text.strip()


def _strip_tool_traces(text: str) -> str:
    text = _LLM_CODE_RE.sub("", text)
    text = _LLM_CODE_OUT_RE.sub("", text)
    text = _CODE_FENCE_RE.sub("", text)
    text = _EXECUTION_RE.sub("", text)

    # Remove common OpenMathInstruct boilerplate lines
    text = re.sub(
        r"(?im)^\s*let'?s\s+(?:solve|write|use)\s+.*\s+(?:using|in|with)\s+(?:python|code|sympy).*$",
        "",
        text,
    ).strip()
    
    text = re.sub(r"(?im)^\s*print\s*\(.*\)\s*$", "", text).strip()

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


def _strip_boxed_in_reasoning(text: str) -> str:
    """
    Prevents the model from seeing the final answer twice during training
    """
    text = _BOXED_NESTED_RE.sub(r"\1", text)
    text = _BOXED_RE.sub(r"\1", text)
    return text.strip()


def _extract_boxed_or_last_number(text: str) -> str:
    """
    Extract final answer from solution text
    """
    m = _BOXED_NESTED_RE.search(text)
    if m:
        return m.group(1).strip()
    
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()

    answer_patterns = [
        r"(?:final answer|the answer|answer|result|solution)\s*(?:is|=|:)\s*\$?([+-]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\$?",
        r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)",  # GSM8K format
        r"=\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*$",
        r"(?:equals?|is)\s+\$?([+-]?\d[\d,]*(?:\.\d+)?)\$?\s*\.?\s*$",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "").strip()

    nums = re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text)
    if nums:
        return nums[-1].replace(",", "").strip()

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else "0"


def _normalize_messages(messages: List[Dict[str, Any]], strip_think_from_all: bool = False) -> List[Dict[str, str]]:
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

        if strip_think_from_all or role in {"system", "user"}:
            content = _strip_think_tokens(content)

        if not content:
            continue

        cleaned.append({"role": role, "content": content})

    return cleaned


def _make_math_messages(user_text: str, reasoning_text: str, final_answer: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Create standardized messages with <think> and \\boxed{} format.
    
    Output format:
        system: {system_prompt}
        user: {question}
        assistant: <think>
                   {reasoning}
                   </think>
                   
                   \\boxed{answer}
    """
    msgs: List[Dict[str, str]] = []

    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt.strip()})

    msgs.append({"role": "user", "content": user_text.strip()})

    reasoning_text = reasoning_text.strip()
    final_answer = final_answer.strip()

    assistant_content = (
        f"<think>\n{reasoning_text}\n</think>\n\n\\boxed{{{final_answer}}}"
    )

    msgs.append({"role": "assistant", "content": assistant_content})

    return msgs


def _compute_content_hash(messages: List[Dict[str, str]]) -> str:
    user_content = ""
    for msg in messages:
        if msg["role"] == "user":
            user_content += msg["content"]
    
    # Normalize whitespace for consistent hashing
    user_content = " ".join(user_content.lower().split())
    return hashlib.md5(user_content.encode("utf-8")).hexdigest()


def _adapt_openmath1(
    sample: Dict[str, Any],
    system_prompt: str,
    require_correct: bool = True,
    require_noexec: bool = True,
) -> List[Dict[str, str]]:
    """
    nvidia/OpenMathInstruct-1 adapter
    Fields: question, generated_solution, expected_answer, predicted_answer,
            error_message, is_correct
    """
    if require_correct and not bool(sample.get("is_correct", False)):
        return []

    if require_noexec:
        error_msg = str(sample.get("error_message", "")).strip()
        if error_msg and error_msg != "<not_executed>":
            return []

    question = sample.get("question")
    solution = sample.get("generated_solution")
    answer = sample.get("expected_answer")

    if not question or not solution or not answer:
        return []

    solution = _strip_tool_traces(str(solution))
    solution = _strip_boxed_in_reasoning(solution)
    
    if len(solution.strip()) < 20:
        return []

    return _make_math_messages(str(question), solution, str(answer), system_prompt)


def _adapt_openmath2(
    sample: Dict[str, Any],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    nvidia/OpenMathInstruct-2 adapter
    Fields: problem, generated_solution, expected_answer, problem_source
    """
    problem = sample.get("problem")
    solution = sample.get("generated_solution")
    answer = sample.get("expected_answer")

    if not problem or not solution or not answer:
        return []

    solution = _strip_tool_traces(str(solution))
    solution = _strip_boxed_in_reasoning(solution)
    
    if len(solution.strip()) < 20:
        return []

    return _make_math_messages(str(problem), solution, str(answer), system_prompt)


def _adapt_bigmath_rl(
    sample: Dict[str, Any],
    system_prompt: str,
    min_solve_rate: float = 0.0,
) -> List[Dict[str, str]]:
    """
    SynthLabsAI/Big-Math-RL-Verified adapter
    Fields: problem, answer, source, domain, llama8b_solve_rate
    """
    solve_rate = sample.get("llama8b_solve_rate", 1.0)
    if solve_rate is not None and float(solve_rate) < min_solve_rate:
        return []

    problem = sample.get("problem")
    answer = sample.get("answer")

    if not problem or not answer:
        return []

    reasoning = f"Let me solve this problem step by step.\n\nAfter working through the problem, the answer is {answer}."

    return _make_math_messages(str(problem), reasoning, str(answer), system_prompt)


def _adapt_numina_cot(
    sample: Dict[str, Any],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    AI-MO/NuminaMath-CoT adapter
    Can have messages format OR problem/solution format
    """
    if isinstance(sample.get("messages"), list) and sample["messages"]:
        msgs = sample["messages"]
        msgs = _normalize_messages(msgs)

        user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)

        if not user or not assistant:
            return []

        reasoning = _strip_tool_traces(str(assistant))
        reasoning = _strip_boxed_in_reasoning(reasoning)
        final = _extract_boxed_or_last_number(str(assistant))
        
        if len(reasoning.strip()) < 20:
            return []

        return _make_math_messages(user, reasoning, final, system_prompt)

    problem = sample.get("problem")
    solution = sample.get("solution")

    if not problem or not solution:
        return []

    solution_str = str(solution)
    reasoning = _strip_tool_traces(solution_str)
    reasoning = _strip_boxed_in_reasoning(reasoning)
    final = _extract_boxed_or_last_number(solution_str)
    
    if len(reasoning.strip()) < 20:
        return []

    return _make_math_messages(str(problem), reasoning, final, system_prompt)


def _adapt_orca_math(
    sample: Dict[str, Any],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    microsoft/orca-math-word-problems-200k adapter
    Fields: question, answer
    """
    question = sample.get("question")
    answer = sample.get("answer")

    if not question or not answer:
        return []

    answer_str = str(answer)
    reasoning = _strip_tool_traces(answer_str)
    reasoning = _strip_boxed_in_reasoning(reasoning)
    final = _extract_boxed_or_last_number(answer_str)
    
    if len(reasoning.strip()) < 20:
        return []

    return _make_math_messages(str(question), reasoning, final, system_prompt)


def _adapt_metamath(
    sample: Dict[str, Any],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    meta-math/MetaMathQA adapter
    Fields: query, response, type, original_question
    """
    query = sample.get("query") or sample.get("original_question")
    response = sample.get("response")

    if not query or not response:
        return []

    response_str = str(response)
    reasoning = _strip_tool_traces(response_str)
    reasoning = _strip_boxed_in_reasoning(reasoning)
    final = _extract_boxed_or_last_number(response_str)
    
    if len(reasoning.strip()) < 20:
        return []

    return _make_math_messages(str(query), reasoning, final, system_prompt)


def _adapt_generic_qa(
    sample: Dict[str, Any],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    Generic question/answer adapter
    Expects: question, answer (or variations like q/a, input/output)
    """
    question = (
        sample.get("question") or 
        sample.get("q") or 
        sample.get("input") or
        sample.get("query") or
        sample.get("instruction")
    )
    answer = (
        sample.get("answer") or 
        sample.get("a") or 
        sample.get("output") or
        sample.get("response")
    )

    if not question or not answer:
        return []

    answer_str = str(answer)
    reasoning = _strip_tool_traces(answer_str)
    reasoning = _strip_boxed_in_reasoning(reasoning)
    final = _extract_boxed_or_last_number(answer_str)
    
    if len(reasoning.strip()) < 10:
        return []

    return _make_math_messages(str(question), reasoning, final, system_prompt)


def _adapt_generic_ps(
    sample: Dict[str, Any],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    Generic problem/solution adapter
    Expects: problem, solution (or variations)
    """
    problem = (
        sample.get("problem") or 
        sample.get("statement") or
        sample.get("question")
    )
    solution = (
        sample.get("solution") or 
        sample.get("answer") or
        sample.get("proof")
    )

    if not problem or not solution:
        return []

    solution_str = str(solution)
    reasoning = _strip_tool_traces(solution_str)
    reasoning = _strip_boxed_in_reasoning(reasoning)
    final = _extract_boxed_or_last_number(solution_str)
    
    if len(reasoning.strip()) < 10:
        return []

    return _make_math_messages(str(problem), reasoning, final, system_prompt)


def _adapt_generic_messages(
    sample: Dict[str, Any],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    Generic OpenAI messages format adapter
    Expects: messages list with role/content dicts
    """
    msgs = sample.get("messages") or sample.get("conversations")

    if not isinstance(msgs, list) or not msgs:
        return []

    if msgs and "from" in msgs[0]:
        converted = []
        for m in msgs:
            role = m.get("from", "")
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            converted.append({"role": role, "content": m.get("value", "")})
        msgs = converted

    msgs = _normalize_messages(msgs)

    user = next((m["content"] for m in msgs if m["role"] == "user"), None)
    assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)

    if not user or not assistant:
        return []

    reasoning = _strip_tool_traces(str(assistant))
    reasoning = _strip_boxed_in_reasoning(reasoning)
    final = _extract_boxed_or_last_number(str(assistant))

    if len(reasoning.strip()) < 10:
        return []

    return _make_math_messages(user, reasoning, final, system_prompt)


def _adapt_dart_math(
    sample: Dict[str, Any],
    system_prompt: str,
    min_reasoning_length: int = 50,
) -> List[Dict[str, str]]:
    """
    DART-Math (hkust-nlp/dart-math-pool-*) adapter
    Fields: query, resp, gt_ans, ans_correct, ans, query_metadata, resp_metadata
    """
    if not sample.get("ans_correct", True):
        return []

    query = sample.get("query")
    response = sample.get("resp")
    answer = sample.get("gt_ans") or sample.get("ans")

    if not query or not response or not answer:
        return []

    response_str = str(response)
    reasoning = _strip_tool_traces(response_str)
    reasoning = _strip_boxed_in_reasoning(reasoning)

    if len(reasoning.strip()) < min_reasoning_length:
        return []

    final_answer = str(answer).strip()

    return _make_math_messages(str(query), reasoning, final_answer, system_prompt)


def _adapt_openmath_reasoning(
    sample: Dict[str, Any],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    nvidia/OpenMathReasoning adapter

    Expected fields (similar to OpenMathInstruct but higher quality):
    - problem/question: The math problem
    - solution/reasoning: The solution with reasoning
    - answer: Final answer

    This dataset focuses on competition math with detailed proofs
    """
    problem = (
        sample.get("problem") or
        sample.get("question") or
        sample.get("query")
    )

    solution = (
        sample.get("solution") or
        sample.get("reasoning") or
        sample.get("response") or
        sample.get("generated_solution")
    )

    answer = (
        sample.get("answer") or
        sample.get("final_answer") or
        sample.get("expected_answer")
    )

    if not problem or not solution:
        return []

    solution_str = str(solution)
    reasoning = _strip_tool_traces(solution_str)
    reasoning = _strip_boxed_in_reasoning(reasoning)

    if len(reasoning.strip()) < 30:
        return []

    if answer:
        final_answer = str(answer).strip()
    else:
        final_answer = _extract_boxed_or_last_number(solution_str)

    return _make_math_messages(str(problem), reasoning, final_answer, system_prompt)


def _adapt_acemath(
    sample: Dict[str, Any],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    AceMath-Instruct adapter
    AceMath uses distilled long CoT from frontier models
    Expected format varies, but typically high-quality messages format
    """
    if "messages" in sample and isinstance(sample["messages"], list):
        msgs = _normalize_messages(sample["messages"])

        user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)

        if user and assistant:
            reasoning = _strip_tool_traces(str(assistant))
            reasoning = _strip_boxed_in_reasoning(reasoning)
            final = _extract_boxed_or_last_number(str(assistant))

            if len(reasoning.strip()) >= 30:
                return _make_math_messages(user, reasoning, final, system_prompt)

    problem = sample.get("problem") or sample.get("instruction") or sample.get("query")
    solution = sample.get("solution") or sample.get("output") or sample.get("response")

    if not problem or not solution:
        return []

    solution_str = str(solution)
    reasoning = _strip_tool_traces(solution_str)
    reasoning = _strip_boxed_in_reasoning(reasoning)
    final = _extract_boxed_or_last_number(solution_str)

    if len(reasoning.strip()) < 30:
        return []

    return _make_math_messages(str(problem), reasoning, final, system_prompt)


def _adapt_scibench(
    sample: Dict[str, Any],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    SciBench (xw27/scibench) adapter

    College-level scientific problems with detailed solutions
    Expected fields: problem_text, solution, answer_number, answer_latex, subject
    """
    problem = (
        sample.get("problem_text") or
        sample.get("problem") or
        sample.get("question")
    )

    solution = sample.get("solution") or sample.get("answer_text") or ""

    if not problem:
        return []

    solution_str = str(solution).strip()
    if not solution_str or len(solution_str) < 20:
        return []

    reasoning = _strip_tool_traces(solution_str)
    reasoning = _strip_boxed_in_reasoning(reasoning)

    if len(reasoning.strip()) < 50:
        return []

    answer = sample.get("answer_number") or sample.get("answer_latex")
    if answer:
        final_answer = str(answer).strip()
    else:
        final_answer = _extract_boxed_or_last_number(solution_str)

    return _make_math_messages(str(problem), reasoning, final_answer, system_prompt)


ADAPTERS: Dict[str, Callable] = {
    "dart_math": _adapt_dart_math,
    "openmath_reasoning": _adapt_openmath_reasoning,
    "acemath": _adapt_acemath,
    "scibench": _adapt_scibench,
    "openmath1": _adapt_openmath1,
    "openmath2": _adapt_openmath2,
    "bigmath_rl": _adapt_bigmath_rl,
    "numina_cot": _adapt_numina_cot,
    "orca_math": _adapt_orca_math,
    "metamath": _adapt_metamath,
    "generic_qa": _adapt_generic_qa,
    "generic_ps": _adapt_generic_ps,
    "generic_messages": _adapt_generic_messages,
}


def get_adapter(name: str) -> Callable:
    """Get adapter function by name"""
    if name not in ADAPTERS:
        available = ", ".join(ADAPTERS.keys())
        raise ValueError(f"Unknown adapter '{name}'. Available: {available}")
    return ADAPTERS[name]


@register("data", "math_thinking_sft")
def build_math_thinking_sft_dataset(
    datasets_config: list,
    max_seq_len: int,
    num_proc: int,
    packing: bool,
    tokenizer,
    cache_dir: str = None,
    max_token_filter: int = 3500,
    system_prompt: str = "You are a helpful math assistant. Show your reasoning step-by-step using <think></think> tags, then provide your final answer as \\boxed{answer}.",
    deduplicate: bool = True,
    eval_split_ratio: float = 0.05,
    **kwargs,
):
    """
    Build multi-dataset math SFT with <think> tags and \\boxed{} answers

    This is a flexible, general-purpose builder for math Chain-of-Thought SFT
    Add new datasets by specifying the appropriate adapter in the config

    Args:
        datasets_config: List of dataset configurations. Each entry:
            - path: HuggingFace dataset path (required)
            - weight: Sampling weight (default: 1.0)
            - adapter: Adapter name (required) - see ADAPTERS dict
            - name: HF dataset config name (optional)
            - split: HF split (default: "train")
            - trust_remote_code: bool (default: False)
        max_seq_len: Maximum sequence length
        num_proc: Number of processes for parallel processing
        packing: Whether to pack sequences
        tokenizer: Tokenizer instance
        cache_dir: Cache directory for datasets
        max_token_filter: Maximum tokens per sample (filters longer samples)
        system_prompt: System prompt for formatting
        deduplicate: Whether to remove duplicate questions
        eval_split_ratio: Ratio of data to use for evaluation

    Config example:
        data:
          type: "math_thinking_sft"
          datasets_config:
            - path: "nvidia/OpenMathInstruct-1"
              weight: 0.50
              adapter: "openmath1"
            - path: "AI-MO/NuminaMath-CoT"
              weight: 0.30
              adapter: "numina_cot"
            - path: "microsoft/orca-math-word-problems-200k"
              weight: 0.20
              adapter: "orca_math"
          max_seq_len: 4096
          deduplicate: true

    Returns:
        Dict with 'train' and 'eval' datasets
    """
    all_train = []
    all_eval = []
    probs = []

    print(f"\n{'='*60}")
    print(f"[Math Thinking SFT] Loading {len(datasets_config)} datasets")
    print(f"{'='*60}")

    for i, ds_cfg in enumerate(datasets_config):
        path = ds_cfg["path"]
        weight = float(ds_cfg.get("weight", 1.0))
        name = ds_cfg.get("name", None)
        split = ds_cfg.get("split", "train")
        adapter_name = ds_cfg.get("adapter", None)
        trust_remote_code = bool(ds_cfg.get("trust_remote_code", False))

        if not adapter_name:
            raise ValueError(f"Dataset '{path}' requires an 'adapter' field.")

        adapter_fn = get_adapter(adapter_name)

        print(f"\n  [{i+1}] {path}")
        print(f"      adapter: {adapter_name}, weight: {weight}")

        # Load dataset
        ds = load_dataset(
            path,
            name=name,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

        print(f"      loaded: {len(ds):,} samples")

        # Apply adapter
        def apply_adapter(sample):
            return {"messages": adapter_fn(sample, system_prompt)}

        ds = ds.map(
            apply_adapter,
            num_proc=num_proc,
            desc=f"Adapting {path.split('/')[-1]}",
        )

        # Filter out empty messages
        ds = ds.filter(
            lambda x: isinstance(x.get("messages"), list) and len(x["messages"]) >= 2,
            num_proc=num_proc,
        )

        print(f"      valid:  {len(ds):,} samples")

        if len(ds) == 0:
            print(f"      ⚠ WARNING: No valid samples! Skipping this dataset.")
            continue

        # Split train/eval
        split_ds = ds.train_test_split(test_size=eval_split_ratio, seed=42)
        all_train.append(split_ds["train"])
        all_eval.append(split_ds["test"])
        probs.append(weight)

    if not all_train:
        raise ValueError("No valid datasets loaded! Check your adapters and data paths.")

    # Normalize weights
    total_weight = sum(probs)
    probs = [p / total_weight for p in probs]

    print(f"\n[Math Thinking SFT] Sampling probabilities: {[f'{p:.1%}' for p in probs]}")

    # Interleave datasets
    train = interleave_datasets(
        all_train,
        probabilities=probs,
        seed=42,
        stopping_strategy="all_exhausted",
    )

    eval_ds = interleave_datasets(
        all_eval,
        probabilities=probs,
        seed=42,
        stopping_strategy="first_exhausted",
    )

    print(f"[Math Thinking SFT] Before dedup - Train: {len(train):,} | Eval: {len(eval_ds):,}")

    # Deduplication
    if deduplicate:
        print("[Math Thinking SFT] Deduplicating by question content...")

        def add_hash(batch):
            hashes = []
            for i in range(len(batch["messages"])):
                msgs = batch["messages"][i]
                if isinstance(msgs, list) and len(msgs) >= 2:
                    hashes.append(_compute_content_hash(msgs))
                else:
                    hashes.append("")
            return {"_content_hash": hashes}

        train = train.map(add_hash, batched=True, batch_size=50, num_proc=num_proc)
        eval_ds = eval_ds.map(add_hash, batched=True, batch_size=50, num_proc=num_proc)

        # Remove duplicates within train
        seen_hashes: Set[str] = set()

        def is_unique_train(example):
            h = example.get("_content_hash", "")
            if not h or h in seen_hashes:
                return False
            seen_hashes.add(h)
            return True

        train = train.filter(is_unique_train, desc="Dedup train")

        # Remove eval samples that appear in train
        train_hashes = seen_hashes.copy()

        def is_unique_eval(example):
            h = example.get("_content_hash", "")
            return h and h not in train_hashes

        eval_ds = eval_ds.filter(is_unique_eval, desc="Dedup eval")

        if "_content_hash" in train.column_names:
            train = train.remove_columns(["_content_hash"])
        if "_content_hash" in eval_ds.column_names:
            eval_ds = eval_ds.remove_columns(["_content_hash"])

        print(f"[Math Thinking SFT] After dedup  - Train: {len(train):,} | Eval: {len(eval_ds):,}")

    print("[Math Thinking SFT] Tokenizing...")

    def to_sft_features(batch):
        input_ids, attention_mask, labels = [], [], []

        batch_size = len(next(iter(batch.values()))) if batch else 0
        for idx in range(batch_size):
            try:
                sample = {k: v[idx] for k, v in batch.items()}
                messages = sample.get("messages", [])

                if not messages:
                    continue

                exs = _tokenize_prompt_and_response(
                    messages, tokenizer=tokenizer, max_seq_len=max_seq_len
                )

                for ex in exs:
                    if len(ex["input_ids"]) <= max_token_filter:
                        input_ids.append(ex["input_ids"])
                        attention_mask.append(ex["attention_mask"])
                        labels.append(ex["labels"])
            except Exception as e:
                # Skip samples that cause errors to prevent subprocess crashes
                continue

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train = train.map(
        to_sft_features,
        batched=True,
        batch_size=50,
        num_proc=num_proc,
        remove_columns=train.column_names,
        desc="Tokenizing train",
    )

    eval_ds = eval_ds.map(
        to_sft_features,
        batched=True,
        batch_size=50,
        num_proc=num_proc,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval",
    )

    print(f"[Math Thinking SFT] After tokenization - Train: {len(train):,} | Eval: {len(eval_ds):,}")

    if packing:
        print("[Math Thinking SFT] Packing sequences...")
        train = _pack_tokenized_dataset(train, max_seq_len, tokenizer.eos_token_id)
        eval_ds = _pack_tokenized_dataset(eval_ds, max_seq_len, tokenizer.eos_token_id)
        print(f"[Math Thinking SFT] After packing - Train: {len(train):,} | Eval: {len(eval_ds):,}")

    columns = ["input_ids", "attention_mask", "labels"]
    train = train.with_format("torch", columns=columns)
    eval_ds = eval_ds.with_format("torch", columns=columns)

    print(f"\n{'='*60}")
    print(f"[Math Thinking SFT] Dataset ready!")
    print(f"  Train: {len(train):,} samples")
    print(f"  Eval:  {len(eval_ds):,} samples")
    print(f"{'='*60}\n")

    return {"train": train, "eval": eval_ds}

