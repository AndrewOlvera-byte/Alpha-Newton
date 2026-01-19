import re
from typing import Optional, Tuple


def extract_answer(text: str) -> Tuple[Optional[str], float]:
    if not text:
        return None, 0.0

    after_think = text.split("</think>")[-1] if "</think>" in text else text

    boxed_match = re.search(r'\\boxed\{([^}]+)\}', after_think)
    if boxed_match:
        return _normalize_answer(boxed_match.group(1)), 1.0

    answer_patterns = [
        (r'(?:answer|result|solution)\s*(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)', 0.9),
        (r'(?:answer|result|solution)\s*(?:is|=|:)\s*(.+?)(?:\.|$)', 0.8),
    ]

    for pattern, confidence in answer_patterns:
        match = re.search(pattern, after_think, re.IGNORECASE)
        if match:
            return _normalize_answer(match.group(1)), confidence

    equals_match = re.search(r'=\s*([+-]?\d+(?:\.\d+)?)\s*$', after_think)
    if equals_match:
        return _normalize_answer(equals_match.group(1)), 0.7

    all_numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', after_think)
    if all_numbers:
        return _normalize_answer(all_numbers[-1]), 0.3

    return None, 0.0

def extract_answer_GSM8K(text: str) -> Tuple[Optional[str], float]:
    if not text:
        return None, 0.0

    after_think = text.split("</think>")[-1] if "</think>" in text else text

    boxed_match = re.search(r'\\boxed\{([^}]+)\}', after_think)
    if boxed_match:
        return _normalize_answer(boxed_match.group(1)), 1.0
    return None, 0.0


def _normalize_answer(answer_str: str) -> str:
    answer_str = answer_str.strip()
    try:
        num = float(answer_str)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return answer_str


def compare_answers(predicted: Optional[str], ground_truth: str) -> bool:
    if predicted is None:
        return False

    pred_norm = _normalize_answer(predicted)
    truth_norm = _normalize_answer(ground_truth)

    if pred_norm == truth_norm:
        return True

    if pred_norm.lower() == truth_norm.lower():
        return True

    try:
        pred_num = float(pred_norm)
        truth_num = float(truth_norm)
        return abs(pred_num - truth_num) < 1e-6
    except (ValueError, TypeError):
        pass

    return False


def check_format_quality(text: str) -> dict:
    has_think_open = "<think>" in text
    has_think_close = "</think>" in text

    think_pattern = r"<think>(.*?)</think>"
    match = re.search(think_pattern, text, re.DOTALL)
    thinking_content = match.group(1).strip() if match else ""

    after_think = text.split("</think>")[-1] if "</think>" in text else ""
    answer_in_right_place = bool(after_think.strip())

    thinking_lines = len([line for line in thinking_content.split('\n') if line.strip()]) if thinking_content else 0

    return {
        "has_think_tags": has_think_open and has_think_close,
        "has_partial_tags": has_think_open or has_think_close,
        "thinking_length": len(thinking_content),
        "thinking_lines": thinking_lines,
        "answer_after_thinking": answer_in_right_place,
        "total_length": len(text),
    }


def _extract_completion_text(completion):
    if isinstance(completion, list) and len(completion) > 0:
        return completion[0].get("content", "")
    elif isinstance(completion, dict):
        return completion.get("content", "")
    else:
        return str(completion)


def ppo_reward_binary(prompts, completions, answer, **kwargs):
    """
    Binary reward function for PPO warmup stage.

    Requires strict GSM8K format: answer must be in \\boxed{} tags.
    This enforces format adherence learned during SFT.

    Args:
        prompts: List of prompt strings (repeated for each generation)
        completions: List of completion strings
        answer: List of ground truth answers (automatically repeated by TRL to match completions)
        **kwargs: Additional fields from dataset

    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)

    Note: TRL automatically repeats all dataset fields to match num_generations,
    so len(prompts) == len(completions) == len(answer).
    """
    rewards = []

    # TRL ensures all lists have the same length after repeating dataset fields
    for completion, correct_answer in zip(completions, answer, strict=True):
        text = _extract_completion_text(completion)
        extracted, confidence = extract_answer_GSM8K(text)
        is_correct = compare_answers(extracted, str(correct_answer))

        reward = 1.0 if is_correct else 0.0
        rewards.append(reward)

    return rewards


def dapo_reward_advanced(prompts, completions, answer, **kwargs):
    """
    Advanced reward function for DAPO training with format quality bonuses.

    Accepts flexible answer extraction but rewards proper formatting with bonuses.
    Designed for later training stages after format has been learned.

    Reward structure:
    - Correct answer: 1.0 base
      + 0.2 bonus for proper <think></think> tags
      + up to 0.1 bonus based on extraction confidence
    - Wrong but extractable answer: 0.2
      + 0.1 bonus for proper <think> tags
    - Malformed responses:
      - 0.3 penalty for partial think tags (e.g., only opening tag)
      - 0.2 penalty for very short responses (< 30 chars)

    Args:
        prompts: List of prompt strings (repeated for each generation)
        completions: List of completion strings
        answer: List of ground truth answers (automatically repeated by TRL to match completions)
        **kwargs: Additional fields from dataset

    Returns:
        List of rewards (range: -0.5 to 1.3)

    Note: TRL automatically repeats all dataset fields to match num_generations,
    so len(prompts) == len(completions) == len(answer).
    """
    rewards = []

    # TRL ensures all lists have the same length after repeating dataset fields
    for completion, correct_answer in zip(completions, answer, strict=True):
        text = _extract_completion_text(completion)
        extracted, confidence = extract_answer(text)
        is_correct = compare_answers(extracted, str(correct_answer))

        format_info = check_format_quality(text)

        reward = 0.0

        if is_correct:
            reward = 1.0
            if format_info["has_think_tags"] and format_info["answer_after_thinking"]:
                reward += 0.2
            reward += confidence * 0.1
        elif extracted is not None:
            reward = 0.2
            if format_info["has_think_tags"]:
                reward += 0.1
        else:
            reward = 0.0

        if format_info["has_partial_tags"] and not format_info["has_think_tags"]:
            reward -= 0.3

        if format_info["total_length"] < 30:
            reward -= 0.2

        rewards.append(reward)

    return rewards


def grpo_reward_reflection(prompts, completions, answer, **kwargs):
    """
    Binary reward function for GRPO reflection training.

    Accepts flexible answer extraction for robustness.
    Simple binary reward based only on correctness.

    Args:
        prompts: List of prompt strings (repeated for each generation)
        completions: List of completion strings
        answer: List of ground truth answers (automatically repeated by TRL to match completions)
        **kwargs: Additional fields from dataset

    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)

    Note: TRL automatically repeats all dataset fields to match num_generations,
    so len(prompts) == len(completions) == len(answer).
    """
    rewards = []

    # TRL ensures all lists have the same length after repeating dataset fields
    for completion, correct_answer in zip(completions, answer, strict=True):
        text = _extract_completion_text(completion)
        extracted, confidence = extract_answer(text)
        is_correct = compare_answers(extracted, str(correct_answer))

        reward = 1.0 if is_correct else 0.0
        rewards.append(reward)

    return rewards


REWARD_FUNCTIONS = {
    "ppo_binary": ppo_reward_binary,
    "dapo_advanced": dapo_reward_advanced,
    "grpo_reflection": grpo_reward_reflection,
}


def get_reward_function(name: str):
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name}. "
            f"Available: {list(REWARD_FUNCTIONS.keys())}"
        )
    return REWARD_FUNCTIONS[name]


math_reward_fn = dapo_reward_advanced
