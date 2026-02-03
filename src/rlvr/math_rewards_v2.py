"""
R1-style dense reward functions for RLVR training.

Format contract:
<think>
[step-by-step reasoning]
</think>

<answer>final_answer</answer>
"""
import re
import math
from typing import Optional, Tuple

# Import helper functions from math_verifier
from src.rlvr.math_verifier import (
    _extract_completion_text,
    _try_parse_numeric,
    _classify_answer_type,
    compare_answers_typed,
)


def _extract_answer_from_answer_tags(text: str) -> Tuple[Optional[str], float]:
    """
    Extract answer from <answer>...</answer> tags.

    Returns:
        (answer, confidence) where confidence is 1.0 if tags found, 0.0 otherwise
    """
    if not text:
        return None, 0.0

    # Look for <answer>...</answer> tags (case-insensitive)
    answer_pattern = r'<answer>\s*(.+?)\s*</answer>'
    match = re.search(answer_pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        answer = match.group(1).strip()
        return answer, 1.0

    return None, 0.0


def _check_R1_format_quality(text: str) -> dict:
    """
    Check format quality for R1-style responses.

    Expected format:
    <think>reasoning</think>
    <answer>final_answer</answer>

    Returns:
        dict with format quality metrics
    """
    has_think_open = "<think>" in text.lower()
    has_think_close = "</think>" in text.lower()
    has_answer_open = "<answer>" in text.lower()
    has_answer_close = "</answer>" in text.lower()

    # Extract thinking content
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, text, re.IGNORECASE | re.DOTALL)
    thinking_content = think_match.group(1).strip() if think_match else ""

    # Extract answer content
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, text, re.IGNORECASE | re.DOTALL)
    answer_content = answer_match.group(1).strip() if answer_match else ""

    # Check if answer comes after thinking
    think_pos = text.lower().find("</think>")
    answer_pos = text.lower().find("<answer>")
    answer_after_thinking = (think_pos != -1 and answer_pos != -1 and answer_pos > think_pos)

    # Count thinking lines
    thinking_lines = len([line for line in thinking_content.split('\n') if line.strip()]) if thinking_content else 0

    # Check for proper structure
    has_complete_think_tags = has_think_open and has_think_close
    has_complete_answer_tags = has_answer_open and has_answer_close
    has_proper_format = has_complete_think_tags and has_complete_answer_tags and answer_after_thinking

    return {
        "has_think_tags": has_complete_think_tags,
        "has_answer_tags": has_complete_answer_tags,
        "has_partial_think": (has_think_open or has_think_close) and not has_complete_think_tags,
        "has_partial_answer": (has_answer_open or has_answer_close) and not has_complete_answer_tags,
        "has_proper_format": has_proper_format,
        "thinking_content": thinking_content,
        "thinking_length": len(thinking_content),
        "thinking_lines": thinking_lines,
        "answer_content": answer_content,
        "answer_length": len(answer_content),
        "answer_after_thinking": answer_after_thinking,
        "total_length": len(text),
    }


def _cheap_repeat_penalty(text: str) -> float:
    """
    Cheap repetition penalty in [0, 0.5].
    Designed to catch loops without heavy n-gram computation.
    """
    if not text:
        return 0.0

    t = re.sub(r"\s+", " ", text.lower()).strip()
    if len(t) < 160:
        return 0.0

    penalty = 0.0

    # Check for repeated 50-char substrings
    subs = [t[i:i+50] for i in range(0, len(t)-50, 25)]
    if subs:
        uniq_ratio = len(set(subs)) / len(subs)
        if uniq_ratio < 0.55:
            penalty = max(penalty, 0.35)

    # Check for repeated words
    words = re.findall(r"\w+", t)
    if len(words) > 120:
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top = max(freq.values())
        if top / len(words) > 0.14:
            penalty = max(penalty, 0.30)

    return min(0.5, penalty)


def dapo_R1_dense(prompts, completions, answer, **kwargs):
    """
    R1-style dense DAPO reward function.

    Format contract:
    <think>
    [step-by-step reasoning]
    </think>

    <answer>final_answer</answer>

    Key principles (R1-style):
    1. CORRECTNESS IS PARAMOUNT - base reward 1.0 for correct, 0.0 for wrong
    2. Enforce strict format: <think> tags for reasoning, <answer> tags for final answer
    3. Dense shaping to guide learning:
       - Reward proper formatting (only when correct)
       - Reward reasoning depth
       - Penalize malformed responses
    4. No partial credit for wrong answers (DeepSeekMath principle)

    Reward Structure:
    ================
    CORRECT ANSWER (1.0 base):
      + 0.20 bonus for complete <think></think> tags with substantial reasoning (≥5 lines)
      + 0.15 bonus for complete <answer></answer> tags
      + 0.10 bonus for proper format (answer after thinking)
      Max: 1.45

    WRONG ANSWER (0.0 base):
      Small negative shaping to distinguish failure modes:
      - Extractable but wrong: -0.05
      - No extractable answer: -0.15
      (Prevents flat reward batches while avoiding style training on wrong answers)

    MALFORMED RESPONSES (penalties from base):
      - 0.50 for partial think tags (breaks format)
      - 0.40 for partial answer tags (breaks format)
      - 0.25 for missing answer tags entirely
      - 0.20 for missing think tags entirely
      - 0.60 for refusals
      - up to 0.50 for repetition
      - 0.20-0.30 for length issues
      Min: ~-1.0

    Args:
        prompts: List of prompt strings
        completions: List of completion strings
        answer: List of ground truth answers
        **kwargs: Additional fields (dataset_source, etc.)

    Returns:
        List of rewards in range [-1.0, 1.45]
    """
    rewards = []

    dataset_source = kwargs.get("dataset_source", [])
    if not isinstance(dataset_source, list):
        dataset_source = [dataset_source] * len(completions)

    # Handle TRL behavior with custom rollout_func
    num_generations = len(completions) // len(answer) if len(answer) > 0 else 1

    for idx, completion in enumerate(completions):
        # Map completion idx to prompt idx
        prompt_idx = idx // num_generations
        correct_answer = answer[prompt_idx]
        text = _extract_completion_text(completion)
        ds_source = dataset_source[prompt_idx] if prompt_idx < len(dataset_source) else "unknown"

        # Extract answer from <answer> tags
        extracted, confidence = _extract_answer_from_answer_tags(text)

        # Classify answer type if extracted
        answer_type = _classify_answer_type(extracted) if extracted else "text"

        # Check correctness
        is_correct = compare_answers_typed(extracted, str(correct_answer), pred_type=answer_type)

        # Check format quality
        fmt = _check_R1_format_quality(text)

        # Check for refusals and repetition
        refusal = bool(re.search(r"\b(as an ai|i can't|i cannot|sorry|unable to)\b", text.lower()))
        repeat_pen = _cheap_repeat_penalty(text)

        total_len = fmt["total_length"]
        thinking_len = fmt["thinking_length"]
        thinking_lines = fmt["thinking_lines"]

        # ========= Base reward (CORRECTNESS IS PARAMOUNT) =========
        reward = 1.0 if is_correct else 0.0

        # ========= Bonuses (ONLY for correct answers) =========
        if is_correct:
            # Complete think tags with substantial reasoning
            if fmt["has_think_tags"]:
                if thinking_lines >= 5:
                    reward += 0.20
                elif thinking_lines >= 2:
                    reward += 0.10
                else:
                    reward += 0.05

            # Complete answer tags
            if fmt["has_answer_tags"]:
                reward += 0.15

            # Proper format (answer after thinking)
            if fmt["has_proper_format"]:
                reward += 0.10

        else:
            # Wrong answer shaping (minimal, just to distinguish failure modes)
            if extracted is not None:
                reward = -0.05  # Extractable but wrong
            else:
                reward = -0.15  # No extractable answer

        # ========= Penalties (apply to all responses) =========

        # Partial/missing tags penalties (malformed format)
        if fmt["has_partial_think"]:
            reward -= 0.50  # Partial think tags
        elif not fmt["has_think_tags"]:
            reward -= 0.20  # Missing think tags entirely

        if fmt["has_partial_answer"]:
            reward -= 0.40  # Partial answer tags
        elif not fmt["has_answer_tags"]:
            reward -= 0.25  # Missing answer tags entirely

        # Missing extractable answer
        if extracted is None:
            reward -= 0.25

        # Refusals (strong penalty)
        if refusal:
            reward -= 0.60

        # Repetition penalty
        reward -= repeat_pen

        # Length penalties (smooth)
        if total_len < 60:
            reward -= 0.30
        elif total_len < 100:
            reward -= 0.15

        if total_len > 4000:
            excess_penalty = min(0.30, (total_len - 4000) / 8000.0)
            reward -= excess_penalty
        if total_len > 6000:
            reward -= 0.50

        # Clamp to reasonable range
        reward = max(-1.0, min(1.45, reward))
        rewards.append(reward)

    return rewards
