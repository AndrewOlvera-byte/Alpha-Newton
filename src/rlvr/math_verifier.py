import re


def extract_answer(text: str) -> tuple[str | None, float]:
    if not text:
        return None, 0.0

    after_think = text.split("</think>")[-1] if "</think>" in text else text

    boxed_match = re.search(r'\\boxed\{([^}]+)\}', after_think)
    if boxed_match:
        return _normalize_number(boxed_match.group(1)), 1.0

    answer_patterns = [
        (r'(?:answer|result|solution)\s*(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)', 0.9),
        (r'=\s*([+-]?\d+(?:\.\d+)?)\s*$', 0.7),
    ]

    for pattern, confidence in answer_patterns:
        match = re.search(pattern, after_think, re.IGNORECASE)
        if match:
            return _normalize_number(match.group(1)), confidence

    all_numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', after_think)
    if all_numbers:
        return _normalize_number(all_numbers[-1]), 0.3

    return None, 0.0


def _normalize_number(s: str) -> str:
    s = s.strip()
    try:
        num = float(s)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return s


def _check_format_quality(text: str) -> dict:
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


def math_reward_fn(prompts, completions, answer, **kwargs):
    rewards = []

    for completion, correct_answer in zip(completions, answer):
        if isinstance(completion, list) and len(completion) > 0:
            text = completion[0].get("content", "")
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        reward = 0.0

        extracted, confidence = extract_answer(text)
        correct = _normalize_number(str(correct_answer).strip())

        if extracted == correct:
            reward += 1.0
        elif extracted is not None:
            reward -= 0.1
        else:
            reward -= 0.3

        format_info = _check_format_quality(text)

        if format_info["has_think_tags"]:
            reward += 0.15

            if format_info["answer_after_thinking"]:
                reward += 0.1
        elif format_info["has_partial_tags"]:
            reward -= 0.2

        if extracted is not None:
            reward += confidence * 0.1

        if format_info["total_length"] < 50:
            reward -= 0.2
        elif format_info["total_length"] > 2000:
            if format_info["thinking_lines"] < 5:
                reward -= 0.15

        if format_info["has_think_tags"]:
            if format_info["thinking_lines"] < 2:
                reward -= 0.1
            elif format_info["thinking_lines"] > 50:
                reward -= 0.05

        rewards.append([reward])

    return rewards
