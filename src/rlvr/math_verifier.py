import re
from typing import Optional, Tuple, Literal
from fractions import Fraction
import math
import hashlib


AnswerType = Literal["numeric", "latex", "symbolic", "text"]


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def _normalize_latex(expr: str) -> str:
    """Normalize LaTeX expressions for comparison."""
    expr = expr.strip()

    # Remove common LaTeX formatting variations
    expr = re.sub(r'\\left\(', '(', expr)
    expr = re.sub(r'\\right\)', ')', expr)
    expr = re.sub(r'\\left\[', '[', expr)
    expr = re.sub(r'\\right\]', ']', expr)
    expr = re.sub(r'\\left\\{', '{', expr)
    expr = re.sub(r'\\right\\}', '}', expr)

    # Normalize whitespace in LaTeX
    expr = re.sub(r'\s+', ' ', expr)
    expr = expr.strip()

    # Common equivalences
    expr = re.sub(r'\\dfrac', r'\\frac', expr)
    expr = re.sub(r'\\tfrac', r'\\frac', expr)

    return expr


def _try_parse_numeric(answer_str: str) -> Optional[float]:
    """Try to parse a string as a numeric value, handling fractions."""
    answer_str = answer_str.strip()

    # Try direct float conversion
    try:
        return float(answer_str)
    except ValueError:
        pass

    # Try fraction (e.g., "3/4" or "1/2")
    if '/' in answer_str and '\\' not in answer_str:
        try:
            parts = answer_str.split('/')
            if len(parts) == 2:
                num = float(parts[0].strip())
                den = float(parts[1].strip())
                if den != 0:
                    return num / den
        except (ValueError, ZeroDivisionError):
            pass

    # Try LaTeX fraction (\frac{a}{b})
    frac_match = re.match(r'\\frac\{([^}]+)\}\{([^}]+)\}', answer_str)
    if frac_match:
        try:
            num = float(frac_match.group(1).strip())
            den = float(frac_match.group(2).strip())
            if den != 0:
                return num / den
        except (ValueError, ZeroDivisionError):
            pass

    return None


def _classify_answer_type(answer_str: str) -> AnswerType:
    """Classify the type of answer for appropriate comparison."""
    answer_str = answer_str.strip()

    # Check if it's a pure number
    if _try_parse_numeric(answer_str) is not None:
        return "numeric"

    # Check for LaTeX math expressions
    latex_indicators = [
        r'\\frac', r'\\sqrt', r'\\pi', r'\\infty', r'\\pm',
        r'\\times', r'\\div', r'\\cdot', r'\\leq', r'\\geq',
        r'\\sin', r'\\cos', r'\\tan', r'\\log', r'\\ln',
        r'\^', r'_'
    ]

    for indicator in latex_indicators:
        if indicator in answer_str:
            return "latex"

    # Check for symbolic expressions (intervals, sets, tuples)
    symbolic_patterns = [
        r'\([^)]+,[^)]+\)',  # (a, b)
        r'\[[^\]]+,[^\]]+\]',  # [a, b]
        r'\{[^}]+\}',  # {a, b, c}
        r'\\in', r'\\cup', r'\\cap', r'\\subset'
    ]

    for pattern in symbolic_patterns:
        if re.search(pattern, answer_str):
            return "symbolic"

    return "text"


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
    """
    Strict extraction for GSM8K format (PPO warmup stage).
    Only accepts answers in \\boxed{} tags.
    """
    if not text:
        return None, 0.0

    after_think = text.split("</think>")[-1] if "</think>" in text else text

    boxed_match = re.search(r'\\boxed\{([^}]+)\}', after_think)
    if boxed_match:
        return _normalize_answer(boxed_match.group(1)), 1.0
    return None, 0.0


def extract_answer_typed(text: str) -> Tuple[Optional[str], AnswerType, float, bool]:
    """
    Enhanced extraction that preserves answer format for MATH/IF datasets.

    Following DeepSeekMath best practices:
    - Prioritize \\boxed{} extraction (highest confidence)
    - Preserve LaTeX expressions without normalization
    - Support flexible fallbacks for partial formatting
    - Track whether proper format was used

    Returns:
        (answer, answer_type, confidence, used_boxed)
        - answer: Raw extracted answer (format preserved)
        - answer_type: Classification (numeric/latex/symbolic/text)
        - confidence: Extraction confidence (0.0-1.0)
        - used_boxed: Whether answer was in \\boxed{} tags
    """
    if not text:
        return None, "text", 0.0, False

    # Extract content after </think> tag if present
    after_think = text.split("</think>")[-1] if "</think>" in text else text
    after_think = after_think.strip()

    # Priority 1: Extract from \boxed{} with nested brace support
    # Handles: \boxed{42}, \boxed{\frac{3}{4}}, \boxed{(-\infty, 5]}
    boxed_pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
    boxed_match = re.search(boxed_pattern, after_think)

    if boxed_match:
        raw_answer = boxed_match.group(1).strip()
        answer_type = _classify_answer_type(raw_answer)
        return raw_answer, answer_type, 1.0, True

    # Priority 2: "answer is X" or "answer: X" patterns
    # Try to capture mathematical expressions including LaTeX
    answer_patterns = [
        # With $ delimiters (LaTeX inline math)
        (r'(?:final answer|the answer|answer|result|solution)\s*(?:is|=|:)\s*\$([^$]+?)\$', 0.85),
        # Without $ but up to sentence boundary
        (r'(?:final answer|the answer|answer|result|solution)\s*(?:is|=|:)\s*(.+?)(?:\.|$)', 0.75),
    ]

    for pattern, confidence in answer_patterns:
        match = re.search(pattern, after_think, re.IGNORECASE)
        if match:
            raw_answer = match.group(1).strip()
            # Clean up common trailing punctuation
            raw_answer = raw_answer.rstrip('.,;:')
            if raw_answer:
                answer_type = _classify_answer_type(raw_answer)
                return raw_answer, answer_type, confidence, False

    # Priority 3: Standalone equation at end (e.g., "= 42")
    equals_match = re.search(r'=\s*([+-]?\d+(?:\.\d+)?)\s*$', after_think)
    if equals_match:
        raw_answer = equals_match.group(1).strip()
        return raw_answer, "numeric", 0.65, False

    # Priority 4: Last number as fallback (low confidence - may be wrong)
    all_numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', after_think)
    if all_numbers:
        raw_answer = all_numbers[-1]
        return raw_answer, "numeric", 0.3, False

    return None, "text", 0.0, False


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
    """
    Legacy comparison for backward compatibility.
    Uses numeric normalization.
    """
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


def compare_answers_typed(
    predicted: Optional[str],
    ground_truth: str,
    pred_type: AnswerType = "text",
    truth_type: AnswerType = None
) -> bool:
    """
    Type-aware answer comparison for MATH/IF datasets.

    Handles:
    - Numeric: float comparison with epsilon tolerance
    - LaTeX: normalized string comparison + numeric fallback
    - Symbolic: normalized string comparison
    - Text: case-insensitive string comparison

    Args:
        predicted: Predicted answer string
        ground_truth: Ground truth answer string
        pred_type: Type of predicted answer (if known)
        truth_type: Type of ground truth (auto-detected if None)

    Returns:
        True if answers match according to type-specific rules
    """
    if predicted is None:
        return False

    predicted = predicted.strip()
    ground_truth = ground_truth.strip()

    # Auto-detect ground truth type if not provided
    if truth_type is None:
        truth_type = _classify_answer_type(ground_truth)

    # Exact match (fastest check)
    if predicted == ground_truth:
        return True

    # Type-specific comparison
    if pred_type == "numeric" or truth_type == "numeric":
        # Try numeric comparison
        pred_num = _try_parse_numeric(predicted)
        truth_num = _try_parse_numeric(ground_truth)

        if pred_num is not None and truth_num is not None:
            # Use relative tolerance for large numbers, absolute for small
            abs_tol = 1e-6
            rel_tol = 1e-9
            diff = abs(pred_num - truth_num)
            return diff <= abs_tol or diff <= rel_tol * max(abs(pred_num), abs(truth_num))

    if pred_type == "latex" or truth_type == "latex":
        # Normalize LaTeX expressions
        pred_latex = _normalize_latex(predicted)
        truth_latex = _normalize_latex(ground_truth)

        if pred_latex == truth_latex:
            return True

        # Try numeric evaluation as fallback for simple expressions
        pred_num = _try_parse_numeric(pred_latex)
        truth_num = _try_parse_numeric(truth_latex)

        if pred_num is not None and truth_num is not None:
            diff = abs(pred_num - truth_num)
            return diff <= 1e-6 or diff <= 1e-9 * max(abs(pred_num), abs(truth_num))

    if pred_type == "symbolic" or truth_type == "symbolic":
        # Normalize whitespace and compare
        pred_norm = re.sub(r'\s+', '', predicted)
        truth_norm = re.sub(r'\s+', '', ground_truth)

        if pred_norm == truth_norm:
            return True

        # Case-insensitive fallback
        if pred_norm.lower() == truth_norm.lower():
            return True

    # Text comparison (case-insensitive, whitespace normalized)
    pred_text = ' '.join(predicted.lower().split())
    truth_text = ' '.join(ground_truth.lower().split())

    return pred_text == truth_text


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

def _extract_sections_plan_work_answer(text: str) -> dict:
    """
    Extract PLAN / WORK / ANSWER sections.
    Returns dict with keys: plan, work, answer, has_plan, has_work, has_answer, answer_tail
    """
    out = {
        "plan": "",
        "work": "",
        "answer": "",
        "has_plan": False,
        "has_work": False,
        "has_answer": False,
        "answer_tail": "",
    }
    if not text:
        return out

    # If think tags exist, ignore that segment completely (optional)
    after_think = text.split("</think>")[-1] if "</think>" in text else text

    # Normalize section headers: allow "PLAN:" or "Plan:" etc
    pattern = r'(?is)(?:^|\n)\s*(PLAN|WORK|ANSWER)\s*:\s*'
    matches = list(re.finditer(pattern, after_think))

    if not matches:
        return out

    # slice into sections
    sections = {}
    for i, m in enumerate(matches):
        label = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(after_think)
        sections[label] = after_think[start:end].strip()

    out["plan"] = sections.get("plan", "")
    out["work"] = sections.get("work", "")
    out["answer"] = sections.get("answer", "")

    out["has_plan"] = "plan" in sections
    out["has_work"] = "work" in sections
    out["has_answer"] = "answer" in sections

    # tail region for answer sanity check
    out["answer_tail"] = after_think[-300:].strip()  # last 300 chars
    return out

def _ngram_repetition_ratio(text: str, n: int = 4) -> float:
    """
    Returns a repetition score in [0,1], higher means more repeated n-grams.
    """
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < n * 3:
        return 0.0

    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if not ngrams:
        return 0.0

    total = len(ngrams)
    unique = len(set(ngrams))
    rep_ratio = 1.0 - unique / total
    return max(0.0, min(1.0, rep_ratio))

def _structure_score_plan_work_answer(text: str) -> Tuple[float, dict]:
    """
    Score format quality for PLAN/WORK/ANSWER structure.
    Returns: (score in [0,1], info)
    """
    info = _extract_sections_plan_work_answer(text)
    score = 0.0

    # basic header presence
    if info["has_plan"]:
        score += 0.2
    if info["has_work"]:
        score += 0.2
    if info["has_answer"]:
        score += 0.2

    # Encourage content in PLAN/WORK/ANSWER
    plan_len = len(info["plan"])
    work_len = len(info["work"])
    ans_len = len(info["answer"])

    # plan should not be empty, but also not huge
    if plan_len >= 20:
        score += 0.1
    if 20 <= plan_len <= 400:
        score += 0.05

    # work should be non-trivial
    if work_len >= 80:
        score += 0.1

    # answer should be present and short-ish
    if 1 <= ans_len <= 200:
        score += 0.05

    # Answer should be near end (discourage "ANSWER" early then ramble)
    answer_pos = text.lower().rfind("answer:")
    if answer_pos != -1 and answer_pos > len(text) * 0.6:
        score += 0.1

    return min(1.0, score), info

def _has_boxed_answer(text: str) -> bool:
    after_think = text.split("</think>")[-1] if "</think>" in text else text
    boxed_pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
    return re.search(boxed_pattern, after_think) is not None

# ------------------------------------------------------------------------------
# Reward functions
# ------------------------------------------------------------------------------

def ppo_reward_binary(prompts, completions, answer, **kwargs):
    """
    Binary reward function for PPO warmup stage

    Requires strict GSM8K format: answer must be in \\boxed{} tags
    This enforces format adherence learned during SFT

    Args:
        prompts: List of prompt strings (repeated for each generation)
        completions: List of completion strings
        answer: List of ground truth answers (automatically repeated by TRL to match completions)
        **kwargs: Additional fields from dataset

    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)

    Note: TRL automatically repeats all dataset fields to match num_generations,
    so len(prompts) == len(completions) == len(answer)
    """
    rewards = []

    for completion, correct_answer in zip(completions, answer, strict=True):
        text = _extract_completion_text(completion)
        extracted, confidence = extract_answer_GSM8K(text)
        is_correct = compare_answers(extracted, str(correct_answer))

        reward = 1.0 if is_correct else 0.0
        rewards.append(reward)

    return rewards


def dapo_reward_advanced(prompts, completions, answer, **kwargs):
    """
    Advanced reward function for DAPO training with format quality bonuses
    (Legacy version - kept for backward compatibility)

    Accepts flexible answer extraction but rewards proper formatting with bonuses
    Designed for later training stages after format has been learned

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
    so len(prompts) == len(completions) == len(answer)
    """
    rewards = []

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


def dapo_reward_deepseek(prompts, completions, answer, **kwargs):
    """
    DeepSeekMath-style DAPO reward function for GSM8K + MATH training

    Following DeepSeekMath principles:
    1. Outcome-based rewards (correctness is paramount)
    2. Format bonuses to encourage proper structure
    3. NO partial credit for wrong answers
    4. Strong penalties for malformed/degenerate responses
    5. Dataset-aware logic for MATH vs GSM8K

    Reward Structure:
    ================
    CORRECT ANSWER (1.0 base):
      + 0.3 bonus for complete <think></think> tags with reasoning
      + 0.2 bonus for \\boxed{} format
      + 0.1 bonus for substantial reasoning (MATH dataset only)
      Max: 1.6

    WRONG ANSWER (0.0 base):
      NO partial credit for wrong answers (DeepSeekMath key insight)
      Format bonuses only apply to correct answers

    MALFORMED RESPONSES (penalties from 0.0):
      - 0.5 for partial think tags (breaks format)
      - 0.3 for very short responses (< 30 chars)
      - 0.2 for no extractable answer
      Min: -1.0

    Args:
        prompts: List of prompt strings
        completions: List of completion strings
        answer: List of ground truth answers
        **kwargs: Additional fields (dataset_source for MATH vs GSM8K routing)

    Returns:
        List of rewards in range [-1.0, 1.6]

    Note: Higher reward ceiling than PPO to provide stronger learning signal
    for format quality in DAPO stage.
    """
    rewards = []

    dataset_source = kwargs.get("dataset_source", [])
    if not isinstance(dataset_source, list):
        dataset_source = [dataset_source] * len(completions)

    for idx, (completion, correct_answer) in enumerate(zip(completions, answer, strict=True)):
        text = _extract_completion_text(completion)

        extracted, answer_type, confidence, used_boxed = extract_answer_typed(text)

        ds_source = dataset_source[idx] if idx < len(dataset_source) else "unknown"

        is_correct = compare_answers_typed(
            extracted,
            str(correct_answer),
            pred_type=answer_type
        )

        format_info = check_format_quality(text)

        if is_correct:
            reward = 1.0
        else:
            reward = 0.0

        if is_correct:
            if format_info["has_think_tags"] and format_info["answer_after_thinking"]:
                if format_info["thinking_lines"] >= 5:
                    reward += 0.3
                elif format_info["thinking_lines"] >= 2:
                    reward += 0.2
                else:
                    reward += 0.1

            if used_boxed:
                reward += 0.2

            if "MATH" in ds_source or "math" in ds_source.lower():
                if format_info["thinking_length"] > 300:
                    reward += 0.1

        if format_info["has_partial_tags"] and not format_info["has_think_tags"]:
            reward -= 0.5

        if format_info["total_length"] < 30:
            reward -= 0.3

        if extracted is None:
            reward -= 0.2

        reward = max(-1.0, min(1.6, reward))

        rewards.append(reward)

    return rewards

#TODO: Implement GRPO reward for self reflection
# def grpo_reward_reflection(prompts, completions, answer, **kwargs):
#     """
#     Binary reward function for GRPO reflection training

#     Accepts flexible answer extraction for robustness
#     Simple binary reward based only on correctness

#     Args:
#         prompts: List of prompt strings (repeated for each generation)
#         completions: List of completion strings
#         answer: List of ground truth answers (automatically repeated by TRL to match completions)
#         **kwargs: Additional fields from dataset

#     Returns:
#         List of rewards (1.0 for correct, 0.0 for incorrect)

#     Note: TRL automatically repeats all dataset fields to match num_generations,
#     so len(prompts) == len(completions) == len(answer)
#     """
#     rewards = []

#     # TRL ensures all lists have the same length after repeating dataset fields
#     for completion, correct_answer in zip(completions, answer, strict=True):
#         text = _extract_completion_text(completion)
#         extracted, confidence = extract_answer(text)
#         is_correct = compare_answers(extracted, str(correct_answer))

#         reward = 1.0 if is_correct else 0.0
#         rewards.append(reward)

#     return rewards

def dapo_diverse(prompts, completions, answer, **kwargs):
    """
    Diverse DAPO reward:
    - Enforces PLAN/WORK/ANSWER + boxed answer contract
    - Correctness is anchor reward
    - Adds dense shaping for reasoning depth & structure
    - Penalizes failure modes: repetition, missing answer, rambling, refusals, malformed tags

    Reward Range: approx [-1.2, 1.9]
    Note: Slightly unstable and also too heavy for training causing long wall times just for reward calcs
    """
    rewards = []

    dataset_source = kwargs.get("dataset_source", [])
    if not isinstance(dataset_source, list):
        dataset_source = [dataset_source] * len(completions)

    for idx, (completion, correct_answer) in enumerate(zip(completions, answer, strict=True)):
        text = _extract_completion_text(completion)
        ds_source = dataset_source[idx] if idx < len(dataset_source) else "unknown"

        extracted, answer_type, confidence, used_boxed = extract_answer_typed(text)

        is_correct = compare_answers_typed(
            extracted,
            str(correct_answer),
            pred_type=answer_type
        )

        struct_score, sections = _structure_score_plan_work_answer(text)
        has_boxed = used_boxed or _has_boxed_answer(text)

        fmt = check_format_quality(text)
        rep4 = _ngram_repetition_ratio(text, n=4)
        rep3 = _ngram_repetition_ratio(text, n=3)
        refusal = bool(re.search(r"\b(as an ai|i can't|i cannot|sorry|unable to)\b", text.lower()))

        total_len = len(text)
        work_len = len(sections.get("work", ""))

        reward = 1.0 if is_correct else 0.0

        prompt_str = prompts[idx] if isinstance(prompts, list) and idx < len(prompts) else str(prompts)
        h = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()
        jitter = (int(h[:8], 16) / 0xFFFFFFFF)  # [0,1]

        w_struct = 0.25 + 0.15 * jitter      # [0.25, 0.40]
        w_depth  = 0.15 + 0.10 * (1 - jitter)  # [0.15, 0.25]

        if is_correct:
            if struct_score > 0.75:
                reward += 0.25

            if has_boxed:
                reward += 0.2

            reward += w_struct * struct_score

            if work_len > 0:
                depth_bonus = 1 - math.exp(-work_len / 250.0)   # ~0->1
                reward += w_depth * depth_bonus

            if re.search(r"(\\frac|\\sqrt|\^|_|\\pi|\\times|\\cdot)", text):
                reward += 0.05

            reward += 0.05 * confidence

            if "math" in str(ds_source).lower() and work_len > 300:
                reward += 0.05

        else:
            # tiny shaping for non-degenerate well-formatted wrong answers
            # (keeps model from abandoning format)
            # If you want DeepSeek strictness, set this to 0.
            if extracted is not None and struct_score > 0.6 and total_len < 2500:
                reward += 0.05 * struct_score

        if fmt["has_partial_tags"] and not fmt["has_think_tags"]:
            reward -= 0.5

        if extracted is None:
            reward -= 0.3

        # Must obey contract (PLAN/WORK/ANSWER)
        if struct_score < 0.35:
            reward -= 0.3

        # If says ANSWER but missing \boxed => penalize
        if sections["has_answer"] and not has_boxed:
            reward -= 0.25

        if refusal:
            reward -= 0.6

        if rep4 > 0.25:
            reward -= 0.4
        if rep3 > 0.35:
            reward -= 0.3

        if total_len < 60:
            reward -= 0.3
        if total_len > 3500:
            reward -= 0.3
        if total_len > 6000:
            reward -= 0.6

        reward = max(-1.2, min(1.9, reward))
        rewards.append(reward)

    return rewards

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

    subs = [t[i:i+50] for i in range(0, len(t)-50, 25)]
    if subs:
        uniq_ratio = len(set(subs)) / len(subs)
        if uniq_ratio < 0.55:
            penalty = max(penalty, 0.35)

    words = re.findall(r"\w+", t)
    if len(words) > 120:
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top = max(freq.values())
        if top / len(words) > 0.14:
            penalty = max(penalty, 0.30)

    return min(0.5, penalty)


def dapo_diverse_stable(prompts, completions, answer, **kwargs):
    """
    Stable diverse reward for tiny saturated models

    Key properties:
    - Narrow range [-0.8, 1.2]
    - No reward jitter
    - Cheap repetition penalty
    - Wrong answers get minimal/no shaping (prevents GRPO training on style when all wrong)
    - Smooth penalties instead of cliffs
    - No confidence-based rewards (prevents reward hacking)
    """
    rewards = []

    dataset_source = kwargs.get("dataset_source", [])
    if not isinstance(dataset_source, list):
        dataset_source = [dataset_source] * len(completions)

    # TRL behavior with custom rollout_func:
    # - If batch_size=2, num_generations=16:
    # - TRL passes: len(prompts)=2 (unreplicated), len(completions)=32, len(answer)=2
    # - Completions are ordered: [p0_g0, p0_g1, ..., p0_g15, p1_g0, p1_g1, ..., p1_g15, ...]
    # - We return 32 rewards (one per completion)
    num_generations = len(completions) // len(answer) if len(answer) > 0 else 1

    for idx, completion in enumerate(completions):
        # Map completion idx to prompt idx: completion i corresponds to prompt i // num_generations
        prompt_idx = idx // num_generations
        correct_answer = answer[prompt_idx]
        text = _extract_completion_text(completion)
        ds_source = dataset_source[prompt_idx] if prompt_idx < len(dataset_source) else "unknown"

        extracted, answer_type, confidence, used_boxed = extract_answer_typed(text)
        is_correct = compare_answers_typed(extracted, str(correct_answer), pred_type=answer_type)

        # Structure contract score
        struct_score, sections = _structure_score_plan_work_answer(text)
        has_boxed = used_boxed or _has_boxed_answer(text)

        fmt = check_format_quality(text)
        total_len = len(text)
        work_len = len(sections.get("work", ""))

        refusal = bool(re.search(r"\b(as an ai|i can't|i cannot|sorry|unable to)\b", text.lower()))
        repeat_pen = _cheap_repeat_penalty(text)

        # ========= Weights (fixed, smooth) =========
        w_struct_correct = 0.30
        w_depth_correct  = 0.10  # Reduced from 0.20 to limit verbosity incentive

        # ========= Base reward =========
        reward = 1.0 if is_correct else 0.0

        # ========= Bonuses (ONLY for correct answers) =========
        if is_correct:
            # boxed bonus
            if has_boxed:
                reward += 0.15

            # Continuous structure bonus (removed cliff bonus to avoid saturation)
            reward += w_struct_correct * struct_score

            # Depth: saturating bonus with cap to prevent excessive verbosity
            # Cap at 250 tokens and use faster saturation curve
            if work_len > 0:
                capped_work_len = min(work_len, 250)
                depth_bonus = 1 - math.exp(-capped_work_len / 120.0)
                reward += w_depth_correct * depth_bonus

            # optional: slightly encourage heavier work on MATH
            if "math" in str(ds_source).lower() and work_len > 350:
                reward += 0.05

        else:
            # Fix 1: Minimal wrong-answer shaping with small spread for extractability
            # Prevents flat reward batches while avoiding style training
            if extracted is not None:
                reward = -0.03  # extractable but wrong
            else:
                reward = -0.12  # no extractable answer

        # ========= Penalties (smooth, not cliff-like) =========
        # Partial tags penalty (malformed format)
        if fmt["has_partial_tags"] and not fmt["has_think_tags"]:
            reward -= 0.4

        # Smooth extraction penalty (Fix 3)
        if extracted is None:
            reward -= 0.20  # slightly reduced from 0.25

        # Smooth structure penalty based on score (Fix 3)
        # Instead of cliff at 0.35, use smooth penalty
        if struct_score < 0.5:
            struct_penalty = (0.5 - struct_score) * 0.40  # max 0.20 penalty
            reward -= struct_penalty

        # Smooth boxing penalty (Fix 3)
        if sections["has_answer"] and not has_boxed:
            reward -= 0.15  # slightly reduced from 0.20

        # Strong refusal penalty (kept strong as this is critical)
        if refusal:
            reward -= 0.6

        # repetition penalty (cheap)
        reward -= repeat_pen

        # Smooth length penalties (Fix 3)
        if total_len < 60:
            reward -= 0.20  # slightly reduced from 0.25
        elif total_len < 100:
            # Smooth transition instead of cliff
            reward -= 0.10

        if total_len > 4500:
            # Smooth penalty for long outputs
            excess_penalty = min(0.30, (total_len - 4500) / 10000.0)
            reward -= excess_penalty
        if total_len > 6500:
            reward -= 0.50  # slightly reduced from 0.55

        # clamp
        reward = max(-0.8, min(1.2, reward))
        rewards.append(reward)

    return rewards


def _compute_quality_score(text: str, extracted: Optional[str], is_correct: bool) -> float:
    """
    Compute deterministic quality score in range [0, 100].

    Scoring breakdown:
    - Correctness: 40 points (binary)
    - Structure: 25 points (PLAN/WORK/ANSWER quality)
    - Format: 15 points (boxed + tags)
    - Depth: 10 points (work section substance)
    - Conciseness: 5 points (length appropriateness)
    - Repetition: 5 points (non-repetitive content)

    Args:
        text: Completion text to score
        extracted: Extracted answer (or None if not extractable)
        is_correct: Whether the extracted answer is correct

    Returns:
        Score in range [0, 100]
    """
    score = 0.0

    # 1. Correctness: 40 points
    if is_correct:
        score += 40.0

    # 2. Structure: 25 points (PLAN/WORK/ANSWER quality)
    struct_score, sections = _structure_score_plan_work_answer(text)
    score += 25.0 * struct_score

    # 3. Format: 15 points
    fmt = check_format_quality(text)
    has_boxed = _has_boxed_answer(text)

    # Complete tags: 7.5 points
    if fmt["has_think_tags"]:
        score += 7.5

    # Boxed answer: 7.5 points
    if has_boxed:
        score += 7.5

    # 4. Depth: 10 points (work section substance)
    work_len = len(sections.get("work", ""))
    if work_len > 0:
        # Saturate at 400 chars for 10 points
        depth_score = min(work_len / 400.0, 1.0) * 10.0
        score += depth_score

    # 5. Conciseness: 5 points (penalize extremes)
    total_len = len(text)
    conciseness_score = 5.0
    if total_len < 80:
        conciseness_score = 0.0
    elif total_len < 150:
        conciseness_score = 2.5
    elif total_len > 6000:
        conciseness_score = 0.0
    elif total_len > 5000:
        conciseness_score = 2.5
    score += conciseness_score

    # 6. Repetition: 5 points (non-repetitive)
    repeat_pen = _cheap_repeat_penalty(text)
    repetition_score = max(0.0, 5.0 - repeat_pen * 10.0)
    score += repetition_score

    return min(100.0, max(0.0, score))


def dapo_rank_stratified(prompts, completions, answer, **kwargs):
    """
    Stratified ranking reward with deterministic quality scoring.

    Ranks completions within correctness tiers to preserve absolute signal
    while adding relative quality comparisons.

    Correctness tiers:
    - Correct answers: ranked by quality → [0.0, 1.0]
    - Wrong answers: ranked by quality → [-1.0, -0.01]

    Quality scoring (deterministic, 0-100 scale):
    - Correctness: 40 points (binary)
    - Structure (PLAN/WORK/ANSWER): 25 points
    - Format (\\boxed{} + tags): 15 points
    - Depth (work section length): 10 points
    - Conciseness (length appropriateness): 5 points
    - Repetition (non-repetitive): 5 points

    Reward range: [-1.0, 1.0]

    Key properties:
    - Preserves correctness as primary signal (can't rank wrong > correct)
    - Adds relative quality signal within tier (distinguishes similar answers)
    - Deterministic and reproducible (no stochastic scoring)
    - Handles edge cases (single completion in tier, all same score)

    Edge cases:
    - Single correct in group: gets +1.0 (preserves full signal on hard problems)
    - All wrong in group: ranked [-1, -0.01] by quality
    - Tied scores: ranked by original order (stable sort)

    Important: ranking is per-prompt (within the num_generations group),
    not across the entire batch. This prevents cross-prompt interference
    where easy-problem completions compete against hard-problem completions.

    Args:
        prompts: List of prompt strings
        completions: List of completion strings
        answer: List of ground truth answers
        **kwargs: Additional fields (dataset_source, etc.)

    Returns:
        List of rewards in range [-1.0, 1.0]
    """
    rewards = []

    dataset_source = kwargs.get("dataset_source", [])
    if not isinstance(dataset_source, list):
        dataset_source = [dataset_source] * len(completions)

    # Handle TRL behavior with custom rollout_func
    num_generations = len(completions) // len(answer) if len(answer) > 0 else 1

    # Process per-prompt groups (ranking must be within same prompt)
    num_prompts = len(answer)
    rewards = [0.0] * len(completions)

    for prompt_idx in range(num_prompts):
        start = prompt_idx * num_generations
        end = start + num_generations
        correct_answer = answer[prompt_idx]

        # Score all completions for this prompt
        group_data = []
        for idx in range(start, end):
            text = _extract_completion_text(completions[idx])

            extracted, _ = extract_answer_GSM8K(text)
            is_correct = compare_answers(extracted, str(correct_answer))
            quality_score = _compute_quality_score(text, extracted, is_correct)

            group_data.append({
                'idx': idx,
                'is_correct': is_correct,
                'quality_score': quality_score,
            })

        # Stratify into correct and wrong tiers
        correct_tier = [d for d in group_data if d['is_correct']]
        wrong_tier = [d for d in group_data if not d['is_correct']]

        def rank_and_assign(tier, reward_min, reward_max):
            """Rank tier by quality and map to [reward_min, reward_max]."""
            if not tier:
                return

            sorted_tier = sorted(tier, key=lambda x: x['quality_score'])
            n = len(sorted_tier)

            if n == 1:
                # Single completion: use upper end to preserve signal on hard problems
                rewards[sorted_tier[0]['idx']] = reward_max
            else:
                for rank, item in enumerate(sorted_tier):
                    normalized_rank = rank / (n - 1)
                    reward = reward_min + normalized_rank * (reward_max - reward_min)
                    rewards[item['idx']] = reward

        # Rank correct tier → [0.0, 1.0]
        rank_and_assign(correct_tier, 0.0, 1.0)

        # Rank wrong tier → [-1.0, -0.01]
        rank_and_assign(wrong_tier, -1.0, -0.01)

    return rewards


def dapo_structure_balanced(prompts, completions, answer, **kwargs):
    """
    Balanced structure-based DAPO reward with strict format requirements.

    Format contract:
    - PLAN: section for problem analysis
    - WORK: section for step-by-step reasoning
    - ANSWER: section with \\boxed{final_answer}

    Key improvements over dapo_diverse_stable:
    - Strict \\boxed{} requirement (like ppo_binary)
    - Clean -1.0 to 1.0 range (symmetric)
    - No overlapping penalties
    - Structure quality modulates correct rewards
    - Wrong answers have clear negative signal

    Reward range: [-1.0, 1.0]

    Reward structure:
    ================
    CORRECT ANSWERS (base 0.7):
      + Structure quality bonus: 0.0 to 0.3 (based on PLAN/WORK/ANSWER presence)
      + Work depth bonus: 0.0 to 0.1 (substantial reasoning in WORK section)
      Subtotal before penalties: 0.7 to 1.1 (clamped to [−1.0, 1.0])

    WRONG ANSWERS:
      - Extractable but wrong: -0.5 base
      - No extractable \\boxed{}: -0.7 base

    PENALTIES (non-overlapping):
      - Refusal detected: reward = -1.0 (terminal, skips all other penalties)
      - Repetition: up to -0.3 (cheap n-gram detection, applies to all)
      - Poor structure: up to -0.2 (wrong answers only, if struct_score < 0.4)
      - Extreme length: -0.2 (< 80 chars OR > 5000 chars, applies to all)

    Final range after clamping: [-1.0, 1.0]

    Args:
        prompts: List of prompt strings
        completions: List of completion strings
        answer: List of ground truth answers
        **kwargs: Additional fields (dataset_source, etc.)

    Returns:
        List of rewards in range [-1.0, 1.0]
    """
    rewards = []

    dataset_source = kwargs.get("dataset_source", [])
    if not isinstance(dataset_source, list):
        dataset_source = [dataset_source] * len(completions)

    # Handle TRL behavior with custom rollout_func
    num_generations = len(completions) // len(answer) if len(answer) > 0 else 1

    for idx, completion in enumerate(completions):
        prompt_idx = idx // num_generations
        correct_answer = answer[prompt_idx]
        text = _extract_completion_text(completion)

        # STRICT \\boxed{} extraction (like ppo_binary)
        extracted, _ = extract_answer_GSM8K(text)
        is_correct = compare_answers(extracted, str(correct_answer))

        # Structure analysis
        struct_score, sections = _structure_score_plan_work_answer(text)
        work_len = len(sections.get("work", ""))
        total_len = len(text)

        # Failure mode detection
        refusal = bool(re.search(r"\b(as an ai|i can't|i cannot|sorry|unable to)\b", text.lower()))
        repeat_pen = _cheap_repeat_penalty(text)

        # ========= BASE REWARD (correctness is primary) =========
        if is_correct:
            # Correct answer: 0.7 base
            reward = 0.7

            # Structure quality bonus (0.0 to 0.3)
            # Rewards PLAN/WORK/ANSWER format compliance
            reward += 0.3 * struct_score

            # Work depth bonus (0.0 to 0.1)
            # Rewards substantial reasoning, saturates at 300 chars
            if work_len > 0:
                depth_factor = min(work_len / 300.0, 1.0)
                reward += 0.1 * depth_factor
        else:
            # Wrong answer: different base depending on extraction
            if extracted is None:
                reward = -0.7  # No extractable \\boxed{} answer
            else:
                reward = -0.5  # Wrong but extractable answer

        # ========= PENALTIES (non-overlapping, apply to all) =========

        # Critical: Refusal is terminal failure
        if refusal:
            reward = -1.0
            rewards.append(reward)
            continue  # Skip other penalties

        # Repetition penalty (capped)
        if repeat_pen > 0:
            reward -= min(0.3, repeat_pen)

        # Poor structure penalty (wrong answers only — correct answers
        # already get structure modulation via the 0.3 * struct_score bonus)
        if not is_correct and struct_score < 0.4:
            structure_penalty = (0.4 - struct_score) * 0.5  # max 0.2
            reward -= structure_penalty

        # Extreme length penalty (non-overlapping conditions)
        if total_len < 80:
            reward -= 0.2
        elif total_len > 5000:
            reward -= 0.2

        # ========= CLAMP TO RANGE =========
        reward = max(-1.0, min(1.0, reward))
        rewards.append(reward)

    return rewards


def _get_dapo_R1_dense():
    """Lazy import to avoid circular dependency."""
    from src.rlvr.math_rewards_v2 import dapo_R1_dense
    return dapo_R1_dense


REWARD_FUNCTIONS = {
    "ppo_binary": ppo_reward_binary,
    "dapo_advanced": dapo_reward_advanced,  # Legacy
    "dapo_deepseek": dapo_reward_deepseek,  # Recommended for mixed datasets
    # "grpo_reflection": grpo_reward_reflection, # TODO: Implement GRPO reward for self reflection
    "dapo_diverse": dapo_diverse,
    "dapo_diverse_stable": dapo_diverse_stable,
    "dapo_structure_balanced": dapo_structure_balanced,  # Clean structure-based with -1 to 1 range
    "dapo_rank_stratified": dapo_rank_stratified,  # Ranking-based with correctness stratification
}


def get_reward_function(name: str):
    # Handle lazy-loaded reward functions
    if name == "dapo_R1_dense":
        return _get_dapo_R1_dense()

    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name}. "
            f"Available: {list(REWARD_FUNCTIONS.keys()) + ['dapo_R1_dense']}"
        )
    return REWARD_FUNCTIONS[name]


# Default reward function
math_reward_fn = dapo_reward_deepseek
