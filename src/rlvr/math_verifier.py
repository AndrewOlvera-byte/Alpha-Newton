import re
from typing import Optional, Tuple, Literal
AnswerType = Literal['numeric', 'latex', 'symbolic', 'text']

def _try_parse_numeric(answer_str: str) -> Optional[float]:
    """Try to parse a string as a numeric value, handling fractions."""
    answer_str = answer_str.strip()
    try:
        return float(answer_str)
    except ValueError:
        pass
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
    frac_match = re.match('\\\\frac\\{([^}]+)\\}\\{([^}]+)\\}', answer_str)
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
    if _try_parse_numeric(answer_str) is not None:
        return 'numeric'
    latex_indicators = ['\\\\frac', '\\\\sqrt', '\\\\pi', '\\\\infty', '\\\\pm', '\\\\times', '\\\\div', '\\\\cdot', '\\\\leq', '\\\\geq', '\\\\sin', '\\\\cos', '\\\\tan', '\\\\log', '\\\\ln', '\\^', '_']
    for indicator in latex_indicators:
        if indicator in answer_str:
            return 'latex'
    symbolic_patterns = ['\\([^)]+,[^)]+\\)', '\\[[^\\]]+,[^\\]]+\\]', '\\{[^}]+\\}', '\\\\in', '\\\\cup', '\\\\cap', '\\\\subset']
    for pattern in symbolic_patterns:
        if re.search(pattern, answer_str):
            return 'symbolic'
    return 'text'

def extract_answer_GSM8K(text: str) -> Tuple[Optional[str], float]:
    """
    Strict extraction for GSM8K format (PPO warmup stage).
    Only accepts answers in \\boxed{} tags.
    """
    if not text:
        return (None, 0.0)
    after_think = text.split('</think>')[-1] if '</think>' in text else text
    boxed_match = re.search('\\\\boxed\\{([^}]+)\\}', after_think)
    if boxed_match:
        return (_normalize_answer(boxed_match.group(1)), 1.0)
    return (None, 0.0)

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
        return (None, 'text', 0.0, False)
    after_think = text.split('</think>')[-1] if '</think>' in text else text
    after_think = after_think.strip()
    boxed_pattern = '\\\\boxed\\{((?:[^{}]|\\{[^{}]*\\})*)\\}'
    boxed_match = re.search(boxed_pattern, after_think)
    if boxed_match:
        raw_answer = boxed_match.group(1).strip()
        answer_type = _classify_answer_type(raw_answer)
        return (raw_answer, answer_type, 1.0, True)
    answer_patterns = [('(?:final answer|the answer|answer|result|solution)\\s*(?:is|=|:)\\s*\\$([^$]+?)\\$', 0.85), ('(?:final answer|the answer|answer|result|solution)\\s*(?:is|=|:)\\s*(.+?)(?:\\.|$)', 0.75)]
    for pattern, confidence in answer_patterns:
        match = re.search(pattern, after_think, re.IGNORECASE)
        if match:
            raw_answer = match.group(1).strip()
            raw_answer = raw_answer.rstrip('.,;:')
            if raw_answer:
                answer_type = _classify_answer_type(raw_answer)
                return (raw_answer, answer_type, confidence, False)
    equals_match = re.search('=\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*$', after_think)
    if equals_match:
        raw_answer = equals_match.group(1).strip()
        return (raw_answer, 'numeric', 0.65, False)
    all_numbers = re.findall('[+-]?\\d+(?:\\.\\d+)?', after_think)
    if all_numbers:
        raw_answer = all_numbers[-1]
        return (raw_answer, 'numeric', 0.3, False)
    return (None, 'text', 0.0, False)

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
    """Compare extracted answers after numeric and case normalization."""
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
        return abs(pred_num - truth_num) < 1e-06
    except (ValueError, TypeError):
        pass
    return False

def check_format_quality(text: str) -> dict:
    has_think_open = '<think>' in text
    has_think_close = '</think>' in text
    think_pattern = '<think>(.*?)</think>'
    match = re.search(think_pattern, text, re.DOTALL)
    thinking_content = match.group(1).strip() if match else ''
    after_think = text.split('</think>')[-1] if '</think>' in text else ''
    answer_in_right_place = bool(after_think.strip())
    thinking_lines = len([line for line in thinking_content.split('\n') if line.strip()]) if thinking_content else 0
    return {'has_think_tags': has_think_open and has_think_close, 'has_partial_tags': has_think_open or has_think_close, 'thinking_length': len(thinking_content), 'thinking_lines': thinking_lines, 'answer_after_thinking': answer_in_right_place, 'total_length': len(text)}

def _extract_completion_text(completion):
    if isinstance(completion, list) and len(completion) > 0:
        return completion[0].get('content', '')
    elif isinstance(completion, dict):
        return completion.get('content', '')
    else:
        return str(completion)

def _extract_sections_plan_work_answer(text: str) -> dict:
    """
    Extract PLAN / WORK / ANSWER sections.
    Returns dict with keys: plan, work, answer, has_plan, has_work, has_answer, answer_tail
    """
    out = {'plan': '', 'work': '', 'answer': '', 'has_plan': False, 'has_work': False, 'has_answer': False, 'answer_tail': ''}
    if not text:
        return out
    after_think = text.split('</think>')[-1] if '</think>' in text else text
    pattern = '(?is)(?:^|\\n)\\s*(PLAN|WORK|ANSWER)\\s*:\\s*'
    matches = list(re.finditer(pattern, after_think))
    if not matches:
        return out
    sections = {}
    for i, m in enumerate(matches):
        label = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(after_think)
        sections[label] = after_think[start:end].strip()
    out['plan'] = sections.get('plan', '')
    out['work'] = sections.get('work', '')
    out['answer'] = sections.get('answer', '')
    out['has_plan'] = 'plan' in sections
    out['has_work'] = 'work' in sections
    out['has_answer'] = 'answer' in sections
    out['answer_tail'] = after_think[-300:].strip()
    return out

def _structure_score_plan_work_answer(text: str) -> Tuple[float, dict]:
    """
    Score format quality for PLAN/WORK/ANSWER structure.
    Returns: (score in [0,1], info)
    """
    info = _extract_sections_plan_work_answer(text)
    score = 0.0
    if info['has_plan']:
        score += 0.2
    if info['has_work']:
        score += 0.2
    if info['has_answer']:
        score += 0.2
    plan_len = len(info['plan'])
    work_len = len(info['work'])
    ans_len = len(info['answer'])
    if plan_len >= 20:
        score += 0.1
    if 20 <= plan_len <= 400:
        score += 0.05
    if work_len >= 80:
        score += 0.1
    if 1 <= ans_len <= 200:
        score += 0.05
    answer_pos = text.lower().rfind('answer:')
    if answer_pos != -1 and answer_pos > len(text) * 0.6:
        score += 0.1
    return (min(1.0, score), info)

def _has_boxed_answer(text: str) -> bool:
    after_think = text.split('</think>')[-1] if '</think>' in text else text
    boxed_pattern = '\\\\boxed\\{((?:[^{}]|\\{[^{}]*\\})*)\\}'
    return re.search(boxed_pattern, after_think) is not None

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

def _cheap_repeat_penalty(text: str) -> float:
    """
    Cheap repetition penalty in [0, 0.5].
    Designed to catch loops without heavy n-gram computation.
    """
    if not text:
        return 0.0
    t = re.sub('\\s+', ' ', text.lower()).strip()
    if len(t) < 160:
        return 0.0
    penalty = 0.0
    subs = [t[i:i + 50] for i in range(0, len(t) - 50, 25)]
    if subs:
        uniq_ratio = len(set(subs)) / len(subs)
        if uniq_ratio < 0.55:
            penalty = max(penalty, 0.35)
    words = re.findall('\\w+', t)
    if len(words) > 120:
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top = max(freq.values())
        if top / len(words) > 0.14:
            penalty = max(penalty, 0.3)
    return min(0.5, penalty)

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
    if is_correct:
        score += 40.0
    struct_score, sections = _structure_score_plan_work_answer(text)
    score += 25.0 * struct_score
    fmt = check_format_quality(text)
    has_boxed = _has_boxed_answer(text)
    if fmt['has_think_tags']:
        score += 7.5
    if has_boxed:
        score += 7.5
    work_len = len(sections.get('work', ''))
    if work_len > 0:
        depth_score = min(work_len / 400.0, 1.0) * 10.0
        score += depth_score
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
    dataset_source = kwargs.get('dataset_source', [])
    if not isinstance(dataset_source, list):
        dataset_source = [dataset_source] * len(completions)
    num_generations = len(completions) // len(answer) if len(answer) > 0 else 1
    num_prompts = len(answer)
    rewards = [0.0] * len(completions)
    for prompt_idx in range(num_prompts):
        start = prompt_idx * num_generations
        end = start + num_generations
        correct_answer = answer[prompt_idx]
        group_data = []
        for idx in range(start, end):
            text = _extract_completion_text(completions[idx])
            extracted, _ = extract_answer_GSM8K(text)
            is_correct = compare_answers(extracted, str(correct_answer))
            quality_score = _compute_quality_score(text, extracted, is_correct)
            group_data.append({'idx': idx, 'is_correct': is_correct, 'quality_score': quality_score})
        correct_tier = [d for d in group_data if d['is_correct']]
        wrong_tier = [d for d in group_data if not d['is_correct']]

        def rank_and_assign(tier, reward_min, reward_max):
            """Rank tier by quality and map to [reward_min, reward_max]."""
            if not tier:
                return
            sorted_tier = sorted(tier, key=lambda x: x['quality_score'])
            n = len(sorted_tier)
            if n == 1:
                rewards[sorted_tier[0]['idx']] = reward_max
            else:
                for rank, item in enumerate(sorted_tier):
                    normalized_rank = rank / (n - 1)
                    reward = reward_min + normalized_rank * (reward_max - reward_min)
                    rewards[item['idx']] = reward
        rank_and_assign(correct_tier, 0.0, 1.0)
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
    dataset_source = kwargs.get('dataset_source', [])
    if not isinstance(dataset_source, list):
        dataset_source = [dataset_source] * len(completions)
    num_generations = len(completions) // len(answer) if len(answer) > 0 else 1
    for idx, completion in enumerate(completions):
        prompt_idx = idx // num_generations
        correct_answer = answer[prompt_idx]
        text = _extract_completion_text(completion)
        extracted, _ = extract_answer_GSM8K(text)
        is_correct = compare_answers(extracted, str(correct_answer))
        struct_score, sections = _structure_score_plan_work_answer(text)
        work_len = len(sections.get('work', ''))
        total_len = len(text)
        refusal = bool(re.search("\\b(as an ai|i can't|i cannot|sorry|unable to)\\b", text.lower()))
        repeat_pen = _cheap_repeat_penalty(text)
        if is_correct:
            reward = 0.7
            reward += 0.3 * struct_score
            if work_len > 0:
                depth_factor = min(work_len / 300.0, 1.0)
                reward += 0.1 * depth_factor
        elif extracted is None:
            reward = -0.7
        else:
            reward = -0.5
        if refusal:
            reward = -1.0
            rewards.append(reward)
            continue
        if repeat_pen > 0:
            reward -= min(0.3, repeat_pen)
        if not is_correct and struct_score < 0.4:
            structure_penalty = (0.4 - struct_score) * 0.5
            reward -= structure_penalty
        if total_len < 80:
            reward -= 0.2
        elif total_len > 5000:
            reward -= 0.2
        reward = max(-1.0, min(1.0, reward))
        rewards.append(reward)
    return rewards
REWARD_FUNCTIONS = {'ppo_binary': ppo_reward_binary, 'dapo_structure_balanced': dapo_structure_balanced, 'dapo_rank_stratified': dapo_rank_stratified}

def get_reward_function(name: str):
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f'Unknown reward function: {name}. Available: {list(REWARD_FUNCTIONS)}')
    return REWARD_FUNCTIONS[name]
