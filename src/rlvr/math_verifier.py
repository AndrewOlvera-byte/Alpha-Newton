"""
Math Verifier for RLVR

Extracts numeric answers from LLM completions and compares against ground truth.
Returns binary rewards for use with GRPOTrainer.
"""

import re


def extract_answer(text: str) -> str | None:
    """
    Extract final numeric answer from model generation.
    
    Handles common formats:
    - \\boxed{42} or \\boxed{3.14}
    - "The answer is 42"
    - "= 42" at end of solution
    - Plain number as last token
    
    Returns normalized string for comparison (strips whitespace, trailing zeros).
    """
    if not text:
        return None
    
    # 1. Try \boxed{...} first (most reliable for math models)
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return _normalize_number(boxed_match.group(1))
    
    # 2. Try "answer is X" pattern
    answer_match = re.search(r'(?:answer|result|solution)\s*(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if answer_match:
        return _normalize_number(answer_match.group(1))
    
    # 3. Try "= X" at end of text
    equals_match = re.search(r'=\s*([+-]?\d+(?:\.\d+)?)\s*$', text)
    if equals_match:
        return _normalize_number(equals_match.group(1))
    
    # 4. Fallback: last number in text
    all_numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
    if all_numbers:
        return _normalize_number(all_numbers[-1])
    
    return None


def _normalize_number(s: str) -> str:
    """Normalize number string for comparison (handle floats, trailing zeros)."""
    s = s.strip()
    try:
        # Convert to float then back to handle "3.0" == "3"
        num = float(s)
        # Return int if whole number, else float with minimal decimals
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return s


def math_reward_fn(prompts, completions, answer, **kwargs):
    """
    TRL-compatible reward function for math verification.
    
    Args:
        prompts: List of prompt dicts (not used, but required by TRL)
        completions: List of completion lists, each containing message dicts
                     Format: [[{"role": "assistant", "content": "..."}], ...]
        answer: List of ground truth answers from dataset
        **kwargs: Additional columns from dataset (ignored)
    
    Returns:
        List of reward lists: [[reward], [reward], ...] where reward is 1.0 or 0.0
    """
    rewards = []
    
    for completion, correct_answer in zip(completions, answer):
        # Extract text from completion
        if isinstance(completion, list) and len(completion) > 0:
            text = completion[0].get("content", "")
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)
        
        # Extract and compare
        extracted = extract_answer(text)
        correct = _normalize_number(str(correct_answer).strip())
        
        reward = 1.0 if extracted == correct else 0.0
        rewards.append([reward])
    
    return rewards
