import sys
sys.path.insert(0, '.')

from src.rlvr.math_verifier import math_reward_fn, extract_answer


def test_reward_function():
    print("Testing improved reward function...\n")

    test_cases = [
        {
            "name": "Perfect answer with thinking",
            "completion": "<think>\nStep 1: 2 + 2 = 4\nStep 2: Verify the calculation\n</think>\n\nThe answer is 4",
            "ground_truth": "4",
            "expected_high": True,
        },
        {
            "name": "Correct answer, no thinking tags",
            "completion": "Let me calculate. 2 + 2 equals 4. The answer is 4",
            "ground_truth": "4",
            "expected_high": False,
        },
        {
            "name": "Wrong answer with thinking",
            "completion": "<think>\nStep 1: 2 + 2 = 5\n</think>\n\nThe answer is 5",
            "ground_truth": "4",
            "expected_high": False,
        },
        {
            "name": "Malformed tags",
            "completion": "<think>\n2 + 2 = 4\nThe answer is 4",
            "ground_truth": "4",
            "expected_high": False,
        },
        {
            "name": "No answer extractable",
            "completion": "<think>\nI'm thinking about this problem...\n</think>\n\nIt's complicated.",
            "ground_truth": "4",
            "expected_high": False,
        },
        {
            "name": "Extremely short response",
            "completion": "4",
            "ground_truth": "4",
            "expected_high": False,
        },
        {
            "name": "Good reasoning with boxed answer",
            "completion": "<think>\nStep 1: Calculate 2 + 2\nStep 2: The result is 4\n</think>\n\n\\boxed{4}",
            "ground_truth": "4",
            "expected_high": True,
        },
    ]

    print("=" * 80)
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test['name']}")
        print(f"Completion (first 100 chars): {test['completion'][:100]}...")
        print(f"Ground truth: {test['ground_truth']}")

        completions = [[{"role": "assistant", "content": test["completion"]}]]
        answers = [test["ground_truth"]]

        rewards = math_reward_fn([], completions, answers)
        reward_value = rewards[0][0]

        extracted, confidence = extract_answer(test["completion"])
        print(f"Extracted answer: {extracted} (confidence: {confidence:.2f})")
        print(f"Reward: {reward_value:.3f}")

        if test["expected_high"]:
            status = "✓ PASS" if reward_value > 0.8 else "✗ FAIL (expected high reward)"
        else:
            status = "✓ PASS" if reward_value < 0.8 else "✗ FAIL (expected low/negative reward)"

        print(f"Status: {status}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("Reward Function Component Breakdown:")
    print("=" * 80)
    print("Correctness:          +1.0   (correct answer)")
    print("Wrong answer:         -0.1   (extractable but wrong)")
    print("No answer:            -0.3   (nothing extractable)")
    print()
    print("Format bonuses:")
    print("  Complete tags:      +0.15  (has <think></think>)")
    print("  Answer placement:   +0.1   (answer after </think>)")
    print("  Confidence bonus:   +0.1   (max, scaled by extraction confidence)")
    print()
    print("Format penalties:")
    print("  Partial tags:       -0.2   (only <think> or </think>, not both)")
    print("  Too short (<50):    -0.2   (rushed response)")
    print("  Too long (>2000):   -0.15  (if <5 thinking lines)")
    print("  Too few steps (<2): -0.1   (insufficient reasoning)")
    print("  Too many steps (>50):-0.05  (overthinking)")
    print()
    print("Best possible reward:  ~1.35 (correct + all bonuses)")
    print("Worst possible reward: ~-0.8 (wrong + all penalties)")
    print("=" * 80)


if __name__ == "__main__":
    test_reward_function()
