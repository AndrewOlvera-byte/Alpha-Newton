import pytest

from src.rlvr.math_verifier import (
    compare_answers,
    extract_answer_typed,
    get_reward_function,
)


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("<think>2 + 2 = 4</think>\\n\\boxed{4}", "4"),
        ("The final answer is 3/4.", "3/4"),
        ("After checking the work, the answer is -12.", "-12"),
    ],
)
def test_answer_extraction(completion, expected):
    extracted, _, _, _ = extract_answer_typed(completion)
    assert compare_answers(extracted, expected)


def test_binary_reward_separates_correct_and_incorrect_answers():
    reward = get_reward_function("ppo_binary")
    scores = reward(
        prompts=["", ""],
        completions=[r"\boxed{4}", r"\boxed{5}"],
        answer=["4", "4"],
    )
    assert scores[0] > 0.5
    assert scores[1] <= 0.5


@pytest.mark.parametrize(
    "name",
    ["ppo_binary", "dapo_rank_stratified", "dapo_structure_balanced"],
)
def test_published_reward_names_resolve(name):
    assert callable(get_reward_function(name))


def test_unknown_reward_is_rejected():
    with pytest.raises(ValueError, match="Unknown reward function"):
        get_reward_function("not-a-reward")
