from pathlib import Path

import pytest

from src.core.config import Config, PROJECT_ROOT


EXPERIMENT_ROOT = PROJECT_ROOT / "configs" / "exp" / "EA_RLVR_ABLATIONS"
EXPERIMENTS = [
    path.relative_to(PROJECT_ROOT / "configs" / "exp").with_suffix("").as_posix()
    for path in EXPERIMENT_ROOT.rglob("*.yaml")
    if not path.name.startswith("_")
]


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_resolves(experiment):
    config = Config.from_experiment(experiment)

    assert config.run["mode"] == "rlvr"
    assert "${" not in config.training["output_dir"]
    assert config.data["train_path"] == "allenai/RLVR-GSM"
    assert config.grpo["reward_function"] in {
        "ppo_binary",
        "dapo_rank_stratified",
        "dapo_structure_balanced",
    }
    assert config.topk_eval["eval_dataset_path"] == "held_out/gsm_eval_v2.jsonl"


def test_phase_one_axes_are_preserved():
    configs = [
        Config.from_experiment(name)
        for name in EXPERIMENTS
        if "/p1_" in f"/{name}"
    ]
    assert {config.grpo["loss_type"] for config in configs} == {"grpo", "dapo"}
    assert {config.grpo["num_generations"] for config in configs} == {8, 16, 32}
    assert {config.grpo["beta"] for config in configs} == {0.0, 0.03}


def test_phase_two_axes_are_preserved():
    configs = [
        Config.from_experiment(name)
        for name in EXPERIMENTS
        if "/p2/" in f"/{name}/"
    ]
    assert all(config.grpo["loss_type"] == "dapo" for config in configs)
    assert all(config.grpo["num_generations"] == 32 for config in configs)
    assert {config.grpo["reward_function"] for config in configs} == {
        "ppo_binary",
        "dapo_rank_stratified",
        "dapo_structure_balanced",
    }
