from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config import Config
from src.core.registry import build


def _latest_checkpoint(output_dir: str) -> str | None:
    checkpoints = []
    for path in Path(output_dir).glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return str(max(checkpoints)[1]) if checkpoints else None


def main(experiment: str) -> None:
    import src.builders.data  # noqa: F401 - imports register the builders
    import src.builders.model  # noqa: F401
    import src.builders.tokenizer  # noqa: F401
    import src.builders.trainer  # noqa: F401
    from src.rlvr.math_verifier import get_reward_function
    from src.rlvr.rl_callbacks import GSMTopKCheckpointCallback

    config = Config.from_experiment(experiment)
    print(
        f"[RLVR] {config.run['name']} | {config.grpo['loss_type']} | "
        f"K={config.grpo['num_generations']} | reward={config.grpo['reward_function']}"
    )

    tokenizer = build("tokenizer", **config.tokenizer)
    model = build("model", **config.model)
    dataset = build("data", tokenizer=tokenizer, **config.data)
    reward = get_reward_function(config.grpo["reward_function"])
    trainer = build(
        "trainer",
        type="trl_grpo",
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_cfg=config.training,
        grpo_cfg=config.grpo,
        wandb_cfg=config.wandb,
        reward_funcs=reward,
        peft_cfg=config.peft,
    )

    selection = config.topk_eval
    if selection.get("enabled"):
        if selection.get("mode") != "gsm":
            raise ValueError("This research snapshot supports only GSM checkpoint selection")
        trainer.add_callback(
            GSMTopKCheckpointCallback(
                k=selection["k"],
                eval_dataset_path=selection["eval_dataset_path"],
                eval_every_n_steps=selection["eval_every_n_steps"],
                reward_function=selection["reward_function"],
                source_type=selection["source_type"],
                output_dir=config.training["output_dir"],
                trainer=trainer,
                tokenizer=tokenizer,
                eval_batch_size=selection["eval_batch_size"],
            )
        )

    checkpoint = _latest_checkpoint(config.training["output_dir"])
    print(f"[RLVR] {'Resuming from ' + checkpoint if checkpoint else 'Starting a new run'}")
    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a DAPO/GRPO math ablation")
    parser.add_argument(
        "--exp",
        required=True,
        help="Config path below configs/exp without the .yaml suffix",
    )
    main(parser.parse_args().exp)
