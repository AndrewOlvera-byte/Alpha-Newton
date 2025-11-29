import wandb
from src.core.registry import register
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig


@register("trainer", "trl_sft")
def build_trl_sft_trainer(model, tokenizer, dataset, training_cfg, wandb_cfg):
    """
    Build SFT trainer with WandB integration

    WandB logs all metrics automatically. Model checkpoints save to output_dir locally.
    """
    # Initialize WandB run
    wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=wandb_cfg["run_name"],
        tags=wandb_cfg.get("tags", []),
        config={
            "model": training_cfg.get("output_dir", "").split("/")[-1],
            "learning_rate": training_cfg.get("learning_rate"),
            "batch_size": training_cfg.get("per_device_train_batch_size"),
            "max_steps": training_cfg.get("max_steps"),
        },
    )

    # Enable WandB reporting in trainer
    training_cfg = {**training_cfg}
    training_cfg["report_to"] = ["wandb"]

    # Remove DPO-specific params (beta is only for DPO)
    training_cfg.pop("beta", None)

    training_args = SFTConfig(**training_cfg)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # Renamed from 'tokenizer' in TRL 0.12.0+
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        args=training_args,
    )
    return trainer


@register("trainer", "trl_dpo")
def build_trl_dpo_trainer(model, tokenizer, dataset, training_cfg, wandb_cfg, ref_model=None):
    """
    Build DPO (Direct Preference Optimization) trainer

    DPO directly optimizes on preference pairs without needing a reward model.
    Works with datasets containing (prompt, chosen, rejected) tuples.

    Args:
        model: Policy model to train
        tokenizer: Tokenizer
        dataset: Dict with 'train' and 'eval' datasets containing preference pairs
        training_cfg: Training configuration (must include 'beta' for DPO temperature)
        wandb_cfg: WandB configuration
        ref_model: Reference model (optional - will use frozen copy of model if None)
    """
    # Initialize WandB run
    wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=wandb_cfg["run_name"],
        tags=wandb_cfg.get("tags", []),
        config={
            "model": training_cfg.get("output_dir", "").split("/")[-1],
            "learning_rate": training_cfg.get("learning_rate"),
            "batch_size": training_cfg.get("per_device_train_batch_size"),
            "max_steps": training_cfg.get("max_steps"),
            "beta": training_cfg.get("beta", 0.1),
        },
    )

    # Enable WandB reporting
    training_cfg = {**training_cfg}
    training_cfg["report_to"] = ["wandb"]

    # Extract DPO-specific beta parameter
    beta = training_cfg.pop("beta", 0.1)

    training_args = DPOConfig(
        **training_cfg,
        beta=beta,  # DPO temperature (higher = more conservative, typical: 0.1-0.5)
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # If None, DPOTrainer creates a frozen copy automatically
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        args=training_args,
    )
    return trainer
