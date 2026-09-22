from __future__ import annotations

import wandb
from peft import LoraConfig, TaskType, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from src.core.registry import register


def _apply_peft(model, config: dict):
    task_type = getattr(TaskType, config.get("task_type", "CAUSAL_LM"), TaskType.CAUSAL_LM)
    lora = LoraConfig(
        r=config.get("r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get("target_modules"),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias=config.get("bias", "none"),
        task_type=task_type,
        use_dora=config.get("use_dora", False),
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()
    method = "DoRA" if lora.use_dora else "LoRA"
    print(f"[PEFT] {method}: {trainable:,}/{total:,} trainable parameters")
    return model


def _init_wandb(training: dict, grpo: dict, config: dict) -> None:
    wandb.init(
        project=config["project"],
        entity=config.get("entity"),
        name=config["run_name"],
        tags=config.get("tags", []),
        config={
            "learning_rate": training.get("learning_rate"),
            "batch_size": training.get("per_device_train_batch_size"),
            "max_steps": training.get("max_steps"),
            "num_generations": grpo.get("num_generations"),
            "loss_type": grpo.get("loss_type"),
            "reward_function": grpo.get("reward_function"),
        },
    )


@register("trainer", "trl_grpo")
def build_trl_grpo_trainer(
    model,
    tokenizer,
    dataset,
    training_cfg: dict,
    grpo_cfg: dict,
    wandb_cfg: dict,
    reward_funcs,
    peft_cfg: dict | None = None,
):
    """Build the single GRPO/DAPO training path used by all ablations."""
    _init_wandb(training_cfg, grpo_cfg, wandb_cfg)
    if peft_cfg and peft_cfg.get("enabled"):
        model = _apply_peft(model, peft_cfg)

    training = {**training_cfg, "report_to": ["wandb"]}
    args = GRPOConfig(
        **training,
        num_generations=grpo_cfg["num_generations"],
        max_completion_length=grpo_cfg["max_completion_length"],
        max_prompt_length=grpo_cfg["max_prompt_length"],
        beta=grpo_cfg["beta"],
        scale_rewards=grpo_cfg["scale_rewards"],
        loss_type=grpo_cfg["loss_type"],
        epsilon=grpo_cfg["epsilon"],
        epsilon_high=grpo_cfg.get("epsilon_high"),
        temperature=grpo_cfg["temperature"],
        top_k=grpo_cfg["top_k"],
        top_p=grpo_cfg["top_p"],
        use_vllm=grpo_cfg["use_vllm"],
        vllm_mode=grpo_cfg["vllm_mode"],
        vllm_gpu_memory_utilization=grpo_cfg["vllm_gpu_memory_utilization"],
        vllm_tensor_parallel_size=grpo_cfg.get("vllm_tensor_parallel_size", 1),
        vllm_max_model_length=grpo_cfg.get("vllm_max_model_length"),
        vllm_enable_sleep_mode=grpo_cfg["vllm_enable_sleep_mode"],
    )
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        reward_funcs=reward_funcs,
        args=args,
    )
