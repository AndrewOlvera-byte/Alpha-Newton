import wandb
from src.core.registry import register
from src.builders.trainer_rollout import create_diverse_rollout_func
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, GRPOTrainer, GRPOConfig
import torch
from typing import Any, Dict, List, Optional
from vllm import SamplingParams


def _apply_lora(model, peft_cfg: dict):
    """Wrap model with LoRA adapters from a peft config dict.

    Expected peft_cfg keys (all optional beyond defaults):
        r                : int   - LoRA rank (default 16)
        lora_alpha       : int   - scaling factor (default 32)
        target_modules   : list  - module names to adapt (default None → auto)
        lora_dropout     : float - dropout on LoRA layers (default 0.05)
        bias             : str   - "none" | "all" | "lora_only" (default "none")
        task_type        : str   - PEFT task type (default "CAUSAL_LM")
    """
    from peft import LoraConfig, get_peft_model, TaskType

    task_type_str = peft_cfg.get("task_type", "CAUSAL_LM")
    task_type = getattr(TaskType, task_type_str, TaskType.CAUSAL_LM)

    use_dora = peft_cfg.get("use_dora", False)

    lora_config = LoraConfig(
        r=peft_cfg.get("r", 16),
        lora_alpha=peft_cfg.get("lora_alpha", 32),
        target_modules=peft_cfg.get("target_modules", None),
        lora_dropout=peft_cfg.get("lora_dropout", 0.05),
        bias=peft_cfg.get("bias", "none"),
        task_type=task_type,
        use_dora=use_dora,
    )

    model = get_peft_model(model, lora_config)

    # Required when gradient checkpointing is active so gradients flow into adapters
    model.enable_input_require_grads()

    method = "DoRA" if use_dora else "LoRA"
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[PEFT] {method} applied — trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    print(f"[PEFT] {method} config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
          f"dropout={lora_config.lora_dropout}, bias={lora_config.bias}")
    if lora_config.target_modules:
        print(f"[PEFT] Target modules: {list(lora_config.target_modules)}")

    return model


class DataCollatorForCausalLMWithLabels:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]):
        labels = []
        features_wo_labels = []

        for f in features:
            lab = f.get("labels")
            if isinstance(lab, torch.Tensor):
                lab = lab.tolist()
            labels.append(lab)
            features_wo_labels.append({k: v for k, v in f.items() if k != "labels"})

        batch = self.tokenizer.pad(
            features_wo_labels,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        pad_side = getattr(self.tokenizer, "padding_side", "right")
        padded_labels = []

        for lab in labels:
            lab = lab or []
            if len(lab) > max_len:
                lab = lab[:max_len]
            pad_len = max_len - len(lab)
            if pad_side == "left":
                padded_labels.append(([-100] * pad_len) + lab)
            else:
                padded_labels.append(lab + ([-100] * pad_len))

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

@register("trainer", "trl_sft")
def build_trl_sft_trainer(model, tokenizer, dataset, training_cfg, wandb_cfg):
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

    training_cfg = {**training_cfg}
    training_cfg["report_to"] = ["wandb"]

    training_cfg.pop("beta", None)

    training_args = SFTConfig(**training_cfg)

    data_collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        args=training_args,
        data_collator=data_collator,
    )
    return trainer


@register("trainer", "trl_dpo")
def build_trl_dpo_trainer(model, tokenizer, dataset, training_cfg, wandb_cfg, ref_model=None):
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

    training_cfg = {**training_cfg}
    training_cfg["report_to"] = ["wandb"]

    beta = training_cfg.pop("beta", 0.1)
    max_length = training_cfg.pop("max_length", 2048)
    max_prompt_length = training_cfg.pop("max_prompt_length", max_length // 3)
    precompute_ref_log_probs = training_cfg.pop("precompute_ref_log_probs", True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    training_args = DPOConfig(
        **training_cfg,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        precompute_ref_log_probs=precompute_ref_log_probs,
        truncation_mode="keep_end",
    )

    if precompute_ref_log_probs:
        print("[DPO Trainer] Precomputing reference log probabilities")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        args=training_args,
    )
    return trainer


@register("trainer", "trl_grpo")
def build_trl_grpo_trainer(model, tokenizer, dataset, training_cfg, grpo_cfg, wandb_cfg, reward_funcs, peft_cfg=None):
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
            "num_generations": grpo_cfg.get("num_generations", 4),
        },
    )
    
    training_cfg = {**training_cfg}
    training_cfg["report_to"] = ["wandb"]

    training_cfg.pop("beta", None)

    if peft_cfg and peft_cfg.get("enabled", False):
        model = _apply_lora(model, peft_cfg)

    config = GRPOConfig(
        **training_cfg,
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 512),
        max_prompt_length=grpo_cfg.get("max_prompt_length", 1024),
        beta=grpo_cfg.get("beta", 0.001),
        scale_rewards=grpo_cfg.get("scale_rewards", "group"),
        loss_type=grpo_cfg.get("loss_type", "dapo"),
        epsilon=grpo_cfg.get("epsilon", 0.2),
        epsilon_high=grpo_cfg.get("epsilon_high", None),
        temperature=grpo_cfg.get("temperature", 0.7),
        top_k=grpo_cfg.get("top_k", 0),
        top_p=grpo_cfg.get("top_p", 1.0),
        # vLLM configuration
        use_vllm=grpo_cfg.get("use_vllm", False),
        vllm_mode=grpo_cfg.get("vllm_mode", "server"),  # "server" or "colocate"
        vllm_gpu_memory_utilization=grpo_cfg.get("vllm_gpu_memory_utilization", 0.3),
        vllm_tensor_parallel_size=grpo_cfg.get("vllm_tensor_parallel_size", 1),
        vllm_max_model_length=grpo_cfg.get("vllm_max_model_length", None),
        vllm_enable_sleep_mode=grpo_cfg.get("vllm_enable_sleep_mode", False),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        reward_funcs=reward_funcs,
        args=config,
    )

    return trainer

@register("trainer", "trl_grpo_diverse")
def build_trl_grpo_diverse_trainer(model, tokenizer, dataset, training_cfg, grpo_cfg, wandb_cfg, reward_funcs, peft_cfg=None):
    """
    Build GRPO trainer with diverse sampling support

    Cycles through discrete sampling configs deterministically for:
    - Reproducible diversity (not random)
    - Efficient vLLM batching (grouped generation)
    - Stable RL training (reduced variance)

    With 16 generations and 4 configs: 4 vLLM calls with n=4 each (fast)

    Config example:
        grpo:
          diverse_sampling:
            enabled: true
            sampling_configs:
              - temperature: 0.7
                top_p: 0.92
              - temperature: 0.85
                top_p: 0.95
              - temperature: 1.0
                top_p: 0.97
            system_prompts:
              - "Think step by step."
              - "Work backwards from the answer."
              - "Use equations to solve."

    Args:
        model: The model to train
        tokenizer: Tokenizer instance
        dataset: Dataset with 'question' field (raw questions, not pre-formatted)
        training_cfg: Training configuration dict
        grpo_cfg: GRPO configuration dict (must contain diverse_sampling section)
        wandb_cfg: W&B configuration dict
        reward_funcs: Reward function for verification

    Returns:
        GRPOTrainer instance with custom rollout_func
    """
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
            "num_generations": grpo_cfg.get("num_generations", 4),
            "diverse_sampling": grpo_cfg.get("diverse_sampling", {}),
        },
    )

    training_cfg = {**training_cfg}
    training_cfg["report_to"] = ["wandb"]

    training_cfg.pop("beta", None)

    if peft_cfg and peft_cfg.get("enabled", False):
        model = _apply_lora(model, peft_cfg)

    config = GRPOConfig(
        **training_cfg,
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 512),
        max_prompt_length=grpo_cfg.get("max_prompt_length", 1024),
        beta=grpo_cfg.get("beta", 0.001),
        scale_rewards=grpo_cfg.get("scale_rewards", "group"),
        loss_type=grpo_cfg.get("loss_type", "dapo"),
        epsilon=grpo_cfg.get("epsilon", 0.2),
        epsilon_high=grpo_cfg.get("epsilon_high", None),
        temperature=grpo_cfg.get("temperature", 0.7),
        top_k=grpo_cfg.get("top_k", 0),
        top_p=grpo_cfg.get("top_p", 1.0),
        # vLLM configuration
        use_vllm=grpo_cfg.get("use_vllm", False),
        vllm_mode=grpo_cfg.get("vllm_mode", "server"),
        vllm_gpu_memory_utilization=grpo_cfg.get("vllm_gpu_memory_utilization", 0.3),
        vllm_tensor_parallel_size=grpo_cfg.get("vllm_tensor_parallel_size", 1),
        vllm_max_model_length=grpo_cfg.get("vllm_max_model_length", None),
        vllm_enable_sleep_mode=grpo_cfg.get("vllm_enable_sleep_mode", False),
    )

    diverse_config = grpo_cfg.get("diverse_sampling", {})

    rollout_func = create_diverse_rollout_func(
        tokenizer=tokenizer,
        diverse_config=diverse_config,
        grpo_config=grpo_cfg
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        reward_funcs=reward_funcs,
        args=config,
        rollout_func=rollout_func,
    )

    print(f"\n[RLVR] Using rollout_func for diverse sampling")
    print(f"       TRL handles batch replication, we control generation")

    return trainer
