import wandb
from src.core.registry import register
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, GRPOTrainer, GRPOConfig
import torch
from typing import Any, Dict, List, Optional


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
def build_trl_grpo_trainer(model, tokenizer, dataset, training_cfg, grpo_cfg, wandb_cfg, reward_funcs):
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
    
    config = GRPOConfig(
        **training_cfg,
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 512),
        max_prompt_length=grpo_cfg.get("max_prompt_length", 1024),
        temperature=grpo_cfg.get("temperature", 0.7),
        use_vllm=grpo_cfg.get("use_vllm", False),
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
