import wandb
from src.core.registry import register
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, GRPOTrainer, GRPOConfig
import torch
from typing import Any, Dict, List, Optional
from vllm import SamplingParams


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
        beta=grpo_cfg.get("beta", 0.001),
        scale_rewards=grpo_cfg.get("scale_rewards", "group"),
        loss_type=grpo_cfg.get("loss_type", "dapo"),
        epsilon=grpo_cfg.get("epsilon", 0.2),
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


def create_diverse_rollout_func(tokenizer, diverse_config, grpo_config):
    """
    Create a custom rollout function for diverse sampling with vLLM n>1 batching

    Returns a function with signature: rollout_func(queries, trainer) -> dict
    """
    sampling_configs = diverse_config.get("sampling_configs", [
        {"temperature": 0.7, "top_p": 0.95}
    ])
    all_system_prompts = diverse_config.get("system_prompts", [
        "Think step by step and show your reasoning."
    ])
    modes_per_cycle = diverse_config.get("modes_per_cycle", len(all_system_prompts))
    system_prompts = all_system_prompts[:modes_per_cycle]

    max_completion_length = grpo_config.get("max_completion_length", 512)
    max_prompt_length = grpo_config.get("max_prompt_length", 512)
    use_vllm = grpo_config.get("use_vllm", False)

    # Calculate efficiency metrics
    num_configs = len(sampling_configs)
    num_modes = len(system_prompts)
    cycle_length = num_configs * num_modes

    print(f"\n[DiverseRollout] Initialized:")
    print(f"  Sampling configs: {num_configs}")
    for i, cfg in enumerate(sampling_configs):
        print(f"    Config {i}: temp={cfg['temperature']:.2f}, top_p={cfg['top_p']:.2f}")
    print(f"  System prompts: {num_modes} active (from {len(all_system_prompts)} total)")
    print(f"  Use vLLM: {use_vllm}")
    print(f"  vLLM n>1 batching: {'Enabled' if use_vllm else 'Disabled (using HF generate)'}")

    def format_with_system_prompt(question: str, system_prompt: str) -> str:
        """Format question with system prompt using chat template."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def rollout_func_vllm(queries, trainer):
        llm = trainer.llm
        num_generations = trainer.num_generations if trainer.model.training else trainer.num_generations_eval
        raw_questions = queries
        num_prompts = len(raw_questions)

        print(f"\n[DiverseRollout-vLLM] Generating {num_generations} completions for {num_prompts} prompts")

        generation_grid = []
        for gen_idx in range(num_generations):
            config_idx = gen_idx % num_configs
            mode_idx = (gen_idx // num_configs) % num_modes
            generation_grid.append((gen_idx, config_idx, mode_idx))

        formatted_cache = {}
        for mode_idx in range(num_modes):
            system_prompt = system_prompts[mode_idx]
            formatted_cache[mode_idx] = [
                format_with_system_prompt(question, system_prompt)
                for question in raw_questions
            ]

        # Group by (config, mode) to leverage n>1 batching
        config_mode_groups = {}
        for gen_idx, config_idx, mode_idx in generation_grid:
            key = (config_idx, mode_idx)
            if key not in config_mode_groups:
                config_mode_groups[key] = []
            config_mode_groups[key].append(gen_idx)

        all_query_tensors = [None] * (num_prompts * num_generations)
        all_response_tensors = [None] * (num_prompts * num_generations)

        print(f"  Efficiency: {num_generations} gens -> {len(config_mode_groups)} vLLM calls")

        for (config_idx, mode_idx), gen_indices in sorted(config_mode_groups.items()):
            config = sampling_configs[config_idx]
            gens_for_group = len(gen_indices)

            formatted_prompts = formatted_cache[mode_idx]

            sampling_params = SamplingParams(
                temperature=config["temperature"],
                top_p=config["top_p"],
                max_tokens=max_completion_length,
                n=gens_for_group,
            )

            mode_preview = system_prompts[mode_idx][:30] + "..." if len(system_prompts[mode_idx]) > 30 else system_prompts[mode_idx]
            print(f"  [{config_idx},{mode_idx}]: temp={config['temperature']:.2f}, top_p={config['top_p']:.2f}, "
                  f"n={gens_for_group}, mode='{mode_preview}'")
            print(f"    Prompts: {len(formatted_prompts)} × n={gens_for_group} = {len(formatted_prompts) * gens_for_group} completions in 1 call")

            outputs = llm.generate(formatted_prompts, sampling_params)

            # Extract tokens from vLLM outputs
            gen_indices_sorted = sorted(gen_indices)
            for prompt_idx, output in enumerate(outputs):
                prompt_token_ids = output.prompt_token_ids

                for local_gen_idx, completion in enumerate(output.outputs):
                    gen_idx = gen_indices_sorted[local_gen_idx]
                    position = prompt_idx * num_generations + gen_idx

                    # Convert to tensors
                    query_tensor = torch.tensor(prompt_token_ids, dtype=torch.long)
                    response_tensor = torch.tensor(completion.token_ids, dtype=torch.long)

                    all_query_tensors[position] = query_tensor
                    all_response_tensors[position] = response_tensor

        print(f"[DiverseRollout-vLLM] Generated {len(all_response_tensors)} completions\n")

        return {
            "query_tensors": all_query_tensors,
            "response_tensors": all_response_tensors,
        }

    def rollout_func_hf(queries, trainer):
        """
        HuggingFace generate fallback: Replicate prompts and use model.generate

        less efficient but works when vLLM is not available
        """
        model = trainer.model
        num_generations = trainer.num_generations if trainer.model.training else trainer.num_generations_eval
        raw_questions = queries
        num_prompts = len(raw_questions)

        print(f"\n[DiverseRollout-HF] Generating {num_generations} completions for {num_prompts} prompts")

        generation_grid = []
        for gen_idx in range(num_generations):
            config_idx = gen_idx % num_configs
            mode_idx = (gen_idx // num_configs) % num_modes
            generation_grid.append((gen_idx, config_idx, mode_idx))

        formatted_cache = {}
        for mode_idx in range(num_modes):
            system_prompt = system_prompts[mode_idx]
            formatted_cache[mode_idx] = [
                format_with_system_prompt(question, system_prompt)
                for question in raw_questions
            ]

        # Group by (config, mode)
        config_mode_groups = {}
        for gen_idx, config_idx, mode_idx in generation_grid:
            key = (config_idx, mode_idx)
            if key not in config_mode_groups:
                config_mode_groups[key] = []
            config_mode_groups[key].append(gen_idx)

        all_query_tensors = [None] * (num_prompts * num_generations)
        all_response_tensors = [None] * (num_prompts * num_generations)

        device = next(model.parameters()).device

        for (config_idx, mode_idx), gen_indices in sorted(config_mode_groups.items()):
            config = sampling_configs[config_idx]
            gens_for_group = len(gen_indices)

            formatted_prompts = formatted_cache[mode_idx]

            encoded = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length
            )
            prompt_tokens = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)

            replicated_prompts = prompt_tokens.repeat_interleave(gens_for_group, dim=0)
            replicated_attention_mask = attention_mask.repeat_interleave(gens_for_group, dim=0)

            print(f"  [{config_idx},{mode_idx}]: temp={config['temperature']:.2f}, top_p={config['top_p']:.2f}, "
                  f"gens={gens_for_group}, batch={replicated_prompts.shape[0]}")

            gen_kwargs = {
                "attention_mask": replicated_attention_mask,
                "max_new_tokens": max_completion_length,
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            model.eval()
            with torch.no_grad():
                outputs = model.generate(replicated_prompts, **gen_kwargs)

            prompt_lens = (replicated_prompts != tokenizer.pad_token_id).sum(dim=1)

            gen_indices_sorted = sorted(gen_indices)
            for local_gen_idx in range(gens_for_group):
                gen_idx = gen_indices_sorted[local_gen_idx]
                for prompt_idx in range(num_prompts):
                    output_idx = local_gen_idx * num_prompts + prompt_idx
                    position = prompt_idx * num_generations + gen_idx

                    prompt_len = prompt_lens[output_idx].item()
                    full_output = outputs[output_idx]

                    query_tensor = full_output[:prompt_len]

                    response_tensor = full_output[prompt_len:]

                    all_query_tensors[position] = query_tensor
                    all_response_tensors[position] = response_tensor

        print(f"[DiverseRollout-HF] Generated {len(all_response_tensors)} completions\n")

        # Return in TRL's expected format
        return {
            "query_tensors": all_query_tensors,
            "response_tensors": all_response_tensors,
        }

    def rollout_func(queries, trainer):
        if hasattr(trainer, 'llm') and trainer.llm is not None:
            print("Using vLLM generation")
            return rollout_func_vllm(queries, trainer)
        else:
            print("Using HF generation")
            return rollout_func_hf(queries, trainer)

    return rollout_func


@register("trainer", "trl_grpo_diverse")
def build_trl_grpo_diverse_trainer(model, tokenizer, dataset, training_cfg, grpo_cfg, wandb_cfg, reward_funcs):
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

    config = GRPOConfig(
        **training_cfg,
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 512),
        max_prompt_length=grpo_cfg.get("max_prompt_length", 1024),
        beta=grpo_cfg.get("beta", 0.001),
        scale_rewards=grpo_cfg.get("scale_rewards", "group"),
        loss_type=grpo_cfg.get("loss_type", "dapo"),
        epsilon=grpo_cfg.get("epsilon", 0.2),
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
