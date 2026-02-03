"""
Clean rollout function for diverse sampling with configurable batching strategies.

CRITICAL FIX: TRL already replicates queries before calling rollout_func!
When batch_size=2 and num_generations=16, rollout_func receives 32 queries.
Generate ONE completion per query (not num_generations per query).
"""
import torch
import time


def create_diverse_rollout_func(tokenizer, diverse_config, grpo_config):
    """
    Create a custom rollout function for diverse sampling.

    CRITICAL: TRL replicates queries internally before calling rollout_func.
    If batch_size=2 and num_generations=16, rollout_func receives 32 queries.
    We should generate ONE completion per query (not replicate again).

    Returns:
        function: rollout_func(queries, trainer) -> dict with prompt_ids, completion_ids, logprobs
    """
    sampling_configs = diverse_config.get("sampling_configs", [
        {"temperature": 0.7, "top_p": 0.95}
    ])
    all_system_prompts = diverse_config.get("system_prompts", [
        "Think step by step and show your reasoning."
    ])
    modes_per_cycle = diverse_config.get("modes_per_cycle", len(all_system_prompts))
    system_prompts = all_system_prompts[:modes_per_cycle]
    batching_strategy = diverse_config.get("batching_strategy", "grouped")

    max_completion_length = grpo_config.get("max_completion_length", 512)
    max_prompt_length = grpo_config.get("max_prompt_length", 512)
    use_vllm = grpo_config.get("use_vllm", False)

    num_configs = len(sampling_configs)
    num_modes = len(system_prompts)
    num_groups = num_configs * num_modes

    print(f"\n[DiverseRollout] Initialized:")
    print(f"  Strategy: {batching_strategy}")
    print(f"  Sampling configs: {num_configs}")
    for i, cfg in enumerate(sampling_configs):
        print(f"    [{i}] temp={cfg['temperature']:.2f}, top_p={cfg['top_p']:.2f}")
    print(f"  System prompts: {num_modes} active")
    for i, prompt in enumerate(system_prompts):
        preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
        print(f"    [{i}] {preview}")
    print(f"  vLLM: {use_vllm} (via trainer._generate)")
    print(f"  Total groups: {num_groups} ({num_configs} configs × {num_modes} prompts)")
    print(f"  IMPORTANT: TRL pre-replicates queries, we generate 1 completion per query")

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

    def rollout_func(queries, trainer):
        """
        Generate diverse completions using trainer._generate (vLLM path).

        CRITICAL: TRL already replicates queries before calling this function!
        If batch_size=2 and num_generations=16, queries will have 32 items.
        We should generate ONE completion per query (not replicate again).

        Args:
            queries: List of ALREADY REPLICATED query strings (length = batch_size × num_generations)
            trainer: GRPOTrainer instance

        Returns:
            dict: {prompt_ids: List[Tensor], completion_ids: List[Tensor], logprobs: List}
        """
        num_queries = len(queries)
        num_generations = trainer.num_generations if trainer.model.training else trainer.num_generations_eval

        # Infer original batch size
        if num_queries % num_generations != 0:
            raise ValueError(f"num_queries={num_queries} not divisible by num_generations={num_generations}")
        batch_size = num_queries // num_generations

        print(f"\n[DiverseRollout] Received {num_queries} queries (batch_size={batch_size} × num_generations={num_generations})")
        print(f"  Generating {num_queries} diverse completions (1 per query)")

        # Map each replicated query to a (config, mode) pair
        # queries[0:num_generations] are replicates of original query 0
        # queries[num_generations:2*num_generations] are replicates of original query 1, etc.
        query_assignments = []
        for query_idx in range(num_queries):
            original_idx = query_idx // num_generations  # which original query
            gen_idx = query_idx % num_generations        # which generation for that query
            config_idx = gen_idx % num_configs
            mode_idx = (gen_idx // num_configs) % num_modes
            query_assignments.append((query_idx, original_idx, gen_idx, config_idx, mode_idx))

        # Group by (config, mode) for batched generation
        config_mode_groups = {}
        for query_idx, original_idx, gen_idx, config_idx, mode_idx in query_assignments:
            key = (config_idx, mode_idx)
            if key not in config_mode_groups:
                config_mode_groups[key] = []
            config_mode_groups[key].append((query_idx, original_idx))

        # Storage for results (in original query order)
        all_prompt_ids = [None] * num_queries
        all_completion_ids = [None] * num_queries
        all_logprobs = [None] * num_queries

        print(f"  Diverse sampling: {num_configs} configs × {num_modes} modes = {len(config_mode_groups)} groups")

        # Save original trainer params
        orig_temp = trainer.temperature
        orig_top_p = trainer.top_p

        # Generate for each (config, mode) group
        for (config_idx, mode_idx), query_infos in sorted(config_mode_groups.items()):
            config = sampling_configs[config_idx]
            system_prompt = system_prompts[mode_idx]
            num_in_group = len(query_infos)

            # Format prompts for this group
            formatted_prompts = []
            query_indices = []
            for query_idx, original_idx in query_infos:
                formatted_prompt = format_with_system_prompt(queries[query_idx], system_prompt)
                formatted_prompts.append(formatted_prompt)
                query_indices.append(query_idx)

            mode_preview = system_prompt[:30] + "..." if len(system_prompt) > 30 else system_prompt
            print(f"    Group[{config_idx},{mode_idx}] temp={config['temperature']:.2f} top_p={config['top_p']:.2f} "
                  f"n={num_in_group} mode='{mode_preview}'")

            # Set trainer sampling params for this group
            trainer.temperature = config["temperature"]
            trainer.top_p = config["top_p"]

            # CRITICAL: Temporarily disable rollout_func to avoid recursion
            orig_rollout_func = trainer.rollout_func
            trainer.rollout_func = None

            # Generate using TRL's internal method
            start_time = time.time()
            result = trainer._generate_single_turn(formatted_prompts)
            elapsed = time.time() - start_time

            # Restore rollout_func
            trainer.rollout_func = orig_rollout_func

            # Unpack result
            if isinstance(result, tuple) and len(result) >= 3:
                prompt_ids, completion_ids, logprobs = result[0], result[1], result[2]
            else:
                raise ValueError(f"Unexpected result format from _generate_single_turn: {type(result)}")

            # Store results in original query order
            for i, query_idx in enumerate(query_indices):
                all_prompt_ids[query_idx] = prompt_ids[i]
                all_completion_ids[query_idx] = completion_ids[i]
                all_logprobs[query_idx] = logprobs[i] if logprobs is not None else None

            print(f"      ✓ Generated {num_in_group} completions in {elapsed:.1f}s")

        # Restore original params
        trainer.temperature = orig_temp
        trainer.top_p = orig_top_p

        print(f"[DiverseRollout] ✓ Returning {num_queries} completions\n")

        return {
            "prompt_ids": all_prompt_ids,           # num_queries items (e.g., 32)
            "completion_ids": all_completion_ids,   # num_queries items
            "logprobs": all_logprobs,               # num_queries items or Nones
        }

    return rollout_func
