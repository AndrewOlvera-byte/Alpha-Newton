# Alpha-Newton

Alpha-Newton is a focused research codebase for a two-phase study of group-based reinforcement learning with verifiable rewards on mathematical reasoning. The study asks two related questions:

1. How do GRPO and DAPO behave as the number of sampled completions per prompt changes?
2. Once that training regime is fixed, how does reward design change the learning signal?

This repository contains the training, reward, configuration, and checkpoint-selection code for those ablations. Model checkpoints, generated evaluation sets, run logs, plots, and result claims are intentionally not part of the public tree.

## Research design

All experiments start from the same Qwen3-0.6B reasoning checkpoint and train on `allenai/RLVR-GSM`. TRL supplies the GRPO trainer and vLLM supplies colocated generation. A deterministic held-out GSM set ranks checkpoints during training.

### Phase 1: optimization and group size

Phase 1 separates the effect of the policy objective from the number of completions sampled for each prompt.

| Factor | Values represented in the configs |
| --- | --- |
| Objective | GRPO, DAPO |
| Group size | 8, 16, 32 generations |
| KL reference penalty | `beta=0.03`, plus DAPO no-KL variants |
| Horizon check | DAPO K=32 extended from 780 to 1,200 steps |

The configs live directly under [`configs/exp/EA_RLVR_ABLATIONS`](configs/exp/EA_RLVR_ABLATIONS). Each file contains only the experimental override; shared model, data, generation, and optimization settings are inherited from `_study.yaml` and the base configs.

### Phase 2: reward shaping

Phase 2 fixes DAPO at 32 generations and compares three reward signals:

- `ppo_binary`: correctness-only verification.
- `dapo_rank_stratified`: ranks responses within correctness strata using a quality score.
- `dapo_structure_balanced`: combines correctness with explicit reasoning-structure checks.

Each reward is represented with and without the KL reference penalty. The code also includes longer 2,000-step checks and a DoRA parameter-efficient variant. These are controlled comparisons, not separate training pipelines.

## Code map

```text
configs/
  base/                         shared RLVR and runtime settings
  exp/EA_RLVR_ABLATIONS/       phase-specific experimental overrides
scripts/posttraining/
  create_held_out_eval.py      deterministic held-out set construction
src/
  builders/                    model, tokenizer, dataset, and trainer assembly
  core/                        config inheritance and component registry
  entrypoints/train_rlvr.py    the single training entrypoint
  rlvr/math_verifier.py        answer extraction and the three reward variants
  rlvr/rl_callbacks.py         held-out evaluation and top-K checkpoint retention
tests/                         config and reward behavior checks
```

The execution path is deliberately small:

```text
experiment YAML
    -> merged research configuration
    -> Qwen model + AllenAI RLVR-GSM dataset
    -> TRL GRPOTrainer configured as GRPO or DAPO
    -> verifiable reward function
    -> held-out checkpoint ranking
```

## Reading the configurations

Configuration inheritance keeps the comparison surface visible:

- `configs/base/common.yaml` defines model-loading, output, and tracking defaults.
- `configs/base/rlvr.yaml` defines the common optimizer, generation, vLLM, and checkpoint-selection behavior.
- `_study.yaml` fixes the starting model and training dataset.
- `_phase.yaml` fixes the Phase 2 DAPO/K=32 design.
- Each named experiment overrides only the variable under study.

For example, `p1_dapo_k16_noKL.yaml` changes the objective, group size, and KL coefficient without restating the rest of the training stack. A config name is passed as its path below `configs/exp`, without the `.yaml` suffix:

```bash
python -m src.entrypoints.train_rlvr \
  --exp EA_RLVR_ABLATIONS/p1_dapo_k16_noKL
```

The starting checkpoint path in `_study.yaml` documents the checkpoint used in the research environment. It can be replaced with another local path or Hugging Face model identifier when reusing the code.

## Checkpoint selection

Training periodically evaluates a deterministic held-out set and keeps the top checkpoints by exact-answer accuracy. The callback writes `topk_eval_history.json`, which records every selection score and the best checkpoint. The held-out JSONL is generated locally rather than committed:

```bash
python scripts/posttraining/create_held_out_eval.py \
  --output held_out/gsm_eval_v2.jsonl \
  --n-samples 500
```

This repository stops at the research implementation. Result tables and conclusions should be added only after the final evaluation protocol is fixed.
