# Flightmare Teacher Policy Deep-Dive Handoff

Last updated: 2026-06-17 22:21 EDT.

This report is the compact context a new session needs to continue the
Flightmare CTBR teacher-policy debugging work.

## Goal

Build a robust privileged-state drone-racing teacher policy for the
Alpha-Newton Flightmare environment. The actor is an MLP/Swift-style fusion
policy trained with PPO from a BC checkpoint. The immediate problem is that
recent configs with the more capable racing plant have struggled to move from
one-gate behavior into reliable multi-gate transitions.

The current best hypothesis is not "insufficient gate-index coverage." It is
that the reset/state distribution does not expose the policy to realistic
post-gate exit states. Synthetic reference starts train gate approach, but do
not teach the true exit-state-to-next-gate distribution needed for multi-gate
flight.

## Key Commands

Always run project commands inside the dependency container:

```bash
docker compose exec alpha-newton <command>
```

Current log benchmark:

```bash
docker compose exec alpha-newton python scratch/analysis/ppo_log_stage_benchmark.py \
  --label v11 \
  wandb/run-20260617_215048-a2kiswg7/files/output.log
```

Recommended next training run:

```bash
docker compose exec alpha-newton python -m src.entrypoints.train_ppo \
  --exp flightmare/paper/teacher/ctbr_stage23_v11_ppo
```

Local policy/env diagnostics across curriculum stages:

```bash
docker compose exec alpha-newton python -m scripts.flightmare_bc.diag_policy_stages \
  --exp flightmare/paper/teacher/ctbr_stage23_v11_ppo \
  --checkpoint outputs/paper_teacher_ctbr_stage12_v11_ppo/checkpoint-2250/model.pt \
  --stage s2_two_gate_windows_finish \
  --stage s3_three_gate_windows \
  --episodes 16 \
  --deterministic
```

Autonomy/eval entrypoint:

```bash
docker compose exec alpha-newton python -m src.entrypoints.eval_flightmare \
  --exp flightmare/paper/teacher/ctbr_stage23_v11_ppo \
  --checkpoint outputs/paper_teacher_ctbr_stage23_v11_ppo/final/model.pt \
  --episodes 50 \
  --backend flightgym \
  --fail-on-fallback \
  --no-plot
```

Relevant tests:

```bash
docker compose exec alpha-newton python -m pytest \
  tests/test_flightmare_rewards.py \
  tests/test_flightmare_obs_v3_geometry.py \
  tests/test_flightmare_integration.py
```

Most recent integration test result observed during this work:
`tests/test_flightmare_integration.py`: 12 passed.

## Important Files

Configs:

- `configs/exp/flightmare/paper/teacher/ctbr_stage12_v10_ppo.yaml`
- `configs/exp/flightmare/paper/teacher/ctbr_stage12_v11_ppo.yaml`
- `configs/exp/flightmare/paper/teacher/ctbr_stage23_v11_ppo.yaml`
- Older reference: `configs/exp/flightmare/paper/ctbr_stage23_ppo.yaml`
- Old simple curriculum lineage mentioned by the user:
  - `configs/exp/flightmare/paper/ctbr_bc.yaml`
  - `configs/exp/flightmare/paper/ctbr_stage12_ppo.yaml`
  - `configs/exp/flightmare/paper/ctbr_stage23_ppo.yaml`

Core training/eval:

- `src/entrypoints/train_ppo.py`
- `src/entrypoints/eval_flightmare.py`
- `src/robotics/flightmare_ppo_trainer.py`
- `src/robotics/flightmare_envs.py`
- `src/robotics/curriculum.py`
- `src/robotics/rewards.py`

Diagnostics and analysis:

- `scratch/analysis/ppo_log_stage_benchmark.py`
- `scripts/flightmare_bc/diag_policy_stages.py`
- `scripts/flightmare_bc/diag_obs_consistency.py`
- `scratch/analysis/eval_ppo_stage_checkpoint.py`
- `scratch/analysis/eval_v9_ckpt_sweep/`

Tests added/extended:

- `tests/test_flightmare_integration.py`

## Code Changes Already Made

Important implemented support:

- `eval_flightmare.py` now applies `robotics.ppo.plant` overrides and reports
  plant details in eval stats.
- `flightmare_ppo_trainer.py` now:
  - applies curriculum in smoke tests;
  - logs `mean_goal_completion`;
  - logs `start_gate_counts`;
  - supports stage-local `actor_lr_override`;
  - supports `reset_best_on_resume`.
- `curriculum.py` now supports per-stage advancement:
  - `advance_metric`;
  - `advance_threshold`;
  - `advance_after_iter`;
  - `advance_patience`.
- `flightmare_envs.py` now supports:
  - post-pass goal modes such as `pass_gate_i_survive` and
    `pass_gate_i_align_next`;
  - segment progress relative to `segment_start_pos`;
  - reference-start jitter:
    `start_lateral_range_m`, `start_vertical_range_m`,
    `start_yaw_noise_rad`, `start_speed_noise_mps`.
- `diag_policy_stages.py` and `diag_obs_consistency.py` now preserve plant,
  action-bound, post-pass, and start-jitter env keys and filter trainer-only
  curriculum keys.

## Current Run State

As of the latest check, no `train_ppo` process is running.

Completed/live-relevant run:

- WandB log: `wandb/run-20260617_215048-a2kiswg7/files/output.log`
- Output dir: `outputs/paper_teacher_ctbr_stage12_v11_ppo`
- Last checkpoint present: `checkpoint-2500`
- The run reached the end of v11 stage 2 and is not active.

Latest v11 summary:

```text
stage 1: s1_one_gate_all
  iters: 1-332
  best success: 96.9% at iter 332
  final success: 96.9%
  start gates: 0,1,2,3,4,5,6

stage 2: s2_two_gate_windows
  iters: 333-2500
  best success: 37.6% at iter 2260
  best goal completion: 66.5% at iter 2484
  tail100 success: 21.3%
  final success: 23.6%
  final goal completion: 60.2%
  final gate completion: 56.0%
  final pass/miss/crash: 148 / 94 / 0
  final action clip fraction: 0.01
  start gates: 0,1,2,3,4,5
```

Interpretation:

- v11 is learning two-gate transitions.
- It is slow, but not collapsing.
- Crashes are nearly gone.
- Action clipping is nearly zero.
- The remaining problem is mostly second-gate misses.
- The fixed transition at iter 2500 is too early.

The next config is:

- `configs/exp/flightmare/paper/teacher/ctbr_stage23_v11_ppo.yaml`
- It resumes from:
  - `outputs/paper_teacher_ctbr_stage12_v11_ppo/checkpoint-2250/model.pt`
  - `outputs/paper_teacher_ctbr_stage12_v11_ppo/checkpoint-2250`
- It uses `reset_best_on_resume: true` so the one-gate best from v11 does not
  dominate checkpoint bookkeeping.

Stage23 v11 schedule:

```text
s2_two_gate_windows_finish:
  until_iter: 4300
  goal_mode: gate_i_to_i+1
  advance only after iter 3900 if success >= 0.68 for 16 rollouts

s3_three_gate_windows:
  until_iter: 7600
  goal_mode: gate_i_to_i+2
  advance only after iter 6500 if success >= 0.55 for 16 rollouts
```

The actor LR was raised from v11's `1.6e-6` to `2.2e-6` because KL was far
below target and clipping was near zero. This is meant to reduce the slow
learning without changing reward or dynamics.

## Experiment Findings

### v9

v9 produced useful local gate-window behavior but failed full-course eval from
course start. The key failure was distribution mismatch:

- PPO-local diagnostics showed some checkpoints could pass sampled gates under
  stage-local environments.
- `eval_flightmare.py` full-course evaluations failed before or around gate 0.
- Later v9 checkpoints tended to miss gate 0 rather than simply crash.
- The policy was not trained on the same course-start state distribution used
  by full-course eval.

### v10

v10 was the complex "broad early replay plus align-next/post-pass" curriculum.
It finished at iter 9000 and failed as a teacher.

Output dir:

- `outputs/paper_teacher_ctbr_stage12_v10_ppo`

Log:

- `wandb/run-20260617_030841-jf0a671n/files/output.log`

Important v10 facts:

- The saved `best` checkpoint is iter 368, inside stage 1, not a full-course
  teacher.
- Stage 1 solved early:
  - best success 99.7% at iter 333/368 depending on metric read;
  - final stage-1 success at fixed transition was only 72.8%.
- It degraded because it kept optimizing an already-solved short task until
  fixed transition.
- Alignment/post-pass stages partially worked, then degraded as constraints
  tightened.
- Chain stages never recovered.
- Course-start stages fully collapsed.

Stage-level v10 summary:

```text
s1 one-gate replay:
  best 99.7%, final 72.8%
s2 loose align:
  best 62.4%, final 45.1%
s3 stabilize:
  best 71.4%, final 58.4%
s4 exit survival:
  best 71.0%, final 35.0%
s5 moderate tighten:
  best 56.1%, final 26.3%
s6 final align hold:
  best 54.7%, final 4.6%
s7 chain warmup:
  final 0.0%
s8 chain partial:
  final 6.5%
s9 chain controlled:
  final 9.3%
s10 raise speed:
  final 2.2%
s11 course-start two-gate:
  final 0.0%
s12 course-start three-gate:
  final 0.0%
s13 course-start full:
  final 0.0%
```

Final v10 at iter 9000:

```text
success: 0.0%
goal: 0.1%
gate completion: 0.1%
pass/miss/crash: 1 / 91 / 32
action clip: 0.22
```

Conclusion:

v10 trained "pass gate then satisfy align/survive constraints" from synthetic
reference starts. That did not become robust exit-state-to-next-gate behavior.
The extra post-pass success definition was a worse target than simply extending
the number of gates to complete.

### v11

v11 intentionally returned to a simple gate-count curriculum:

```text
1 gate -> 2 gates -> 3 gates -> 4-gate windows -> full course
```

It removed `pass_gate_i_align_next` as the main target and kept dense
trajectory/dynamics rewards. This is closer to the original successful config
family and closer to Swift-style dense progress training.

v11 is promising:

- Stage 1 solved cleanly and metric-gated at iter 332.
- Stage 2 is improving, with very low clipping and nearly zero crashes.
- It needs a stage23 continuation because the fixed two-gate window is too
  short.

## Dynamics / Reward / Controller Findings

The current plant is more capable than the earlier slow plant. Configs include
the plant under `robotics.ppo.plant`, for example:

```yaml
mass: 0.73
arm_length: 0.17
inertia: [0.0025, 0.0021, 0.0043]
k_thrust: 1.91e-6
k_torque: 2.6e-7
motor_omega_min: 150.0
motor_omega_max: 3000.0
motor_tau: 0.0001
thrust_map: [1.3298253500372892e-06, 0.0038360810526746033, -1.7689986848125325]
kappa: 0.016
omega_max_body: [18.0, 18.0, 18.0]
```

Potential dynamics/control concerns:

- Action bounds and plant overrides must be used consistently in training,
  local diagnostics, and eval. This has been patched and tested.
- v10 action clipping rose substantially during failures; v11 has almost none.
- v11 speed is high enough to pass one gate and approach the next, but it
  misses the second gate. That points more to trajectory/state distribution
  than motor authority.

Reward concerns:

- Large terminal `completion_bonus` values can create big value discontinuities
  when changing goal modes.
- v10 showed large value loss around abrupt objective changes.
- Do not add more hand-authored success constraints before verifying simple
  gate-count scaling.

## Reset Distribution Finding

Start-gate variety is not the missing piece by itself.

Evidence:

- v10 and v11 stage 1 both covered all start gates immediately.
- v11 stage 2 covers start gates 0-5.
- v11 still struggles with two-gate success despite broad gate-index coverage.

Likely missing Swift-style piece:

- Reset from states sampled around previous successful trajectories/gate passes.
- Synthetic reference starts use clean pre-gate poses, fixed-ish speed, and
  limited angular/attitude diversity.
- Multi-gate flight depends on the actual exit state after the previous gate:
  position, velocity vector, attitude, body rates, and previous action.

Recommended future engineering task:

Add a trajectory-state reset buffer:

- During PPO rollouts, store states around successful gate passes:
  `pos`, `quat`, `vel`, `omega`, previous action, gate index, next gate index.
- Add a reset mode such as `trajectory_replay_or_reference`.
- Sample from per-gate buffers with bounded perturbations.
- Fall back to `reference_state` until the buffer has enough samples.
- Initial implementation can be per-env local to avoid vector-worker shared
  state complexity.
- Longer-term implementation can aggregate pass states in trainer `info`s and
  broadcast to workers.

## Swift / Champion-Level Policy Research Takeaways

Primary sources:

- Song et al. 2021, "Autonomous Drone Racing with Deep Reinforcement Learning":
  https://ar5iv.labs.arxiv.org/html/2103.08624
- Kaufmann et al. 2023, "Champion-level drone racing using deep reinforcement
  learning" / Swift:
  https://www.nature.com/articles/s41586-023-06419-4
- RPG "Bootstrapping Reinforcement Learning with Imitation for Vision-based
  Agile Flight":
  https://rpg.ifi.uzh.ch/bootstrap-rl-with-il/index.html

Important takeaways:

- The core recipe is simple and dense:
  - relative gate observations;
  - path/segment progress reward;
  - safety or gate-margin terms;
  - body-rate/control penalties;
  - high parallel sampling;
  - staged or automatic curriculum.
- Song et al. specifically emphasize distributed initialization:
  - early training initializes around path segments so rollouts cover the whole
    track;
  - once gate passing is reliable, initial states are sampled from previous
    trajectories.
- Swift uses a feedforward neural control policy trained with model-free
  on-policy RL in simulation.
- Swift's action interface is collective thrust plus body rates, then a
  low-level controller maps to motor commands.
- Swift/champion-level transfer depends heavily on accurate dynamics,
  controller/actuator modelling, and empirical residual/noise models from real
  data.
- BC/IL is not the main missing teacher-training ingredient here. In the
  teacher-policy setting, BC is mainly an initialization or later
  distillation/student mechanism. The teacher itself needs the right on-policy
  state distribution.

## Recommended Next Steps

1. Run `ctbr_stage23_v11_ppo`.
2. Monitor with `ppo_log_stage_benchmark.py`.
3. Do not advance from two-gate until success is at least roughly stable in the
   60-70% range, not just a one-rollout spike.
4. After three-gate windows, add a separate stage34 continuation rather than
   jumping directly to full course.
5. If stage23 still learns slowly or plateaus, implement trajectory-state replay
   resets before adding more reward shaping.

Suggested stage schedule after stage23:

```text
stage 4:
  four-gate windows
  2500-3000 iterations

course bridge:
  course_start two/three-gate bridge
  1000-1500 iterations

full course:
  finish_remaining_course
  3000+ iterations
```

## What Not To Do Next

- Do not judge progress only by final full-course eval from a checkpoint that
  has never trained course-start.
- Do not reintroduce the v10 `align_next`/post-pass curriculum as the main
  route unless simple gate-count stages fail after enough runtime.
- Do not compare `best/model.pt` blindly across runs. v10 and v11 can save
  one-gate best checkpoints that are not valid full-course teachers.
- Do not run another Flightmare trainer concurrently if the existing trainer is
  active, because worker ports can conflict.

