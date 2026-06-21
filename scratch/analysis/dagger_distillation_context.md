# DAgger-Style Classical Expert Distillation Context

## Purpose

Open-loop BC is not the right mechanism for producing the initial CTBR teacher.
The current runs show that one-step action loss can become tiny while closed-loop
Flightmare rollouts remain at zero success and near-zero gate completion. The
next plan session should treat the initial teacher as a closed-loop distillation
problem:

1. Run the student policy in the real target rollout stack.
2. Query a classical expert on the states the student actually visits.
3. Train the student against those expert labels.
4. Repeat, selecting checkpoints by rollout metrics rather than offline loss.

The expert should be a classical controller, likely the existing MPPI controller
or geometric SE(3) controller, but the label interface must provide enough
information for the distillation objective. A deterministic raw CTBR action is
the minimum; a confidence or covariance estimate is strongly preferred.

## Current State In The Repo

Relevant files:

- `src/robotics/flightmare_experts/backend.py`
  Defines the current expert protocol: `compute(...) -> dict` with raw CTBR
  command fields.

- `src/robotics/flightmare_experts/geometric.py`
  Thin backend around `GeometricSE3Controller`. Fast and robust in the validated
  low-speed envelope, but myopic and does not naturally expose uncertainty.

- `src/robotics/flightmare_experts/mppi.py`
  Sampling-based receding-horizon CTBR controller. It samples action sequences,
  rolls out a reduced CTBR model, scores trajectories, and returns the
  MPPI-weighted first action. This is the best current candidate for a stronger
  distillation expert because it can also provide a meaningful action
  distribution from weighted samples.

- `scripts/flightmare_bc/collect_dataset.py`
  Already has a partial DAgger path. It loads a CTBR policy, rolls that policy
  in `FlightmareExpertEnv`, labels visited states with the classical controller,
  and stores `prev_policy_ctbr`. This is close to what we need, but it is not yet
  a full iterative distillation pipeline.

- `configs/flightmare_bc/ctbr_v6.yaml`
  Uses a mixture of expert, synthetic recovery, and DAgger data. The current
  DAgger block points at a BC checkpoint and uses the geometric controller as
  labeler.

- `configs/flightmare_bc/ctbr_v7_mpcc.yaml`
  Existing config surface for MPPI/MPCC-style expert collection. Useful as the
  starting point for high-quality expert label generation.

- `src/robotics/models/flightmare/MLPFusionGaussianExpertActor.py`
  Student actor already supports GNLL. For distillation, this should be extended
  or reused so expert labels can be trained with per-sample and per-channel
  uncertainty.

## Why Static Open-Loop BC Failed

The failure mode is not that the model cannot reduce offline loss. It can.
The problem is that offline action imitation is a poor proxy for closed-loop
racing performance:

- CTBR actions are temporally smooth, so `prev_action` is a strong shortcut.
- Bounds-normalized CTBR channels do not have equal scale, so plain averaged
  Huber overweights thrust relative to body rates.
- The supervised dataset contains many states near hand-crafted references, but
  the student fails on its own induced state distribution.
- Offline loss keeps improving after rollout score stops improving or gets
  worse.

Future work should treat offline loss as a diagnostic only. Checkpoint
selection should use closed-loop rollout score, gate completion, gate miss rate,
and saturation metrics.

## Target Algorithm

Use iterative closed-loop DAgger distillation:

1. Initialize a student from the best available BC seed or a small expert-only
   dataset.
2. For each DAgger round:
   - Sample courses from the target procedural distribution.
   - Reset from curriculum-selected states, starting with local one-gate and
     gate-transition starts.
   - Execute the student policy in Flightmare.
   - At each visited state, query the classical expert for the corrective CTBR
     command.
   - Record observation, student action, executed action, expert label,
     expert confidence, gate context, and terminal outcome.
   - Aggregate data with previous rounds plus a small amount of clean expert
     data.
   - Retrain or fine-tune the student.
   - Select by rollout score, not eval loss.
3. Advance curriculum only when the student reaches useful gate completion on
   the current distribution.

This is not RL from scratch. The student receives dense supervised expert
labels at every visited state, but the visited states come from its own
closed-loop behavior.

## Expert Options

### Geometric SE(3) Expert

Pros:

- Fast.
- Deterministic.
- Already validated for lower-speed clean expert collection.
- Simple label: one raw CTBR action.

Cons:

- Myopic. It tracks a reference instantaneously rather than planning through
  gate transitions.
- No natural uncertainty estimate.
- May produce labels that are locally reasonable but not enough for high-speed
  racing.

Recommended use:

- Bootstrap rounds.
- Low-speed local correction.
- Fallback when MPPI fails safety checks.

### MPPI Expert

Pros:

- Receding-horizon planner under CTBR dynamics and action limits.
- Can accelerate, brake, and trade off future gate alignment.
- Its weighted rollout samples can define an expert action distribution, not
  only a single action.
- It already returns `mppi_cost`; this can become part of a confidence signal.

Cons:

- More compute per label.
- Reduced model may differ from Flightmare C++ dynamics.
- Needs safety validation so bad solves are never silently written.

Recommended use:

- Primary expert for high-quality closed-loop distillation.
- Provide both mean action and covariance/confidence for the loss.

### Hybrid Expert

A practical labeler can be:

- MPPI first.
- Geometric fallback if MPPI is unsafe, nonfinite, too saturated, or has poor
  cost margin.
- Optional safety supervisor to reject labels entirely.

The dataset should record which source produced the label.

## Expert Label Contract

The current expert protocol returns a dict with:

```python
{
    "thrust_newton": float,
    "thrust_normalized": float,      # raw [0, 1]
    "body_rates": np.ndarray(3,),    # raw rad/s
    "motor_normalized": np.ndarray(4,),
    "source": str,
}
```

For distillation, extend this to an `ExpertLabel` concept. Minimum fields:

```python
{
    "action_ctbr_raw": np.ndarray(4,),       # [T_norm, wx, wy, wz]
    "action_ctbr_norm": np.ndarray(4,),      # same normalization as student target
    "source": str,                           # "mppi", "geometric_minjerk", "fallback", etc.
    "valid": bool,
    "weight": float,                         # sample weight for training
}
```

Preferred fields:

```python
{
    "action_log_std_norm": np.ndarray(4,),   # diagonal expert uncertainty
    "action_cov_diag_norm": np.ndarray(4,),  # equivalent variance if easier
    "confidence": float,                     # derived from MPPI cost/margin/safety
    "expert_cost": float | None,
    "cost_margin": float | None,
    "saturation_frac": float,
    "safety_status": str,
}
```

Useful context fields:

```python
{
    "student_action_ctbr_raw": np.ndarray(4,),
    "student_action_ctbr_norm": np.ndarray(4,),
    "executed_action_ctbr_raw": np.ndarray(4,),
    "prev_action_ctbr_raw": np.ndarray(4,),
    "prev_action_ctbr_norm": np.ndarray(4,),
    "gate_index": int,
    "course_family": str,
    "reset_mode": str,
    "termination_reason": str | None,
    "distance_to_gate": float,
    "gate_lateral_norm": float,
    "gate_vertical_norm": float,
    "gate_signed_distance": float,
}
```

The training dataset should expose at least:

- `state`
- `prev_actions`
- `action` or `expert_action`
- optional `expert_log_std` or `expert_weight`
- optional metadata used for stratified sampling and diagnostics

## Distillation Objectives

### Option A: Weighted GNLL On Expert Mean

Use student Gaussian `p_theta(a | s) = N(mu_theta, sigma_theta)` and expert mean
action `a_e`.

Loss:

```text
L = w(s) * NLL_student(a_e | s)
```

This can reuse the current GNLL path if the dataset supplies the expert action
as `batch["action"]`. It is the simplest first implementation.

Important settings:

- `bc_loss_type: gnll`
- `state_dependent_std: true`
- normalize CTBR labels exactly like rollout action normalization
- log raw per-channel MAE and saturation metrics

### Option B: KL From Expert Gaussian To Student Gaussian

If MPPI supplies a weighted action distribution over first actions, fit a
diagonal Gaussian expert:

```text
q_e(a | s) = N(mu_e, sigma_e)
```

Then minimize:

```text
L = w(s) * KL(q_e || p_theta)
```

For diagonal Gaussians this gives a stable objective that teaches both mean and
uncertainty. This is the best distillation math if MPPI covariance is available.

For geometric labels, use a fixed covariance by action channel. That fixed
covariance should be calibrated in normalized action space, not raw units.

### Option C: Weighted MSE or Huber

Only use if explicitly channel-weighted. Plain averaged bounds-space Huber is
not acceptable for CTBR.

If used:

```text
L = w(s) * mean_i alpha_i * rho((mu_i - a_e_i) / sigma_i)
```

where `alpha_i` or `sigma_i` corrects channel scale. Otherwise thrust dominates
and body-rate control can be undertrained.

## MPPI Distribution Extraction

The current `MPPIController.compute` samples action sequences and computes
softmax weights over rollout cost. It currently returns only the weighted first
action and `mppi_cost`.

For KL distillation, modify MPPI to expose:

- weighted first-action mean
- weighted first-action diagonal variance
- min cost
- effective sample size
- cost margin between best and typical samples
- saturation rate of sampled first actions

Pseudo-output:

```python
weights = softmax(-(cost - cost.min()) / temperature)
a0_samples = actions[:, 0, :]
mu = sum(weights[:, None] * a0_samples)
var = sum(weights[:, None] * (a0_samples - mu) ** 2)
ess = 1.0 / sum(weights ** 2)
```

Then normalize `mu` and `var` into the same bounds-normalized CTBR space used
by the student before writing the dataset.

## Collection Design

The existing `collect_dagger_episode` should become a round-based collector:

```text
round_000:
  behavior: seed BC / expert mixture
  labeler: geometric or MPPI
  starts: local one-gate and transition starts

round_001:
  behavior: student trained on round_000
  labeler: MPPI preferred, geometric fallback
  starts: same plus harder transitions

round_N:
  behavior: latest selected student
  labeler: MPPI
  starts: full procedural distribution
```

Execution policy:

- Usually execute the student action to collect true induced states.
- Optionally use expert intervention or action mixing early in training:
  `a_exec = beta * a_expert + (1 - beta) * a_student`.
- Record both student and executed action.
- If the student immediately misses gates, keep those states but reset often.
  They are useful correction data.
- Do not require full-episode success to write data. The point is to collect
  corrective labels before and during failure.

Sampling priorities:

- Gate approach.
- Gate plane crossing.
- Post-gate handoff.
- States with high lateral/vertical normalized error.
- States where student and expert disagree.
- States before gate miss.

Avoid letting the dataset become mostly smooth straight-line action copying.

## Curriculum For Distillation

Start with local closed-loop skills:

1. One gate, aligned starts.
2. One gate, lateral/vertical/yaw perturbations.
3. Gate pass plus post-gate stabilization.
4. Two-gate transitions.
5. Mixed procedural flow/chicane.
6. Inverted and split-S scenarios.
7. Full target distribution.

Each stage should advance on closed-loop metrics:

- gate completion
- gate miss rate
- final distance to next gate
- thrust/body-rate saturation
- mean speed only after reliability improves

Offline loss should never advance curriculum by itself.

## Dataset Aggregation Strategy

Keep a balanced replay mixture:

- Clean expert rollouts: small anchor, 10-25 percent.
- Synthetic recovery: limited, mainly for coverage around gates.
- DAgger rounds: majority after round 1.
- Recent DAgger rounds should be oversampled relative to stale rounds.
- Hard failures and gate-transition samples should be oversampled.

Store round id and sample kind in metadata so the loader can stratify.

Suggested split keys:

```text
sample_kind: expert | synthetic_recovery | dagger
dagger_round: int
expert_source: mppi | geometric_minjerk | fallback
course_family: str
stage: str
```

## Prev-Action Handling

The current policy includes `prev_actions`, and the dataset makes previous
action highly predictive. That is useful for actuator continuity, but it can
hide weak state-to-action learning.

For DAgger distillation:

- Always record the executed previous action, not the nominal target previous
  action.
- Add training-time prev-action dropout or noise.
- Track an ablation with `include_prev_action: false`.
- Log `prev_action` baseline loss for every dataset round.

If the student cannot beat a `prev_action` copy baseline by much, offline loss
is not meaningful.

## Safety And Label Rejection

The expert labeler should not silently write bad labels. Reject or downweight
when:

- nonfinite action
- thrust outside [0, 1] before clamp
- body rates near hard limits for sustained periods
- MPPI effective sample size is too low
- MPPI cost is too high
- model rollout state is far outside the expert validity envelope
- gate geometry is invalid or ambiguous

Rejected states can still be logged for diagnostics, but should not train the
student unless labeled by fallback.

## Metrics To Log

Training/offline:

- GNLL or KL
- raw CTBR MAE by channel
- normalized CTBR MAE by channel
- expert uncertainty by channel
- student log_std by channel
- sample weight distribution
- loss by sample kind, DAgger round, course family, gate index
- prev-action copy baseline

Closed-loop:

- success rate
- mean gate completion
- mean gates completed
- gate miss rate
- final distance to next gate
- mean speed
- thrust saturation fraction
- body-rate saturation fraction
- planner action p95
- termination reasons

Checkpoint selection:

- primary: rollout score or gate completion
- tie-break: lower gate miss rate
- second tie-break: lower saturation and final distance
- never select by offline loss when rollout metrics are available

## Required Code Changes

1. Add an `ExpertLabel` dataclass or structured dict.
   - Source: `src/robotics/flightmare_experts/backend.py`
   - Must support deterministic geometric labels and distributional MPPI labels.

2. Extend `MPPIController.compute`.
   - Return weighted first-action variance, effective sample size, and cost
     diagnostics.
   - Keep backwards compatibility with existing command dict fields.

3. Add a round-based DAgger collector.
   - Source candidate: extend `scripts/flightmare_bc/collect_dataset.py`
   - Or create `scripts/flightmare_bc/collect_dagger_distill.py`.
   - Inputs: behavior checkpoint, behavior config, expert config, output dir,
     round id, curriculum stage.

4. Extend `EpisodeWriter`.
   - Store expert label diagnostics and student/executed actions.
   - Preserve existing `action/ctbr` for compatibility, but add explicit
     distillation fields.

5. Extend dataset loader.
   - Expose `expert_log_std`, `sample_weight`, and sample metadata if present.
   - Keep old configs working when fields are absent.

6. Extend actor/trainer loss.
   - First pass: weighted GNLL against expert action.
   - Better pass: diagonal Gaussian KL when expert covariance exists.

7. Add config files.
   - `configs/flightmare_bc/ctbr_dagger_distill.yaml`
   - `configs/exp/flightmare/paper/teacher/ctbr_bc_dagger_distill.yaml`

8. Add diagnostics.
   - Dataset report: label source mix, raw action stats, expert uncertainty,
     prev-action baseline.
   - Rollout report: compare seed, round checkpoints, and expert controller.

## First Concrete Experiment

Use a conservative first round:

- Behavior: latest GNLL v6 BC checkpoint.
- Expert: geometric if MPPI is not ready; MPPI if controller eval passes.
- Course distribution: local one-gate and two-gate transition starts, not full
  courses from the start.
- Execute: student action, with optional expert intervention if state becomes
  unsafe.
- Record: every step near gate approach/crossing and all states preceding miss.
- Train: GNLL with state-dependent std, sample weights enabled.
- Select: rollout gate completion and gate miss rate.

If MPPI is used, additionally train KL to MPPI diagonal action distribution.

## Open Decisions For The Future Plan Session

1. Should the first implementation be weighted GNLL only, or should we implement
   MPPI Gaussian KL immediately?
2. Should behavior execution be pure student, beta-mixed expert/student, or
   safety-shielded student?
3. Which curriculum stage should round 0 start on: one-gate local, two-gate
   transition, or current full v6 distribution?
4. Do we keep `include_prev_action: true` with dropout/noise, or run an ablation
   without previous action?
5. What MPPI confidence thresholds reject labels?
6. Should the dataset store all expert diagnostics in HDF5 now, or start with
   minimal fields and add diagnostics after the first round?

## Bottom Line

The right path is not more open-loop BC and not long-horizon RL from scratch.
It is closed-loop classical-expert distillation. The key technical requirement
is to upgrade the expert label from "one raw CTBR action" to a structured
distillation target with source, confidence, and ideally covariance. MPPI is the
best fit for that because its weighted samples naturally define a label
distribution; geometric SE(3) remains useful as a fast bootstrap and fallback.
