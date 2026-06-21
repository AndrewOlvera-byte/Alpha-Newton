# Flightmare Procedural Course And High-Speed MPC Expert Report

Date: 2026-06-18

## Executive Summary

Adding more gate noise is low difficulty because the code already supports per-gate position/yaw noise, anisotropic XYZ noise in some paths, full gate quaternions, strict aperture validation, and v3 gate-corner observations.

Adding a real procedural course distribution is moderate difficulty. The main missing piece is not math; it is making course generation a first-class shared module instead of a set of ad hoc branches inside `scripts/flightmare_bc/collect.py`. Collection, controller eval, BC rollout eval, policy eval, and PPO reset/curriculum should all sample from the same scenario distribution and log the same metadata.

Adding a high-speed realistic expert is the harder part. The current geometric/min-jerk controller is now a reliable bootstrap expert only in the validated `1.8-2.6 m/s` envelope. For high-speed racing labels that put PPO close to its target distribution, the better next expert is an MPC/MPCC style planner/controller that optimizes progress under gate/corridor and actuator constraints, then emits CTBR or motor commands. That should be introduced as a new expert backend, not as another gain tweak to the min-jerk tracker.

## Current Support Already In The Repo

### Gate/course generation

Relevant files:

- `scripts/flightmare_bc/collect.py`
  - `sample_gate_course(...)` supports `swift_like`, `swift_v4`, `fixed_gates`, and procedural `gates`.
  - `_swift_v4_gate_course(...)` already creates the canonical Split-S layout with an inverted gate.
  - `_gates_from_centers(...)` already applies fixed gate position/yaw noise and supports `fixed_gate_pos_noise_xyz`.
  - `_layout_gate_course(...)` loads JSON gate layouts with optional `quat`.
  - `waypoints_from_gates(...)` already honors full gate quaternions through `_gate_forward(...)`.
  - `_gate_record(...)` persists gate `quat` into `index.json`.
  - `episode_gate_quality(...)` re-validates recorded trajectories with strict apertures.

- `scripts/flightmare_bc/expert_env.py`
  - `GateSpec` has `pos`, `yaw`, `size`, and optional full quaternion `quat`.
  - `add_gates(...)` uses `addGateQuat` when available, so inverted/tilted gates can be sent to Flightmare/Unity.

- `src/robotics/flightmare_autonomy_fsw/gates.py`
  - `gate_frame(...)`, `signed_gate_distance(...)`, `gate_offsets(...)`, and `StrictGateTracker` are quaternion-aware.
  - This is the right validation layer for inverted gates and arbitrary gate orientations.

### Observation support

Relevant files:

- `scripts/flightmare_bc/obs_v3.py`
  - v3 encodes gate corners in body frame.
  - This can represent full 3D orientation, including inverted gates and tilted gates.

- `scripts/flightmare_bc/transform_to_v3.py`
  - Converts recorded `index.json` gate quaternions into v3 gate-corner observations.

- `scripts/flightmare_bc/mission.py`
  - v2 mission encodes gate position and forward vector, but not the full up/lateral frame.
  - For arbitrary tilted/inverted gate research, use v3. v2 is weaker and should not be the main path.

### RL support

Relevant files:

- `src/robotics/flightmare_envs.py`
  - PPO env samples courses through `sample_gate_course(...)`.
  - v3 observations are available in `_make_obs(...)`.
  - reset modes use `gate_frame(...)`, so reference starts around inverted gates are mostly compatible.

- `src/robotics/flightmare_autonomy_fsw/nodes.py`
  - `CourseConfig` carries the course parameters used by the FSW/eval graph.
  - `FlightmareStateNode` samples course and resets Flightmare.

- `src/robotics/flightmare_ppo_trainer.py`
  - `_env_kwargs(...)` already forwards several course knobs, including `fixed_gate_pos_noise_xyz` and `inverted_roll_jitter_rad`.

- `src/robotics/flightmare_policy_eval.py`
  - BC rollout eval builds a `CourseConfig`, but currently does not forward every course knob that PPO supports.

### Tests already near this surface

Relevant files:

- `tests/test_flightmare_obs_v3_geometry.py`
  - Verifies inverted gate quaternions persist and v3 observations use the quaternion-aware encoder.

- `tests/test_flightmare_bc_collection.py`
  - Verifies the collection config and strict controller eval.

- `tests/test_flightmare_integration.py`
  - Covers PPO reset/curriculum plumbing and some inverted-gate config propagation.

## Gaps To Close For Rich Procedural Courses

### 1. Course generation is not a shared abstraction yet

Right now, course generation lives mostly in `scripts/flightmare_bc/collect.py`. This works, but it makes new scenario families awkward because all users call into it indirectly:

- BC collection
- synthetic recovery collection
- DAgger collection
- controller preflight eval
- BC rollout eval
- PPO env reset
- FSW graph eval

Recommended change:

- Create a shared course module, for example `scripts/flightmare_bc/course_generation.py` or `src/robotics/flightmare_courses.py`.
- Move `GateSpec` or a compatible course dataclass to a neutral location, or keep `GateSpec` in `expert_env.py` but make all course utilities import it from one place.
- Add explicit dataclasses:
  - `CourseDistributionConfig`
  - `CourseScenarioConfig`
  - `GeneratedCourse`
  - `GateRole` or metadata fields for `entry`, `chicane`, `split_s_top`, `inverted`, `finish`, etc.

Difficulty: moderate. The code paths already agree on `GateSpec`; this is mostly cleanup and tests.

### 2. Config surface is incomplete for collection

Some knobs exist in lower-level code or PPO but are not first-class in `configs/flightmare_bc/ctbr_v6.yaml` / `scripts/flightmare_bc/collection_config.py`.

Examples:

- `fixed_gate_pos_noise_xyz`
- `inverted_gate_index`
- `inverted_roll_jitter_rad`
- future `inverted_gate_probability`
- future `gate_size_range`
- future `course_distribution`
- future `scenario_mix`

Recommended change:

- Extend `collection_config.py` with a `course_distribution` section.
- Preserve the simple current `course` block for v6 compatibility.
- Log the fully resolved scenario distribution into `index.json`.
- Add a schema validator that rejects impossible or ambiguous combinations.

Difficulty: low to moderate.

### 3. Synthetic recovery currently needs full gate-frame cleanup

`collect_synthetic_recovery_episode(...)` in `scripts/flightmare_bc/collect_dataset.py` has a local `_gate_frame(...)` helper. It uses the gate forward vector but reconstructs lateral/up from world up. That is good enough for upright gates, but not correct for arbitrary rolled/inverted gates.

Recommended change:

- Replace that local helper with `src.robotics.flightmare_autonomy_fsw.gates.gate_frame(...)`.
- This makes recovery perturbations respect the real gate lateral/up axes for inverted and tilted gates.

Difficulty: low.

### 4. Policy eval needs the same course config as PPO

`src/robotics/flightmare_policy_eval.py` builds a `CourseConfig` but does not appear to forward all course fields that PPO supports, such as `fixed_gate_pos_noise_xyz` and `inverted_roll_jitter_rad`.

Recommended change:

- Make `flightmare_policy_eval.py` use a shared `course_config_from_dict_or_manifest(...)`.
- Use the same helper in:
  - `collect_dataset.py`
  - `eval_controller.py`
  - `flightmare_policy_eval.py`
  - `flightmare_envs.py`
  - `flightmare_ppo_trainer.py`

Difficulty: low to moderate.

## Proposed Procedural Course Distribution

The goal is not just noisy copies of one track. The dataset should include structured racing scenarios that force generalizable geometry and recovery.

Suggested scenario families:

1. `swift_v4_canonical`
   - Current Split-S track.
   - Low noise.
   - Useful anchor for comparisons and continuity with existing runs.

2. `swift_v4_noisy`
   - Same topology.
   - Per-gate XYZ noise, yaw noise, small gate-size variation.
   - Inverted gate roll jitter.

3. `procedural_flow`
   - Forward-progressing gates with bounded heading changes.
   - Controlled spacing, z bounds, yaw aligned to path tangent.
   - No inverted gates by default.

4. `procedural_chicane`
   - Alternating lateral offsets and heading changes.
   - Trains high lateral acceleration and path re-acquisition.

5. `procedural_split_s`
   - Template includes climb, inverted/rolled top gate, descent, and recovery gate.
   - Vary entry direction, height, spacing, and roll jitter.

6. `mixed_inverted`
   - Not every course is Split-S.
   - Sample 0, 1, or 2 inverted/rolled gates with constraints that keep the track physically flyable.

7. `fixed_layout_bank`
   - JSON layouts for known benchmark courses.
   - Useful for held-out evaluation and paper plots.

Minimum metadata to log per episode:

- `scenario_name`
- `scenario_family`
- `scenario_seed`
- `difficulty`
- `path_length_m`
- `num_inverted_gates`
- `gate_roles`
- `noise_params`
- `course_bounds`
- `generator_version`

This metadata matters later for filtering, curriculum staging, held-out evals, and debugging failure modes.

## Geometry Validity Rules

Every generated course should pass a geometry validator before collection or RL reset.

Recommended checks:

- Minimum pairwise gate-center distance.
- Minimum consecutive spacing.
- Maximum consecutive spacing.
- Minimum gate altitude.
- Maximum climb/descent angle between consecutive gates.
- Maximum heading change between adjacent path segments.
- Gate aperture large enough after vehicle radius.
- No duplicate or near-duplicate gates.
- No gate normal that creates an impossible approach from previous gate.
- For inverted/tilted gates, pre/center/exit waypoints must cross the correct side of the gate plane.

This validator should be shared by collection, controller eval, and PPO env creation. Bad course samples should be rejected before labels are written or an RL episode starts.

## BC Collection Changes

Files involved:

- `configs/flightmare_bc/ctbr_v6.yaml`
- `scripts/flightmare_bc/collection_config.py`
- `scripts/flightmare_bc/collect_dataset.py`
- `scripts/flightmare_bc/collect.py` or new `course_generation.py`
- `scripts/flightmare_bc/eval_controller.py`
- `scripts/flightmare_bc/transform_to_v3.py`
- `scripts/flightmare_bc/recompute_v3_norm_stats.py`
- `tests/test_flightmare_bc_collection.py`
- `tests/test_flightmare_obs_v3_geometry.py`

Required work:

1. Add scenario mixture config.
2. Sample a scenario per episode.
3. Generate a course from that scenario.
4. Validate the course geometry.
5. Collect expert, synthetic recovery, or DAgger samples.
6. Store scenario metadata in episode manifest.
7. Preflight controller eval per scenario family, not just one aggregate course.
8. Compute dataset summaries by scenario and speed bin.

Important: do not train on a scenario distribution unless the expert preflight passes per scenario family. Aggregate success can hide one bad family.

## Future RL Support

Files involved:

- `src/robotics/flightmare_envs.py`
- `src/robotics/flightmare_ppo_trainer.py`
- `src/robotics/flightmare_policy_eval.py`
- `src/robotics/curriculum.py`
- `src/robotics/flightmare_autonomy_fsw/nodes.py`
- `src/robotics/flightmare_autonomy_fsw/graph.py`
- PPO configs under `configs/exp/flightmare/paper/...`
- `tests/test_flightmare_integration.py`
- `tests/test_flightmare_ppo_setup.py`

Required work:

1. Make PPO env sample the same `course_distribution` as collection.
2. Allow curriculum stages to override scenario weights and difficulty ranges.
3. Add held-out scenario evals:
   - canonical Split-S
   - noisy Split-S
   - procedural upright
   - procedural inverted
   - held-out fixed layouts
4. Log success, gate completion, speed, gate misses, and termination by scenario.
5. Keep v3 as the default observation schema for arbitrary gate orientation.

Recommended curriculum:

- Stage A: canonical/noisy upright courses at low speed.
- Stage B: noisy Split-S with low speed and course-start resets.
- Stage C: random-start-gate, recovery resets, mixed inverted gates.
- Stage D: high-speed expert-matched distribution.
- Stage E: held-out procedural/fixed-layout validation.

## High-Speed Expert: Why MPC/MPCC Is The Right Next Backend

The current geometric/min-jerk expert is useful for bootstrap data, but it is a trajectory tracker. It does not directly optimize racing progress, actuator constraints, or gate/corridor safety at the limit. In current testing, it is reliable only in a conservative speed envelope.

For high-speed labels, use an MPC/MPCC-style expert:

- MPCC optimizes progress along a path while controlling contouring/lag error.
- Corridor-based MPCC adds hard safety corridors.
- MPCC++ adds stronger safety constraints, learned residual dynamics, and automatic tuning.
- Neural MPC can incorporate learned dynamics residuals while remaining in a real-time MPC loop.
- MPPI is also worth considering if the gate-progress objective becomes hard to express smoothly.

Relevant references:

- CMPCC, corridor-based MPCC for aggressive drone flight: https://arxiv.org/abs/2007.03271
- MPCC++ for time-optimal flight with safety constraints: https://arxiv.org/abs/2403.17551
- Real-time Neural MPC for quadrotors and agile platforms: https://arxiv.org/abs/2203.07747
- Autonomous Drone Racing with Deep Reinforcement Learning: https://arxiv.org/abs/2103.08624
- Reference-free racing MPPI direction: https://arxiv.org/abs/2509.14726
- Iterative learning MPC for drone racing: https://arxiv.org/abs/2508.01103

## Proposed MPC/MPCC FSW Stack

Do not replace the existing policy stack. Add a parallel expert stack that can produce dataset labels and also serve as a privileged evaluation/planning baseline.

Suggested node structure:

1. `CourseNode`
   - Owns generated course, gate frames, scenario metadata, and corridor constraints.

2. `StateNode`
   - Existing `FlightmareStateNode`.
   - Publishes pose, velocity, attitude, body rates.

3. `EstimatorNode`
   - For simulation this can be identity.
   - Later can model latency/noise if real-world transfer matters.

4. `MPCPlannerNode` or `MPCCPlannerNode`
   - Input: state, course, current gate/progress estimate.
   - Output: CTBR, motor, or a short horizon of reference states and controls.
   - Logs solve status, objective, constraint violation, and predicted horizon.

5. `SafetySupervisorNode`
   - Rejects labels if solver fails, constraints are violated, or action saturation is excessive.
   - Optional fallback to geometric controller for low-speed recovery labels, but labels should be tagged by source.

6. `ControllerNode`
   - If MPC emits CTBR, use the existing `BaseCTBRController`.
   - If MPC emits motor thrusts, use `BaseMotorController`.
   - If MPC emits full-state references, use a tracking controller only as a low-level actuator adapter.

7. `DatasetWriter`
   - Logs state, v3 obs, action, prev action, reference horizon, gate index, scenario metadata, and expert diagnostics.

## MPC/MPCC Backend Options

Best practical path:

1. Start with an offline or batched nonlinear trajectory optimizer to generate clean high-speed trajectories for a course.
2. Add a receding-horizon MPCC/MPPI controller that can recover from perturbed states.
3. Use DAgger-style correction by querying MPC/MPCC from the BC policy's actual states.

Likely implementation options:

- `acados` or `CasADi` based nonlinear MPC.
- MPPI in Python/JAX/PyTorch for faster iteration and nonsmooth gate-progress objectives.
- External C++/Python MPCC backend invoked through an adapter.

The repo already has an explicit disabled `mpcc` config slot in `configs/flightmare_bc/ctbr_v6.yaml`. That is the right extension point. The adapter should be real and test-gated before it is allowed to collect high-speed labels.

## MPC Label Schema

For future-proofing, each MPC-labeled sample should store more than just the chosen action.

Recommended HDF5 additions:

- `expert/source`: `geometric_minjerk`, `mpcc`, `mppi`, `fallback`, etc.
- `expert/solve_status`
- `expert/objective`
- `expert/constraint_violation`
- `expert/progress`
- `expert/predicted_states`: optional horizon tensor
- `expert/predicted_actions`: optional horizon tensor
- `expert/current_gate`
- `expert/scenario_name`

Recommended manifest additions:

- per-episode scenario metadata
- expert backend version
- MPC config hash/path
- success/failure counts by solver status
- strict gate metrics
- speed-bin metrics

## Test Plan

### Unit tests

Files:

- `tests/test_flightmare_obs_v3_geometry.py`
- `tests/test_flightmare_bc_collection.py`
- new `tests/test_flightmare_course_generation.py`

Tests:

- Deterministic generation for fixed seed.
- Scenario weights produce expected families.
- Generated gates satisfy geometry constraints.
- Inverted/tilted gates have correct quaternion frames.
- `waypoints_from_gates(...)` crosses the correct gate plane.
- v3 encoder changes when gate roll changes.
- Recovery perturbations use full gate frame, not world-up approximation.
- Manifest records scenario metadata.

### Controller/expert eval tests

Files:

- `scripts/flightmare_bc/eval_controller.py`
- future `scripts/flightmare_bc/eval_expert_backend.py`
- `tests/test_flightmare_bc_collection.py`

Tests:

- Geometric expert passes only its configured safe envelope.
- MPCC expert passes high-speed envelope before collection.
- Eval reports per-scenario and per-speed-bin results.
- Collection refuses labels if any required scenario family fails.

### RL/eval tests

Files:

- `src/robotics/flightmare_envs.py`
- `src/robotics/flightmare_policy_eval.py`
- `tests/test_flightmare_integration.py`

Tests:

- PPO env samples scenario distributions.
- Curriculum stage overrides scenario weights.
- Policy eval uses same course distribution as dataset/eval config.
- Held-out scenario eval works with v3 observations.

## Difficulty Estimate

1. More gate position/yaw noise in collection
   - Difficulty: low.
   - Estimate: 0.5 to 1 day with tests.
   - Most plumbing already exists.

2. Generalized inverted/tilted gate support in collection/recovery/eval
   - Difficulty: low to moderate.
   - Estimate: 1 to 2 days.
   - Main fix is using full `gate_frame(...)` everywhere and exposing config.

3. First-class procedural course distribution shared across BC and RL
   - Difficulty: moderate.
   - Estimate: 2 to 5 days depending on validator depth and metadata.
   - Worth doing before generating a large v7 dataset.

4. Scenario-aware controller preflight and dataset analysis
   - Difficulty: moderate.
   - Estimate: 1 to 2 days.
   - This is mandatory for avoiding bad labels.

5. High-speed MPC/MPCC expert backend
   - Difficulty: high.
   - Estimate: 1 to 3 weeks for a practical first backend if using an existing solver stack.
   - More if implementing and tuning from scratch.
   - This is the right path for high-speed data, but it should be isolated behind an expert adapter and strict preflight gates.

## Recommended Next Implementation Order

1. Extract shared course generation and add scenario metadata.
2. Expose course noise and inverted-gate knobs in `collection_config.py`.
3. Fix recovery sampling to use quaternion-aware `gate_frame(...)`.
4. Add scenario-mixture configs and generator tests.
5. Extend `eval_controller.py` to report per-scenario results.
6. Regenerate a v7 bootstrap dataset with geometric expert only in its safe speed envelope.
7. Add MPCC/MPPI expert adapter behind the existing `mpcc` config slot.
8. Gate MPCC collection with strict Flightmare eval and solver diagnostics.
9. Generate high-speed expert + recovery + DAgger data.
10. Use that as PPO initialization, with PPO speed targets inside the demonstrated high-speed data distribution.

## Bottom Line

The procedural/noisy/inverted course idea is very feasible and fits the current codebase well, especially with v3 observations. The highest leverage cleanup is to make course generation a shared, scenario-aware module and make every eval path use it.

For high-speed racing, do not stretch the geometric/min-jerk controller. Keep it as a conservative bootstrap/recovery source and add a real MPC/MPCC expert backend for fast labels. That gives BC a policy already living near the racing distribution, which is exactly what PPO needs if the goal is speed fine-tuning rather than rediscovering high-speed flight from sparse gate rewards.
