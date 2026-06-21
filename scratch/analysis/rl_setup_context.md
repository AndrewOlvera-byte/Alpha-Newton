# Flightmare Teacher: Distillation Findings → RL Setup Context

**Author:** investigation 2026-06-19/20
**Purpose:** Consolidate everything learned while trying to produce a privileged,
state-only CTBR teacher that reliably chains gates, so the next iteration can be
built as an **RL setup** rather than re-deriving these conclusions. Read this
before designing the new PPO run.

---

## 1. Goal

A privileged **state-only** policy (proprio + gate-corner geometry, no vision)
that flies a quadrotor through multi-gate racing courses (incl. inverted /
split-S) at, ideally, champion level. Intended pipeline: classical expert →
distill to a "good-enough" init → RL to make it fast and reliable (multi-gate
chaining). This policy is later distilled into experimental student
architectures, so the teacher must be genuinely capable, not just a warm body.

Course distribution of record: swift_v4 (the "real" 7-gate course) as an anchor,
plus procedural flow / chicane / split-S / mixed-inverted at ~10–13 m spacing for
generalization. The geometric SE(3)/min-jerk expert flies this at **95.8% success
/ 0.064 m track error @ 1.8–2.6 m/s** (validated via `eval_controller` preflight).

---

## 2. What we proved about Behavior Cloning / DAgger (the distillation arm)

Headline: **BC/DAgger from the geometric expert does NOT produce a usable
standalone teacher.** Best result across many configs was ~0–1 gate, unreliable.
The reasons are now fully understood.

### 2.1 Raw CTBR (body-rate) BC is closed-loop unstable
- The cloned policy matches the expert almost perfectly **open-loop**: pred-vs-expert
  action corr **0.994–0.998**, body-rate MAE **~0.018 rad/s**, held-out loss ≈ 0.
- In closed loop it still fails: drifts off-line, `gate_miss_rate ≈ 0` (never even
  reaches a gate plane), ends ~3.3 m off. Body-rate commands integrate twice
  (rate → attitude → position), so tiny errors compound and the policy leaves the
  data manifold, where it has no supervision.
- **Implication:** raw rate BC cannot be a final policy. This is the known reason
  real systems (Swift, etc.) learn CTBR with RL, not BC.

### 2.2 Waypoint+speed action (LQR inner loop) is the right *standalone* interface, but only marginally stable
- Policy emits a waypoint+speed setpoint; `WaypointLQRController` closes the inner
  loop with full state feedback (Markovian, stabilizing).
- Bootstrap clone tracks to **0.78–1.0 gate** vs 3.3 m for CTBR — clearly better.
- BUT closed-loop result is **high variance**: identical bootstrap procedure gave
  **0.0, 0.71, and 1.0 gates** across three runs with near-zero fit loss each time.
  The clone sits on a stability knife-edge. BC alone → marginal, unreliable.

### 2.3 DAgger had two real implementation bugs (both fixed) — but even fixed, the BC ceiling stands
1. **Stale relabel reference:** DAgger relabeled the student's deviated states
   against the wall-clock-time-indexed trajectory point (`traj.sample(t)`). A
   drifting/lagging student was told to fly to an unreachable "where you should've
   been," producing physically inconsistent recovery labels.
   *Fix:* project to the nearest path point + lookahead (`_nearest_ref`), matching
   how `synthetic_recovery` already labeled. (`collect_dataset.py`)
2. **Sample-balancing overfit (the big one):** `sample_kind_balancing` forced
   `dagger: 0.65` of every batch when dagger was ~0.5% of samples (a handful of
   short episodes), upsampling them ~100× and erasing the clean policy each round.
   *Proven on identical data:* balanced → **0.00 gates**; natural → **0.50 gates**.
   *Fix:* disabled aggressive balancing; rely on keep-all aggregation + β-decay.

   After both fixes, **inner-BC loss no longer degrades across rounds** (was
   −3.9 → −1.9; now flat), i.e. DAgger no longer corrupts the data — but the
   closed-loop ceiling is unchanged because the underlying clone is marginal.

### 2.4 Final 6-round waypoint DAgger result (fixes applied)
| round | gates (/~6) | gate_miss_rate | final dist | score |
|------|------------|----------------|-----------|-------|
| 0 (bootstrap) | 0.00 | 0.04 | 3.26 m | −0.06 |
| 1 (best) | 0.00 | 0.00 | 4.09 m | −0.03 |
| 2 | 0.00 | 0.00 | 8.18 m | −0.19 |
| 3 | 0.00 | 0.00 | 10.95 m | −0.27 |
| 4 | 0.08 | 0.96 | 2.19 m | −1.81 |
| 5 | 0.21 | 0.54 | 4.52 m | −0.74 |

No collapse (the fixes hold), faint late upward signal (rounds 4–5 begin reaching
gates as dagger data accumulates), but nowhere near usable. **Distillation's
correct role is an RL warm-start, not a standalone teacher.**

---

## 3. What we proved about RL (the arm that actually works)

The existing PPO curriculum **already warm-starts from a CTBR BC checkpoint**
(`paper_teacher_ctbr_bc_v5/best/model.pt`) and it works where it matters:

### 3.1 The BC warm-start is effective for RL — keep it
From `paper_teacher_ctbr_stage12_v11_ppo`:
- **Stage 0 (single gate): 82% success by iteration 50.** The CTBR BC init gives
  stable flight + gate passing essentially for free. From scratch would waste huge
  compute rediscovering this.
- Reconciles with §2.1: BC instability only matters when BC is the *final* policy.
  As an RL initializer it just seeds flight behavior + action scales; PPO learns
  the stabilizing feedback in closed loop. **CTBR is fine in RL.** No need to
  switch the RL action space to waypoint.

### 3.2 The real bottleneck: the gate-chaining transition cliff
- `stage12_v11`: at the stage 0→1 (single → chaining) transition (~iter 450),
  success **crashes 0.82 → 0.03**, `mean_reward` spikes to **−1791**, then claws
  back only to ~0.18–0.25 over 2000 iters. Never stably chains.
- `stage23_v13`: stage 0 ~0.5 → transition → **0.0**.
- `stage23_v12`: stuck oscillating 0.1–0.5 at stage 0, never stably advances.
- The −1791 reward spike at the transition says the **reward changes too sharply**
  when the goal becomes multi-gate — the policy is thrown far off its operating
  point and PPO can't recover. This is the [[gate-chaining-cliff]] problem.

---

## 4. Recommendations for the new RL setup

1. **Warm-start from CTBR BC. Do not train from scratch, do not switch to waypoint
   for RL.** Use the latest BC warmup (v5/v6 lineage; same arch as
   `configs/exp/flightmare/paper/ctbr_bc.yaml`). It buys ~82% single-gate success
   at iter 0.
2. **Make the BC warmup match the deployment course distribution.** The current BC
   warmups (bc_v4/v5) are swift_v4-only. Re-collect the warmup on the *aligned*
   procedural+swift distribution (`configs/flightmare_bc/ctbr_v7_geom.yaml`, which
   now writes `course_distribution` into the manifest so eval auto-matches) so the
   RL init isn't off-distribution from the chaining courses.
3. **Attack the chaining transition, not the init.** Likely levers:
   - **Reward continuity** across the 1→N gate change — the −1791 spike must be
     removed (e.g. per-gate progress reward that doesn't discontinuously re-baseline
     when the goal count changes; cap per-step negative reward).
   - **Stable-advancement gating** — only advance a stage when stage 0 is *stably*
     solved over a window (v11 attempted this; tighten the criterion).
   - **Gentler curriculum** — smaller jumps (1 → 2 → 3 gates) and/or overlap stages
     so the distribution shifts gradually.
   - **KL / entropy control at the transition** — prevent the policy from being
     blown off its operating point when the reward landscape shifts.
4. **Keep the privileged state-only obs and CTBR action** — both are validated.

---

## 5. Reusable assets produced during this investigation

- **`scripts/flightmare_bc/collect_dataset.py`**
  - DAgger collection generalized to CTBR *or* waypoint students
    (`BaseAutonomyController` dispatch + `VehicleState`); previously CTBR-only.
  - Nearest-point relabel (`_nearest_ref` + lookahead) for DAgger.
  - `course_distribution` surfaced to the manifest top level so closed-loop eval
    auto-matches the collection course (fixes a silent train/eval mismatch).
- **`src/robotics/data.py`** — `confidence_floor` consumes `expert/confidence` to
  down-weight low-confidence relabels (no-op at 1.0).
- **`src/robotics/distill_trainer.py`** — per-round schedules: `bc_loss_schedule`
  (Huber→GNLL), `clean_expert_fraction_schedule` / `synthetic_recovery_fraction_schedule`
  (β-decay), optional `scenario_weight_schedule`.
- **`configs/flightmare_bc/ctbr_v7_geom.yaml`** — geometric procedural+swift
  collection, preflight-gated; records all action labels (ctbr/waypoint/motor), so a
  single dataset serves any action interface.
- **Configs** `ctbr_distill_procedural_v2.yaml` (ctbr) and
  `wp_distill_procedural_v1.yaml` (waypoint) — aligned distill loops with balancing
  off. Useful as warm-up generators, not as final teachers.
- **MPPI expert is wired but currently fails preflight** (0% success, doesn't reach
  gate 1 — a real bug in `src/robotics/flightmare_experts/mppi.py`). Geometric is the
  working expert. MPPI is preflight-gated, not used.

---

## 6. One-paragraph summary for the RL design doc

Distillation from the geometric expert cannot produce a capable teacher: raw CTBR
BC is closed-loop unstable (perfect open-loop clone, 0 gates closed-loop), waypoint
BC is only marginally stable (0–1 gate, high variance), and DAgger — even after
fixing a stale-reference bug and a catastrophic sample-balancing overfit — only
reaches ~0.2 gates because the underlying clone is marginal. RL is the answer and
already half-works: a CTBR BC warm-start gives 82% single-gate success at iteration
0, confirming the warm-start is the right move (not from scratch, and CTBR is fine
in RL because PPO supplies the closed-loop feedback BC can't). The unsolved problem
is the **gate-chaining transition**, where reward collapses (−1791 spike) the moment
the goal becomes multi-gate. Build the new RL setup as: CTBR + privileged state-only
obs + BC warm-start re-collected on the aligned procedural distribution, with the
engineering effort spent on reward continuity and stable curriculum advancement
through the 1→N gate transition.
