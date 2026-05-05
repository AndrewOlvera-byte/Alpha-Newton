#!/usr/bin/env bash
set -euo pipefail

# Flightmare BC v4 collection — REAL Flightmare dynamics (flightgym backend).
#
# Course strategy (see scripts/flightmare_bc/collect.py:_swift_like_gate_course
# and the README in this directory): canonical Swift Split-S 7-gate layout
# with per-gate position and yaw randomization. This is a deliberate choice:
# fully procedural S-switch geometries (course_mode=swift_v4) are infeasible
# for the current SE3+polynomial planner under flightgym dynamics — the BC
# collector throws away ~100% of episodes when asked to fly aggressive
# alternating S-curves at any non-trivial speed. swift_like + per-gate noise
# gives 100% acceptance, real procedural variation, and total course length
# matched to Swift.
#
# Speed envelope is locked to 1.8-2.4 m/s — empirically the regime where the
# SE3 controller + min-jerk polynomial tracks under flightgym dynamics with
# negligible error (~0.05 m mean). Higher speeds cause the polynomial to
# overshoot the first-gate aperture; they are recovered later via PPO speed-
# pressure rewards (see paper/*_curriculum_ppo.yaml).
#
# Run inside the alpha-newton Docker service:
#   docker compose exec alpha-newton bash scripts/flightmare_bc/collect_v4.sh

EPISODES=${EPISODES:-2000}
SEED=${SEED:-0}

python -m scripts.flightmare_bc.collect \
  --out data/flightmare/bc_v4 \
  --episodes "$EPISODES" \
  --backend flightgym \
  --course-mode swift_v4 \
  --num-gates 7 \
  --gate-size 1.6 \
  --gate-approach-m 3.0 \
  --fixed-gate-pos-noise 0.20 \
  --fixed-gate-yaw-noise 0.05 \
  --speed-range 1.8 2.4 \
  --lookahead-s 0.3 \
  --trajectory-min-segment-s 0.3 \
  --control-hz 100 \
  --max-steps 5000 \
  --max-world-radius 350.0 \
  --max-collective-thrust-g 5.5 \
  --gate-vehicle-radius 0.20 \
  --action-normalization bounds \
  --no-render \
  --val-frac 0.1 \
  --min-gate-completion 0.99 \
  --seed "$SEED"
