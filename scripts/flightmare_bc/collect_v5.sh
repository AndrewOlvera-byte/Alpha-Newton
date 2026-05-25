#!/usr/bin/env bash
set -euo pipefail

# Flightmare BC v5 collection — REAL Flightmare dynamics on the RACING plant.
#
# Difference vs v4:
#   v4 collected on the stock Flightmare plant, where the C++ integrator clamps
#   per-axis body rates at 6 rad/s. Downstream teacher PPO configs use a racing
#   plant with omega_max_body=[18,18,18]. Loading a v4 BC checkpoint into PPO
#   on the racing plant means every BC-imitated action has been recorded under
#   different realized dynamics than what PPO will actually execute — so the
#   policy starts off-distribution and immediately saturates the action bounds.
#
#   v5 collects on the same racing plant the teacher PPO will use, so the BC
#   (state, action) pairs are consistent with deployment dynamics.
#
# Speed envelope and SE3 controller are unchanged. At 1.8-2.4 m/s the SE3
# controller rarely commands rates above ~3 rad/s, so the higher integrator
# cap mostly changes the available headroom rather than the recorded actions
# themselves; what changes is that the recorded state evolution is now
# faithful to a plant that can rotate at 18 rad/s when PPO needs it to.
#
# Run inside the alpha-newton Docker service:
#   docker compose exec alpha-newton bash scripts/flightmare_bc/collect_v5.sh

EPISODES=${EPISODES:-2000}
SEED=${SEED:-0}
OUT=${OUT:-data/flightmare/bc_v5}

python -m scripts.flightmare_bc.collect \
  --out "$OUT" \
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
  --seed "$SEED" \
  --mass 0.73 \
  --arm-length 0.17 \
  --inertia 0.0025 0.0021 0.0043 \
  --k-thrust 1.91e-6 \
  --k-torque 2.6e-7 \
  --motor-omega-min 150.0 \
  --motor-omega-max 3000.0 \
  --motor-tau 0.0001 \
  --thrust-map 1.3298253500372892e-06 0.0038360810526746033 -1.7689986848125325 \
  --kappa 0.016 \
  --omega-max-body 18.0 18.0 18.0

python -m scripts.flightmare_bc.transform_to_v3 \
  --data-dir "$OUT" \
  --force

python -m scripts.flightmare_bc.recompute_v3_norm_stats \
  --data-dir "$OUT" \
  --split train

python -m scripts.flightmare_bc.diag_bc_v3_geometry \
  --data-dir "$OUT" \
  --episodes 20 \
  --fail-on-missing-quat
