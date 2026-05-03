#!/usr/bin/env bash
set -euo pipefail

# State-only v3 BC recollection for human-scale racing experiments.
# Run from the repo root inside the alpha-newton Docker service, or execute
# this script through docker compose exec.

python -m scripts.flightmare_bc.collect \
  --out data/flightmare/bc_v3_swift_like \
  --episodes 2000 \
  --backend flightgym \
  --course-mode swift_like \
  --random-start-gate \
  --num-gates 7 \
  --gate-size 1.6 \
  --fixed-gate-pos-noise 0.15 \
  --fixed-gate-yaw-noise 0.05 \
  --speed-range 4.0 8.0 \
  --lookahead-s 0.35 \
  --gate-approach-m 3.0 \
  --trajectory-min-segment-s 0.60 \
  --control-hz 100 \
  --max-steps 2500 \
  --max-world-radius 350.0 \
  --max-collective-thrust-g 5.5 \
  --gate-vehicle-radius 0.15 \
  --action-normalization bounds \
  --no-render \
  --val-frac 0.1 \
  --min-gate-completion 0.999 \
  --seed 0
