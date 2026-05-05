#!/usr/bin/env bash
set -euo pipefail

# Train both v4 BC ablations (waypoint + motor action spaces) on bc_v4 data.
# Same hyperparameters except action_type — produces matched BC checkpoints
# that the paper PPO curriculum runs warm-start from.
#
# Run inside the alpha-newton Docker service. Sequential by default (sharing
# one GPU); set PARALLEL=1 to launch both in parallel (requires sufficient
# VRAM for two flightmare_actor instances + dataloaders).
#
#   docker compose exec alpha-newton bash scripts/flightmare_bc/train_both_v4.sh
#
# Outputs:
#   outputs/paper_waypoint_bc_v4/best/model.pt
#   outputs/paper_motor_bc_v4/best/model.pt

PARALLEL=${PARALLEL:-0}
SEED=${SEED:-0}

WAYPOINT_EXP=flightmare/paper/waypoint_bc
MOTOR_EXP=flightmare/paper/motor_bc

run_bc() {
  local exp=$1
  echo "===================================================================="
  echo "[train_both_v4] starting $exp"
  echo "===================================================================="
  python -m src.entrypoints.train_bc --exp "$exp"
}

if [[ "$PARALLEL" == "1" ]]; then
  run_bc "$WAYPOINT_EXP" &
  WAY_PID=$!
  run_bc "$MOTOR_EXP" &
  MOTOR_PID=$!
  wait "$WAY_PID" "$MOTOR_PID"
else
  run_bc "$WAYPOINT_EXP"
  run_bc "$MOTOR_EXP"
fi

echo
echo "[train_both_v4] both BC runs complete."
echo "  waypoint best: outputs/paper_waypoint_bc_v4/best/model.pt"
echo "  motor    best: outputs/paper_motor_bc_v4/best/model.pt"
