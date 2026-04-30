#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p outputs/_ppo_logs

run_one() {
  local exp="$1"
  local log_name="$2"
  echo "[run_ppo_both] starting ${exp}"
  python -m src.entrypoints.train_ppo --exp "${exp}" 2>&1 | tee "outputs/_ppo_logs/${log_name}.log"
}

run_one "flightmare/flightmare_ctbr_ppo_state_v1" "flightmare_ctbr_ppo_state_v1_$(date +%Y%m%d_%H%M%S)"
run_one "flightmare/flightmare_waypoint_ppo_state_v1" "flightmare_waypoint_ppo_state_v1_$(date +%Y%m%d_%H%M%S)"

echo "[run_ppo_both] done"
