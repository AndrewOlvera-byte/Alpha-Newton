#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

LOG_DIR="${LOG_DIR:-outputs/_ppo_logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="${LOG_DIR}/run_ppo_both_${RUN_ID}.log"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$*" | tee -a "${MASTER_LOG}"
}

run_one() {
  local exp="$1"
  local run_name="$2"
  local log_file="${LOG_DIR}/${run_name}_${RUN_ID}.log"
  local -a cmd=(python -u -m src.entrypoints.train_ppo --exp "${exp}")

  log "starting ${exp}"
  log "per-run log: ${log_file}"
  log "command: ${cmd[*]}"

  set +e
  {
    printf '[%s] command: %s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "${cmd[*]}"
    "${cmd[@]}"
  } 2>&1 | tee -a "${MASTER_LOG}" "${log_file}"
  local status="${PIPESTATUS[0]}"
  set -e

  if [[ "${status}" -ne 0 ]]; then
    log "FAILED ${exp} exit=${status}"
    exit "${status}"
  fi
  log "finished ${exp}"
}

log "logs directory: ${LOG_DIR}"
log "WANDB_MODE=${WANDB_MODE}"
log "PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"

run_one "flightmare/flightmare_ctbr_ppo_state_v1" "flightmare_ctbr_ppo_state_v1"
run_one "flightmare/flightmare_waypoint_ppo_state_v1" "flightmare_waypoint_ppo_state_v1"

log "done"
