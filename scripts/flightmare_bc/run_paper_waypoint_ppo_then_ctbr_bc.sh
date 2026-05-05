#!/usr/bin/env bash
# Sequentially run:
#   1) flightmare/paper/waypoint_curriculum_ppo  (PPO, warmstarted from BC)
#   2) flightmare/paper/ctbr_bc                  (BC pretraining for CTBR)
#
# Wandb is left ONLINE — the container is already `wandb login`-ed, so it
# just needs to be allowed to talk to the API. We only unset WANDB_MODE if
# something upstream forced it offline.
#
# Usage (inside the alpha-newton container, or via docker compose exec):
#   bash scripts/flightmare_bc/run_paper_waypoint_ppo_then_ctbr_bc.sh
#
# Resume-safe: if step 1 already produced its final/best checkpoint and you
# only want step 2, just rerun — step 1 will execute again unless you comment
# it out. (PPO doesn't have a clean "skip if done" sentinel like BC does.)

set -uo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs outputs

# Wandb online — assumes container already has `wandb login` credentials.
unset WANDB_MODE          # in case a parent shell forced offline
unset WANDB_DISABLED
export WANDB_SILENT=false
# Reduce HDF5 lock contention on shared filesystems.
export HDF5_USE_FILE_LOCKING=FALSE

ts=$(date +%Y%m%d_%H%M%S)
overall_log="logs/paper_wp_ppo_then_ctbr_bc_${ts}.log"
echo "[paper_seq] start $(date -Is) -> ${overall_log}" | tee -a "${overall_log}"

run_step() {
  local kind="$1"      # bc | ppo
  local exp="$2"
  local name="$3"
  local log="logs/${name}_${ts}.log"

  echo "[paper_seq] >>> ${kind} ${name} (log: ${log})" | tee -a "${overall_log}"
  local start
  start=$(date +%s)
  if ! python -m "src.entrypoints.train_${kind}" --exp "${exp}" 2>&1 | tee "${log}"; then
    local rc=${PIPESTATUS[0]}
    echo "[paper_seq] FAIL ${name} rc=${rc} after $(( $(date +%s) - start ))s" | tee -a "${overall_log}"
    return "${rc}"
  fi
  echo "[paper_seq] OK ${name} in $(( $(date +%s) - start ))s" | tee -a "${overall_log}"
  return 0
}

status=0

# Step 1 — PPO curriculum on waypoint actions.
if ! run_step ppo \
      "flightmare/paper/waypoint_curriculum_ppo" \
      "paper_waypoint_curriculum_ppo_v4"; then
  status=$?
  echo "[paper_seq] aborting after step 1 failure" | tee -a "${overall_log}"
  echo "[paper_seq] done $(date -Is) status=${status}" | tee -a "${overall_log}"
  exit "${status}"
fi

# Step 2 — BC pretraining on CTBR actions.
if ! run_step bc \
      "flightmare/paper/ctbr_bc" \
      "paper_ctbr_bc_v4"; then
  status=$?
fi

echo "[paper_seq] done $(date -Is) status=${status}" | tee -a "${overall_log}"
exit "${status}"
