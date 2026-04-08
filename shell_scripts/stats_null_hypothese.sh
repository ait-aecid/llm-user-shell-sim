#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results

DATASET="Data"
N_START=21
N_END=108

LOG_TYPES=(
  #"syslog"
  #"nextcloud"
  "audit"
)

echo "=== $(date) NULL-PERMUTATION EXPERIMENTS start ==="

for LOG_TYPE in "${LOG_TYPES[@]}"; do
  echo "=== $(date) dataset=${DATASET} log_type=${LOG_TYPE} start ==="

  for i in $(seq "$N_START" "$N_END"); do
    echo "=== $(date) dataset=${DATASET} log_type=${LOG_TYPE} assignment_idx=${i} complexity start ==="

    python -m stats_tools.complexity_metrics \
      --log_type "$LOG_TYPE" \
      --assignment_mode indexed_stratified \
      --assignment_idx "$i" \
      2>&1 | tee "logs/complexity_${DATASET}_${LOG_TYPE}_assignment_${i}.log"

    echo "=== $(date) dataset=${DATASET} log_type=${LOG_TYPE} assignment_idx=${i} complexity done ==="

    echo "=== $(date) dataset=${DATASET} log_type=${LOG_TYPE} assignment_idx=${i} one_gram start ==="

    python -m stats_tools.one_gram \
      --log_type "$LOG_TYPE" \
      --assignment_mode indexed_stratified \
      --assignment_idx "$i" \
      2>&1 | tee "logs/one_gram_${DATASET}_${LOG_TYPE}_assignment_${i}.log"

    echo "=== $(date) dataset=${DATASET} log_type=${LOG_TYPE} assignment_idx=${i} one_gram done ==="
  done
done

echo "=== $(date) NULL-PERMUTATION EXPERIMENTS DONE ==="