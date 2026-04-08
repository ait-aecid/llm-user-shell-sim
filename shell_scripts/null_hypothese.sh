#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results

N_JOBS=7
DATASET="Data"

LOG_TYPES=(
  #"audit"
  "syslog"
  "nextcloud"
)

TFIDF_MODELS=(
  "svm"
)

echo "=== $(date) ALL EXPERIMENTS start ==="

for LOG_TYPE in "${LOG_TYPES[@]}"; do
  echo "=== $(date) dataset=${DATASET} log_type=${LOG_TYPE} TFIDF start ==="

  for MODEL in "${TFIDF_MODELS[@]}"; do
    echo "=== $(date) tfidf model=${MODEL} dataset=${DATASET} log_type=${LOG_TYPE} start ==="

    for i in $(seq 5 119); do
      echo "=== $(date) tfidf model=${MODEL} dataset=${DATASET} log_type=${LOG_TYPE} assignment_idx=${i} start ==="

      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      python -m experiments.tfidf_360_nested \
        --model "$MODEL" \
        --n_jobs "$N_JOBS" \
        --log_type "$LOG_TYPE" \
        --dataset "$DATASET" \
        --limit_outer 50 \
        --benchmark \
        --randomize_actor_labels \
        --assignment_idx "$i" \
        --out_csv "results/tfidf_${MODEL}_100_nested_${DATASET}_${LOG_TYPE}_assignment_${i}.csv" \
        2>&1 | tee "logs/tfidf_${MODEL}_100_${DATASET}_${LOG_TYPE}_assignment_${i}.log"

      echo "=== $(date) tfidf model=${MODEL} dataset=${DATASET} log_type=${LOG_TYPE} assignment_idx=${i} done ==="
    done
  done
done

echo "=== $(date) ALL EXPERIMENTS DONE ==="