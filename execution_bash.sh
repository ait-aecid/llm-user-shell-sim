#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results

N_JOBS=3
DATASET="Data_WP"

LOG_TYPES=(
  #"audit"
  "syslog"
)

#INTER_MODELS=(
#  dummy_most_frequent
#  dummy_stratified
#  gnb
#  logreg
#  svm
#  sgd_hinge
#  sgd_log
#  ridge
#  knn
#)

TFIDF_MODELS=(
  dummy_most_frequent
  dummy_stratified
  svm
  logreg
  sgd_hinge
  sgd_log
  pa_like
  ridge
  mnb
  cnb
  bnb
)

echo "=== $(date) ALL EXPERIMENTS start ==="

for LOG_TYPE in "${LOG_TYPES[@]}"; do

  #echo "=== $(date) dataset=${DATASET} log_type=${LOG_TYPE} INTER_TIMES start ==="

  #for MODEL in "${INTER_MODELS[@]}"; do
  #  echo "=== $(date) inter_times model=${MODEL} dataset=${DATASET} log_type=${LOG_TYPE} start ==="
  #
  #  OMP_NUM_THREADS=1 \
  #  MKL_NUM_THREADS=1 \
  #  OPENBLAS_NUM_THREADS=1 \
  #  NUMEXPR_NUM_THREADS=1 \
  #  python -m experiments.inter_times_360_nested \
  #    --model "$MODEL" \
  #    --n_jobs "$N_JOBS" \
  #    --log_type "$LOG_TYPE" \
  #    --dataset "$DATASET" \
  #    --benchmark \
  #    --out_csv "results/inter_times_${MODEL}_100_nested_${DATASET}_${LOG_TYPE}.csv" \
  #    2>&1 | tee "logs/inter_times_${MODEL}_100_${DATASET}_${LOG_TYPE}.log"
  #done

  echo "=== $(date) dataset=${DATASET} log_type=${LOG_TYPE} TFIDF start ==="

  for MODEL in "${TFIDF_MODELS[@]}"; do
    echo "=== $(date) tfidf model=${MODEL} dataset=${DATASET} log_type=${LOG_TYPE} start ==="

    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    python -m experiments.tfidf_360_nested \
      --model "$MODEL" \
      --n_jobs "$N_JOBS" \
      --log_type "$LOG_TYPE" \
      --dataset "$DATASET" \
      --limit_outer 100 \
      --benchmark \
      --out_csv "results/tfidf_${MODEL}_100_nested_${DATASET}_${LOG_TYPE}.csv" \
      2>&1 | tee "logs/tfidf_${MODEL}_100_${DATASET}_${LOG_TYPE}.log"
  done

done

echo "=== $(date) ALL EXPERIMENTS DONE ==="