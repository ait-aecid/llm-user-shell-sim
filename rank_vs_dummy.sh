#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-}"                 # 50 or 360
METRIC="${METRIC:-test_f1_macro}"

if [[ -z "$GROUP" ]]; then
  echo "Usage: $0 {50|360}"
  echo "Example: METRIC=test_mcc $0 50"
  exit 1
fi

if [[ "$GROUP" == "50" ]]; then
  DUMMY="results/dummy_stratified_tfidf_50_nested_results.csv"
  GLOB="results/*_50_nested_results.csv"
elif [[ "$GROUP" == "360" ]]; then
  DUMMY="results/inter_times_dummy_stratified_360_nested.csv"
  GLOB="results/inter_times_*_360_nested.csv"
else
  echo "GROUP must be 50 or 360"
  exit 1
fi

if [[ ! -f "$DUMMY" ]]; then
  echo "Dummy CSV not found: $DUMMY"
  exit 1
fi

# Header
echo -e "model\tmeanΔ\tmedianΔ\tstdΔ\tci_lo\tci_hi\twins\tlosses\tties\tpairs\tp_sign\tp_wilcoxon"

# IMPORTANT: model is CSV A, dummy is CSV B -> delta = model - dummy (positive = improvement)
for f in $GLOB; do
  [[ "$f" == "$DUMMY" ]] && continue
  python analysis/paired_significance.py "$f" "$DUMMY" --metric "$METRIC" --tsv
done | sort -t $'\t' -k2,2nr