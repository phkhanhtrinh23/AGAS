#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# RQ1 (Performance): how does AGAS perform across victim families on ML-100K?
# -----------------------------------------------------------------------------
# Tiny reviewer-friendly sample: ML-100K x 3 representative victims x 1 seed.
# For the full benchmark (6 datasets x 11 victims x 5 seeds) sweep all values
# in the loops below — see experiment.tex.
#
# Outputs:
#   outputs/rq1_performance/<dataset>__<victim>__seed<seed>.json
# Mapping to the paper:
#   - tables/benchmark_unpopular.tex      (HR@10, NDCG@10 head/mid table)
#   - figures/benchmark_popularity_regime_summary.png  (radar plot)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/outputs/rq1_performance"
mkdir -p "${OUT_DIR}"

DATASETS=("ml-latest-small")   # add ml-1m / netflix / amazon / douban / genome2021 for full sweep
VICTIMS=("gmf" "lightgcn" "xsimgcl")
SEEDS=(42)
ROUNDS="${ROUNDS:-18}"
N_WORKERS="${N_WORKERS:-8}"
BUDGET="${BUDGET:-0.01}"

echo "[RQ1] Performance benchmark — tiny ML-100K sample"
echo "[RQ1] Output directory: ${OUT_DIR}"
echo "[RQ1] Map outputs to tables/benchmark_unpopular.tex"

for dataset in "${DATASETS[@]}"; do
  for victim in "${VICTIMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out="${OUT_DIR}/${dataset}__${victim}__seed${seed}.json"
      echo "=== ${dataset} / ${victim} / seed=${seed} ==="
      python "${ROOT}/scripts/run_agas.py" \
        --dataset "${dataset}" \
        --victim "${victim}" \
        --rounds "${ROUNDS}" \
        --seed "${seed}" \
        --n_workers "${N_WORKERS}" \
        --budget "${BUDGET}" \
        --out "${out}"
    done
  done
done

echo "[RQ1] Done. Expected outputs in ${OUT_DIR}"
