#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# RQ5 (Efficiency): round-wise trade-off, runtime comparison, token-to-Top-10,
# and budget sweep.
# -----------------------------------------------------------------------------
# Outputs:
#   outputs/rq5_efficiency/round_<R>__seed<seed>.json   (round sweep)
#   outputs/rq5_efficiency/budget_<B>__seed<seed>.json  (budget sweep)
# Mapping to the paper:
#   - figures/efficiency_rounds_tradeoff.png       (HR@10 vs runtime by rounds)
#   - figures/efficiency_method_time_compare.png   (HR@10 + runtime per method)
#   - figures/efficiency_token_topk.png            (tokens until target enters Top-10)
#   - figures/efficiency_agents_tradeoff.png       (HR@10 vs budget 0.5%–3.0%)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/outputs/rq5_efficiency"
mkdir -p "${OUT_DIR}"

DATASET="${DATASET:-ml-latest-small}"
VICTIM="${VICTIM:-lightgcn}"
SEED="${SEED:-42}"
N_WORKERS="${N_WORKERS:-8}"

echo "[RQ5] Efficiency sweep on ${DATASET} / ${VICTIM}"
echo "[RQ5] Output directory: ${OUT_DIR}"
echo "[RQ5] Maps to figures/efficiency_rounds_tradeoff.png, efficiency_method_time_compare.png,"
echo "      efficiency_token_topk.png, efficiency_agents_tradeoff.png"

# 1) Round-wise trade-off: vary T in {4, 8, 12, 18}.
for rounds in 4 8 12 18; do
  out="${OUT_DIR}/round_${rounds}__seed${SEED}.json"
  echo "=== rounds=${rounds} ==="
  python "${ROOT}/scripts/run_agas.py" \
    --dataset "${DATASET}" \
    --victim "${VICTIM}" \
    --rounds "${rounds}" \
    --seed "${SEED}" \
    --n_workers "${N_WORKERS}" \
    --budget 0.01 \
    --out "${out}"
done

# 2) Budget sweep: vary the fake-user budget L in {0.005, 0.01, 0.02, 0.03}
#    (i.e. 0.5% to 3.0% of users) — see figures/efficiency_agents_tradeoff.png.
for budget in 0.005 0.01 0.02 0.03; do
  out="${OUT_DIR}/budget_${budget}__seed${SEED}.json"
  echo "=== budget=${budget} ==="
  python "${ROOT}/scripts/run_agas.py" \
    --dataset "${DATASET}" \
    --victim "${VICTIM}" \
    --rounds 18 \
    --seed "${SEED}" \
    --n_workers "${N_WORKERS}" \
    --budget "${budget}" \
    --out "${out}"
done

echo "[RQ5] Done."
