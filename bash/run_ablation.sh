#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# RQ4 (Ablation): switch off ONE thing at a time and measure the drop against
# the full system.  "w/o X" means "without X".
# -----------------------------------------------------------------------------
# Runs, for every seed, the Full config plus one run per ablation row of both
# heatmaps (figures/ablation_strategy_heatmap.png and ablation_size_heatmap.png).
# Every row here is a real, self-contained flag -- nothing needs a code edit.
#
# Output: outputs/rq4_ablation/<dataset>__<victim>__<config>__seed<seed>.json
# Aggregate afterwards with:
#   python scripts/aggregate_results.py outputs/rq4_ablation --csv rq4.csv

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/outputs/rq4_ablation"
mkdir -p "${OUT_DIR}"

DATASET="${DATASET:-ml-latest-small}"
VICTIM="${VICTIM:-lightgcn}"
ROUNDS="${ROUNDS:-18}"
N_WORKERS="${N_WORKERS:-8}"
BUDGET="${BUDGET:-0.01}"
# The paper averages over 5 seeds. Override with e.g. SEEDS="42" for a quick smoke.
SEEDS="${SEEDS:-42 43 44 45 46}"

COMMON=(--dataset "${DATASET}" --victim "${VICTIM}" --rounds "${ROUNDS}" \
        --n_workers "${N_WORKERS}" --budget "${BUDGET}")

run() {   # run <config-name> <extra flags...>
  local cfg="$1"; shift
  local out="${OUT_DIR}/${DATASET}__${VICTIM}__${cfg}__seed${SEED}.json"
  echo "=== ${cfg} (seed ${SEED}) ==="
  python "${ROOT}/scripts/run_agas.py" "${COMMON[@]}" --seed "${SEED}" "$@" \
    --out "${out}" || echo "  (${cfg} failed; continuing)"
}

echo "[RQ4] Ablation sweep on ${DATASET} / ${VICTIM}, seeds: ${SEEDS}"
echo "[RQ4] Output directory: ${OUT_DIR}"

for SEED in ${SEEDS}; do
  # Full system (everything on): the baseline the "w/o" rows are compared against.
  run full --coordinator-agent-memory --profile-validator

  # ---- component / size heatmap rows (figures/ablation_size_heatmap.png) ----
  run wo_coordinator   --random-coordinator
  run wo_profiler      --disable-roles pr
  run wo_sniper        --disable-roles sn
  run wo_camouflageur  --disable-roles ca
  run wo_inactive      --disable-roles in
  run wo_memory        --no-coordinator-agent-memory
  run wo_signals       --disable-signals
  run wo_validator     --no-profile-validator

  # ---- strategy heatmap rows (figures/ablation_strategy_heatmap.png) ----
  for s in s1 s2 s3 s4 s5 s6 s7 s8; do
    run "wo_${s}" --disable-strategies "${s}"
  done
done

echo "[RQ4] Done. Aggregate with: python scripts/aggregate_results.py ${OUT_DIR} --csv rq4.csv"
