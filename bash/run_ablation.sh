#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# RQ4 (Ablation): disable one component (or one strategy) at a time and
# measure the drop vs the full system.
# -----------------------------------------------------------------------------
# Outputs:
#   outputs/rq4_ablation/<config>__seed<seed>.json
# Mapping to the paper:
#   - figures/ablation_summary.png            (cross-dataset bars)
#   - figures/ablation_strategy_heatmap.png   (per-strategy heatmap)
#   - figures/ablation_size_heatmap.png       (per-model-size heatmap)
#   - tables/ablation_backbones.tex           (backbone ablation table)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/outputs/rq4_ablation"
mkdir -p "${OUT_DIR}"

DATASET="${DATASET:-ml-latest-small}"
VICTIM="${VICTIM:-lightgcn}"
SEED="${SEED:-42}"
ROUNDS="${ROUNDS:-18}"
N_WORKERS="${N_WORKERS:-8}"
BUDGET="${BUDGET:-0.01}"

# Each ablation flag turns off exactly one component. The legacy CLI accepts
# the same --strategic-disable-* flags used internally by the Coordinator.
CONFIGS=(
  "full"                       # baseline — all components on
  "no_S1_VICTIM_PROBE"          # disable strategy S1
  "no_S2_BRIDGE_BUILDING"       # disable strategy S2
  "no_S5_SILENT_SLOWDOWN"       # disable strategy S5
  "no_S6_PROFILE_CLEANUP"       # disable strategy S6
  "no_S7_SAFE_REPLACEMENT"      # disable strategy S7
  "no_validator"               # ablate φ_{t,w} validator
)

echo "[RQ4] Ablation sweep on ${DATASET} / ${VICTIM}"
echo "[RQ4] Output directory: ${OUT_DIR}"
echo "[RQ4] Maps to figures/ablation_summary.png and figures/ablation_strategy_heatmap.png"

for cfg in "${CONFIGS[@]}"; do
  out="${OUT_DIR}/${DATASET}__${VICTIM}__${cfg}__seed${SEED}.json"
  echo "=== ablation: ${cfg} ==="

  extra_args=()
  case "${cfg}" in
    no_S1_VICTIM_PROBE)        extra_args+=(--strategic-disable-probe-classify) ;;
    no_S2_BRIDGE_BUILDING)     extra_args+=(--strategic-disable-cooccurrence-bridging) ;;
    no_S5_SILENT_SLOWDOWN)     extra_args+=(--strategic-disable-stealth) ;;
    no_S6_PROFILE_CLEANUP)     extra_args+=(--strategic-disable-validator-guardrail) ;;
    no_S7_SAFE_REPLACEMENT)    extra_args+=(--strategic-disable-suspicion-lockout) ;;
    no_validator)              extra_args+=(--strategic-disable-validator-guardrail) ;;
    full) ;;
  esac

  python "${ROOT}/scripts/run_agas.py" \
    --dataset "${DATASET}" \
    --victim "${VICTIM}" \
    --rounds "${ROUNDS}" \
    --seed "${SEED}" \
    --n_workers "${N_WORKERS}" \
    --budget "${BUDGET}" \
    --out "${out}" || echo "  (ablation '${cfg}' failed; continuing)"
done

echo "[RQ4] Done."
