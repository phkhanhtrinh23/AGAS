#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# RQ2 (Stealth) + RQ3 (Detector): run AGAS once, then evaluate t-SNE + 5
# detectors against the same poisoned matrix.
# -----------------------------------------------------------------------------
# Outputs:
#   outputs/rq2_stealth/<dataset>__<victim>__seed<seed>.json   (poisoned trace)
#   outputs/rq3_detect/<dataset>__<victim>__seed<seed>.json    (detector metrics)
#
# Detectors evaluated (see experiment_detect.tex):
#   BaseDetect, DHAGCN, PCASelectUsers, GAGE, MD-CBA
# Mapping to the paper:
#   - figures/benchmark_fig7_tsne_distance_users_target.png   (RQ2 left)
#   - figures/stealth_tsne_clean_vs_fake.png                  (RQ2 right)
#   - tables/detection_mf.tex                                 (RQ3 table)
#   - figures/detect_rec10_by_victim.png                      (RQ3 Rec@10)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STEALTH_DIR="${ROOT}/outputs/rq2_stealth"
DETECT_DIR="${ROOT}/outputs/rq3_detect"
mkdir -p "${STEALTH_DIR}" "${DETECT_DIR}"

DATASET="${DATASET:-ml-latest-small}"
VICTIM="${VICTIM:-lightgcn}"
SEED="${SEED:-42}"
ROUNDS="${ROUNDS:-18}"
N_WORKERS="${N_WORKERS:-8}"
BUDGET="${BUDGET:-0.01}"

out_stealth="${STEALTH_DIR}/${DATASET}__${VICTIM}__seed${SEED}.json"
out_detect="${DETECT_DIR}/${DATASET}__${VICTIM}__seed${SEED}.json"

echo "[RQ2/RQ3] Stealth + detector evaluation"
echo "[RQ2] Output: ${out_stealth}"
echo "[RQ3] Output: ${out_detect}"
echo "[RQ2] Maps to figures/benchmark_fig7_tsne_distance_users_target.png and stealth_tsne_clean_vs_fake.png"
echo "[RQ3] Maps to tables/detection_mf.tex and figures/detect_rec10_by_victim.png"

# Step 1: produce poisoned matrix.
python "${ROOT}/scripts/run_agas.py" \
  --dataset "${DATASET}" \
  --victim "${VICTIM}" \
  --rounds "${ROUNDS}" \
  --seed "${SEED}" \
  --n_workers "${N_WORKERS}" \
  --budget "${BUDGET}" \
  --out "${out_stealth}"

# Step 2: post-hoc detector + stealth evaluation.
# The legacy CLI exposes the detector battery via `evaluate-detectors` (uses
# the poisoned interactions saved in --episode JSON above).
PYTHONPATH="${ROOT}/src" python -m agas.cli evaluate-detectors \
  --episode "${out_stealth}" \
  --detectors "BaseDetect,DHAGCN,PCASelectUsers,GAGE,MD-CBA" \
  --output "${out_detect}" || true

echo "[RQ2/RQ3] Done."
