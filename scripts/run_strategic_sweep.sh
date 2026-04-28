#!/bin/bash
# Strategic-coordinator ablation sweep.
#
# Reproduces the table in README.md under
# "Strategic coordinator policy (--coordinator-policy strategic)".
#
# Usage:
#   bash scripts/run_strategic_sweep.sh           # run v2 (current defaults)
#   STRATEGIC_VERSION=v1 bash scripts/...         # reproduce v1 (loose stealth)
#
# v1 used: --strategic-stealth-aggregate 0.55, no cooldown, full suppression.
# v2 uses: --strategic-stealth-aggregate 0.70, --strategic-stealth-cooldown 3,
#          partial suppression (only worst-aggregate sniper locked).
# v2 thresholds are now the policy defaults; v1 is reproduced by passing the
# old threshold and disabling the cooldown.

set -e
cd "$(dirname "$0")/.."

VERSION="${STRATEGIC_VERSION:-v2}"
OUTDIR="outputs/strategic_sweep_${VERSION}"
mkdir -p "${OUTDIR}"

if [[ "${VERSION}" == "v1" ]]; then
  STEALTH_FLAGS="--strategic-stealth-aggregate 0.55 --strategic-stealth-cooldown 0"
else
  STEALTH_FLAGS=""  # use v2 defaults baked into the policy
fi

run_one() {
  local tag=$1
  shift
  local out="${OUTDIR}/${tag}.json"
  local log="${OUTDIR}/${tag}.log"
  echo "=== ${tag} ==="
  PYTHONPATH=src python -m agas.cli run-transfer \
    --transfer-mode option-a \
    --processed-root processed \
    --dataset ml-latest-small \
    --max-interactions 10000 \
    --target-item-id 593 \
    --target-keyword horror \
    --num-steps 10 \
    --goal-rank 3 \
    --num-agents 8 \
    --transfer-attack-roles sniper,profiler \
    --worker-policy rule \
    --episode-model lightgcn \
    --victim-model-hint mf \
    --probe-steps 0 \
    --rule-max-snipers 3 \
    --sniper-start-step 2 \
    --target-models lightgcn \
    --transfer-candidate-set all_items \
    --target-epochs 3 \
    --target-embedding-dim 32 \
    --target-batch-size 512 \
    --no-stop-on-goal \
    --profiler-actions 3 \
    --camouflaguer-actions 3 \
    --profiler-bridge-method auto \
    --profiler-bridge-auto-threshold 20 \
    --profile-validator \
    --profile-validator-threshold 0.7 \
    ${STEALTH_FLAGS} \
    "$@" \
    --output "${out}" > "${log}" 2>&1
  tail -2 "${log}"
}

# Cell 1: rule-based baseline (no strategic wrapper at all).
run_one rule_baseline       --coordinator-policy rule
# Cell 2: strategic_full — both STEALTH_REBUILD and CONSENSUS_HOLD enabled.
run_one strategic_full      --coordinator-policy strategic
# Cell 3: STEALTH_REBUILD only.
run_one strategic_stealth   --coordinator-policy strategic --strategic-disable-reprobe
# Cell 4: CONSENSUS_HOLD only.
run_one strategic_reprobe   --coordinator-policy strategic --strategic-disable-stealth
# Cell 5: composition wrapper with both strategies disabled (= rule-based via wrapper).
run_one strategic_neither   --coordinator-policy strategic \
                            --strategic-disable-stealth --strategic-disable-reprobe
