#!/usr/bin/env bash
set -euo pipefail

NS="llm-d-precise"
CM="gaie-kv-events-epp"
DEPLOY="gaie-kv-events-epp"
CM_KEY="precise-prefix-cache-config.yaml"
BASE_DIR="/mnt/data/pavani/inference-perf"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run-name-prefix>"
  exit 1
fi

RUN_PREFIX="$1"

patch_plugin_params() {
  local strategy="$1"
  local params="$2"

  local current
  current=$(kubectl -n "$NS" get cm "$CM" -o json | python3 -c "
import sys, json
cm = json.load(sys.stdin)
print(cm['data']['$CM_KEY'])
")

  local updated
  updated=$(export _PARAMS="$params"; echo "$current" | python3 -c "
import sys, os, yaml

doc = yaml.safe_load(sys.stdin)
new_params = yaml.safe_load(os.environ['_PARAMS'])
for p in doc['plugins']:
    if p['type'] == 'program-aware-fairness':
        p['parameters'] = new_params
        break
yaml.dump(doc, sys.stdout, default_flow_style=False, sort_keys=False)
")

  kubectl -n "$NS" create cm "$CM" --from-literal="$CM_KEY=$updated" --dry-run=client -o yaml | \
    kubectl -n "$NS" apply -f -
}

run_experiment() {
  local name="$1"
  local strategy="$2"
  local params="$3"

  echo "========================================"
  echo "Run: $name | Strategy: $strategy"
  echo "========================================"

  if [ -d "${BASE_DIR}/${name}/reports" ]; then
    echo "SKIP: ${BASE_DIR}/${name}/reports already exists."
    return 0
  fi

  echo "Patching configmap..."
  patch_plugin_params "$strategy" "$params"

  echo "Restarting deployment $DEPLOY..."
  kubectl -n "$NS" rollout restart deployment/"$DEPLOY"
  kubectl -n "$NS" rollout status deployment/"$DEPLOY" --timeout=300s

  echo "Running experiment: $name"
  ./run.sh "$name"

  echo "Done: $name"
  echo ""
}

# Run 1: LAS
run_experiment "${RUN_PREFIX}-las" "las" "
strategy: las
"

# Run 2: DRR
run_experiment "${RUN_PREFIX}-drr" "drr" "
strategy: drr
quantumTokens: 6000
weightDeficit: 0.85
weightDrrHeadWait: 0.15
"

# Run 3: RR
run_experiment "${RUN_PREFIX}-rr" "rr" "
strategy: rr
deferRRCursor: true
"

# --- Generate comparison plots ---
COMPARE_OUT="${BASE_DIR}/${RUN_PREFIX}-comparison/"
echo "Generating comparison plots in ${COMPARE_OUT}..."
cd "${BASE_DIR}"
source env/bin/activate
python3 compare_reports.py \
    --drr "${BASE_DIR}/${RUN_PREFIX}-drr/reports" \
    --las "${BASE_DIR}/${RUN_PREFIX}-las/reports" \
    --rr  "${BASE_DIR}/${RUN_PREFIX}-rr/reports" \
    -o "${COMPARE_OUT}"
echo "Comparison plots saved to ${COMPARE_OUT}"

echo "All 3 runs complete."