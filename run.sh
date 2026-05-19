#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
BASE_DIR="/mnt/data/pavani/otel/inference-perf"
PROMETHEUS_DIR="/mnt/data/prometheus-2.53.0.linux-amd64"
SCRAPER_DIR="/mnt/data/sai/loadgen-llm-d-scheduler"
DEPLOYMENT="gaie-kv-events-epp"
MODEL_DEPLOY="ms-kv-events-llm-d-modelservice-decode"
NAMESPACE="llm-d-precise"
GATEWAY_SVC="infra-kv-events-inference-gateway-istio"
EPP_SVC="gaie-kv-events-epp"
SCRAPE_SUBSYSTEM="program_aware"
SCRAPE_DURATION=100
CONFIG_FILE="examples/otel/configs/simple/two-sessions-sequential.yml"

# --- Parse arguments ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder-name> [config-file]"
    echo "  folder-name : results folder created under ${BASE_DIR}"
    echo "  config-file : inference-perf config (default: ${CONFIG_FILE})"
    exit 1
fi

FOLDER_NAME="$1"
CONFIG_FILE="${2:-$CONFIG_FILE}"
RESULTS_DIR="${BASE_DIR}/${FOLDER_NAME}"
METRICS_FILE="${RESULTS_DIR}/metrics.jsonl"

# --- Create results folder ---
mkdir -p "${RESULTS_DIR}"
echo "Results directory: ${RESULTS_DIR}"

# --- Track background PIDs for cleanup ---
PIDS=()
INTERRUPTED=false

cleanup() {
    echo ""
    echo "Shutting down gracefully..."

    # Send SIGTERM first to allow processes to exit cleanly
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Sending SIGTERM to PID $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    # Wait up to 10 seconds for graceful exit
    local timeout=10
    for pid in "${PIDS[@]}"; do
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$timeout" ]; do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "  PID $pid did not exit in ${timeout}s, sending SIGKILL"
            kill -9 "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done

    echo "All background processes stopped."
}
trap 'INTERRUPTED=true; cleanup' INT TERM
trap cleanup EXIT

# --- Step 0a: Restart model service (scale 0 → 1) ---
echo "Scaling down model service ${MODEL_DEPLOY}..."
kubectl -n "${NAMESPACE}" scale deployment/"${MODEL_DEPLOY}" --replicas=0
kubectl -n "${NAMESPACE}" rollout status deployment/"${MODEL_DEPLOY}" --timeout=120s
echo "Model service scaled to 0."

echo "Scaling up model service ${MODEL_DEPLOY}..."
kubectl -n "${NAMESPACE}" scale deployment/"${MODEL_DEPLOY}" --replicas=1

MAX_RETRIES=30
RETRY_DELAY=30
for ((i=1; i<=MAX_RETRIES; i++)); do
    if kubectl -n "${NAMESPACE}" rollout status deployment/"${MODEL_DEPLOY}" --timeout=60s 2>/dev/null; then
        echo "Model service is ready."
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo "ERROR: Model service did not become ready after ${MAX_RETRIES} attempts."
        exit 1
    fi
    echo "  Attempt ${i}/${MAX_RETRIES} failed (cluster may be unreachable), retrying in ${RETRY_DELAY}s..."
    sleep "${RETRY_DELAY}"
done

# --- Step 0b: Restart EPP ---
echo "Restarting deployment ${DEPLOYMENT}..."
kubectl rollout restart deployment "${DEPLOYMENT}"
echo "Waiting for rollout to complete..."
kubectl rollout status deployment "${DEPLOYMENT}" --timeout=300s
echo "Deployment ready."

# --- Step 1: Start Prometheus ---
echo "Starting Prometheus..."
"${PROMETHEUS_DIR}/prometheus" \
    --config.file="${PROMETHEUS_DIR}/prometheus.yml" \
    --web.listen-address=:9090 \
    &>"${RESULTS_DIR}/prometheus.log" &
PIDS+=($!)
echo "  Prometheus PID: ${PIDS[-1]}"
sleep 2

# --- Step 2: Port-forward inference gateway ---
echo "Port-forwarding inference gateway..."
kubectl port-forward -n "${NAMESPACE}" "svc/${GATEWAY_SVC}" 8080:80 \
    &>"${RESULTS_DIR}/gateway-portforward.log" &
PIDS+=($!)
echo "  Gateway port-forward PID: ${PIDS[-1]}"

# --- Step 3: Port-forward EPP ---
echo "Port-forwarding EPP..."
kubectl port-forward "svc/${EPP_SVC}" 9091:9090 \
    &>"${RESULTS_DIR}/epp-portforward.log" &
PIDS+=($!)
echo "  EPP port-forward PID: ${PIDS[-1]}"

sleep 3
echo "Port-forwards ready."

# --- Step 4: Start metrics scraper ---
echo "Starting metrics scraper (duration: ${SCRAPE_DURATION}s)..."
python3 "${SCRAPER_DIR}/scrape_metrics.py" \
    --url http://localhost:9091/metrics \
    --subsystem "${SCRAPE_SUBSYSTEM}" \
    --duration "${SCRAPE_DURATION}" \
    --output "${METRICS_FILE}" \
    &>"${RESULTS_DIR}/scraper.log" &
PIDS+=($!)
echo "  Scraper PID: ${PIDS[-1]}"

# --- Step 5: Create patched config with local_storage.path pointing to results dir ---
REPORTS_SUBDIR="reports"
REPORTS_DIR="${RESULTS_DIR}/${REPORTS_SUBDIR}"
PATCHED_CONFIG="${RESULTS_DIR}/config.yml"
sed "s|^\(\s*path:\s*\).*|\1\"${REPORTS_DIR}\"|" "${BASE_DIR}/${CONFIG_FILE}" > "${PATCHED_CONFIG}"
echo "Patched config written to ${PATCHED_CONFIG}"

# --- Step 6: Run inference-perf ---
echo "Running inference-perf with config: ${PATCHED_CONFIG}"
cd "${BASE_DIR}"
source env/bin/activate
OTEL_TRACES_ENABLED=true inference-perf --config_file "${PATCHED_CONFIG}"
echo "inference-perf finished."

# --- Step 7: Generate plots ---
echo "Generating session duration plots..."
if [ -d "${REPORTS_DIR}" ]; then
    echo "  Using reports directory: ${REPORTS_DIR}"
    python3 "${BASE_DIR}/analyze_reports.py" \
        "${REPORTS_DIR}" \
        -o "${RESULTS_DIR}/"
    echo "  Plots saved to ${RESULTS_DIR}/"
else
    echo "  WARNING: Reports directory ${REPORTS_DIR} not found, skipping plots."
fi

# --- Step 8: Generate scraper plots ---
echo "Generating EPP scraper plots..."
if [ -f "${METRICS_FILE}" ]; then
    python3 "${SCRAPER_DIR}/analyze_epp.py" \
        "${METRICS_FILE}" \
        -o "${RESULTS_DIR}/scraper_plots"
    echo "  Scraper plots saved to ${RESULTS_DIR}/scraper_plots/"
else
    echo "  WARNING: Metrics file ${METRICS_FILE} not found, skipping scraper plots."
fi

# --- Done: cleanup runs via trap ---
echo "Run complete. Results are in: ${RESULTS_DIR}"
