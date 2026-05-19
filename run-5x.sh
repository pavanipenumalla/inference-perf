#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run-name-prefix>"
  echo "  Runs run-all.sh 5 times with names: <prefix>-1, <prefix>-2, ..., <prefix>-5"
  exit 1
fi

PREFIX="$1"
TOTAL=5

for ((i=1; i<=TOTAL; i++)); do
  echo "========================================"
  echo "Iteration ${i}/${TOTAL}: ${PREFIX}-${i}"
  echo "========================================"
  ./run-all.sh "${PREFIX}-${i}"
  echo "Iteration ${i}/${TOTAL} complete."
  echo ""
done

echo "All ${TOTAL} iterations complete."