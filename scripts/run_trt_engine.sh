#!/bin/bash

MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
  echo "Usage: bash run_engine.sh <model_name>"
  exit 1
fi

BASE_DIR=artifacts/${MODEL_NAME}

if [ ! -d "$BASE_DIR" ]; then
  echo "Model directory not found: $BASE_DIR"
  exit 1
fi

echo "Running engines for model: $MODEL_NAME"
echo "Base directory: $BASE_DIR"
echo ""

for PREC_DIR in ${BASE_DIR}/*; do
  if [ -d "$PREC_DIR" ]; then

    ENGINE_PATH=$(find "$PREC_DIR" -maxdepth 1 -name "*.engine" | head -n 1)

    if [ -z "$ENGINE_PATH" ]; then
      echo "No engine found in $PREC_DIR"
      continue
    fi

    LOG_PATH="${PREC_DIR}/run.log"
    JSON_PATH="${PREC_DIR}/times.json"
    METRIC_PATH="${PREC_DIR}/metrics.txt"

    echo "---------------------------------------"
    echo "Precision dir : $PREC_DIR"
    echo "Engine        : $ENGINE_PATH"
    echo ""

    trtexec \
      --loadEngine="$ENGINE_PATH" \
      --iterations=200 \
      --warmUp=20 \
      --exportTimes="$JSON_PATH" \
      > "$LOG_PATH" 2>&1

    if [ ! -f "$JSON_PATH" ]; then
      echo "Timing JSON not generated."
      continue
    fi

    # Average latency 계산
    AVG_LATENCY=$(jq '[.[].latencyMs] | add / length' "$JSON_PATH")

    # Throughput (qps)
    THROUGHPUT=$(awk "BEGIN {print 1000/$AVG_LATENCY}")

    echo "Average Latency (ms): $AVG_LATENCY" > "$METRIC_PATH"
    echo "Throughput (qps): $THROUGHPUT" >> "$METRIC_PATH"

    echo "Saved metrics to $METRIC_PATH"
    echo ""

  fi
done

echo "All runs completed."