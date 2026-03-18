#!/bin/bash

MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
  echo "Usage: bash run_trt_engine.sh <model_name>"
  exit 1
fi

CONFIG_PATH="configs.yaml"
BASE_DIR="artifacts/${MODEL_NAME}"
METRIC_PATH="${BASE_DIR}/metric.txt"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
  echo "Model directory not found: $BASE_DIR"
  exit 1
fi

# -----------------------------
# YAML → 환경변수
# -----------------------------
eval $(python3 - <<EOF
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}"))

print("ITER=" + str(cfg["run"]["iterations"]))
print("WARMUP=" + str(cfg["run"]["warmup"]))
print("STREAMS=" + str(cfg["run"]["streams"]))
print("DEVICES=\"" + " ".join(cfg["build"]["devices"]) + "\"")
print("PRECISIONS=\"" + " ".join(cfg["build"]["precisions"]) + "\"")
EOF
)

echo "Model: ${MODEL_NAME}" > "$METRIC_PATH"
echo "=========================================" >> "$METRIC_PATH"
echo "" >> "$METRIC_PATH"

for DEVICE in $DEVICES
do
  echo "[${DEVICE}]" >> "$METRIC_PATH"
  echo "" >> "$METRIC_PATH"

  for PRECISION in $PRECISIONS
  do
    TARGET_DIR="${BASE_DIR}/${DEVICE}_${PRECISION}"
    ENGINE_PATH="${TARGET_DIR}/model.engine"

    if [ ! -f "$ENGINE_PATH" ]; then
      continue
    fi

    LOG_PATH="${TARGET_DIR}/run.log"
    JSON_PATH="${TARGET_DIR}/times.json"
    TEGRA_LOG="${TARGET_DIR}/tegrastats.log"
    RUNTIME_LAYERINFO_PATH="${TARGET_DIR}/run_layer_device.json"

    echo "Running ${DEVICE}_${PRECISION}..."

    tegrastats --interval 100 --logfile "$TEGRA_LOG" &
    TEGRA_PID=$!
    sleep 1

    trtexec \
      --loadEngine="$ENGINE_PATH" \
      --iterations=${ITER} \
      --warmUp=${WARMUP} \
      --streams=${STREAMS} \
      --exportTimes="$JSON_PATH" \
      --exportLayerInfo="$RUNTIME_LAYERINFO_PATH" \
      > "$LOG_PATH" 2>&1

    sleep 1
    kill $TEGRA_PID 2>/dev/null

    if [ ! -f "$JSON_PATH" ]; then
      continue
    fi

    AVG_LATENCY=$(jq '[.[].latencyMs] | add / length' "$JSON_PATH")
    P95=$(jq '[.[].latencyMs] | sort | .[length*0.95|floor]' "$JSON_PATH")
    P99=$(jq '[.[].latencyMs] | sort | .[length*0.99|floor]' "$JSON_PATH")
    STD=$(jq '
      ([.[].latencyMs] | add / length) as $mean
      | [.[].latencyMs | pow(. - $mean;2)]
      | add / length
      | sqrt
    ' "$JSON_PATH")

    TRT_THROUGHPUT=$(grep "\[I\] Throughput" "$LOG_PATH" | head -n1 | awk -F'Throughput: ' '{print $2}' | awk '{print $1}')

    GPU_UTIL=$(grep "GR3D_FREQ" "$TEGRA_LOG" | \
      awk -F'GR3D_FREQ ' '{print $2}' | \
      awk -F'%' '{sum+=$1; n++} END {if(n>0) print sum/n; else print 0}')

    DLA_UTIL=$(grep "DLA0" "$TEGRA_LOG" | \
      awk -F'DLA0 ' '{print $2}' | \
      awk -F'%' '{sum+=$1; n++} END {if(n>0) print sum/n; else print 0}')

    AVG_RAM=$(grep "RAM" "$TEGRA_LOG" | \
      awk -F'RAM ' '{print $2}' | \
      awk -F'/' '{print $1}' | \
      awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print 0}')

    AVG_POWER=$(grep "VDD_GPU_SOC" "$TEGRA_LOG" | \
      awk -F'VDD_GPU_SOC ' '{print $2}' | \
      awk -F'mW' '{sum+=$1; n++} END {if(n>0) print sum/n/1000; else print 0}')

    TRT_THROUGHPUT=${TRT_THROUGHPUT:-0}
    AVG_POWER=${AVG_POWER:-0}

    PERF_PER_WATT=$(awk -v q="$TRT_THROUGHPUT" -v p="$AVG_POWER" \
    'BEGIN { if(p>0) print q/p; else print 0 }')

    FALLBACK_COUNT=$(grep -i "falling back to GPU" "$LOG_PATH" | wc -l)

    {
      echo "  ${PRECISION}"
      echo "    Avg Latency (ms): ${AVG_LATENCY}"
      echo "    P95 (ms): ${P95}"
      echo "    P99 (ms): ${P99}"
      echo "    Std Dev (ms): ${STD}"
      echo "    QPS: ${TRT_THROUGHPUT}"
      echo "    GPU Util (%): ${GPU_UTIL}"
      echo "    DLA Util (%): ${DLA_UTIL}"
      echo "    Avg RAM (MB): ${AVG_RAM}"
      echo "    Avg GPU Power (W): ${AVG_POWER}"
      echo "    Perf/Watt: ${PERF_PER_WATT}"
      echo "    Fallback Count: ${FALLBACK_COUNT}"
      echo ""
    } >> "$METRIC_PATH"

  done
done

echo "All runs completed."