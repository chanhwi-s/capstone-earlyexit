#!/bin/bash

MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: bash build_trt_engine.sh <model_name>"
    exit 1
fi

CONFIG_PATH="configs.yaml"
MODEL_DIR="artifacts/${MODEL_NAME}"
ONNX_PATH="${MODEL_DIR}/model.onnx"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$ONNX_PATH" ]; then
    echo "ONNX file not found: $ONNX_PATH"
    exit 1
fi

# -----------------------------
# YAML → 환경변수 (Python 사용)
# -----------------------------
eval $(python3 - <<EOF
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}"))

print("DEVICES=\"" + " ".join(cfg["build"]["devices"]) + "\"")
print("PRECISIONS=\"" + " ".join(cfg["build"]["precisions"]) + "\"")
print("DLA_CORE=" + str(cfg["build"]["dla_core"]))
print("ALLOW_FALLBACK=" + str(cfg["build"]["allow_gpu_fallback"]).lower())
print("WORKSPACE=" + str(cfg["build"]["workspace"]))
print("OPT_LEVEL=" + str(cfg["build"]["opt_level"]))
print("USE_CUDA_GRAPH=" + str(cfg["build"]["use_cuda_graph"]).lower())
EOF
)

for DEVICE in $DEVICES
do
  for PRECISION in $PRECISIONS
  do
      DIR_NAME="${DEVICE}_${PRECISION}"
      TARGET_DIR="${MODEL_DIR}/${DIR_NAME}"
      mkdir -p "$TARGET_DIR"

      ENGINE_PATH="${TARGET_DIR}/model.engine"
      LOG_PATH="${TARGET_DIR}/build_verbose.log"
      PROFILE_PATH="${TARGET_DIR}/layer_profile.json"
      LAYERINFO_PATH="${TARGET_DIR}/build_layer_device.json"

      PREC_FLAG=""
      if [ "$PRECISION" = "fp16" ]; then
          PREC_FLAG="--fp16"
      elif [ "$PRECISION" = "int8" ]; then
          PREC_FLAG="--int8"
      fi

      DEVICE_FLAG=""
      if [ "$DEVICE" = "dla" ]; then
          DEVICE_FLAG="--useDLACore=${DLA_CORE}"
          if [ "$ALLOW_FALLBACK" = "true" ]; then
              DEVICE_FLAG="$DEVICE_FLAG --allowGPUFallback"
          fi
      fi

      CUDA_GRAPH_FLAG=""
      if [ "$USE_CUDA_GRAPH" = "true" ]; then
          CUDA_GRAPH_FLAG="--useCudaGraph"
      fi

      echo "========================================="
      echo "Building ${MODEL_NAME} (${DEVICE}, ${PRECISION})"
      echo "========================================="

      trtexec \
        --onnx="$ONNX_PATH" \
        --saveEngine="$ENGINE_PATH" \
        $PREC_FLAG \
        $DEVICE_FLAG \
        --memPoolSize=workspace:${WORKSPACE} \
        --builderOptimizationLevel=${OPT_LEVEL} \
        --dumpProfile \
        --exportLayerInfo="$LAYERINFO_PATH" \
        --exportProfile="$PROFILE_PATH" \
        $CUDA_GRAPH_FLAG \
        2>&1 | tee "$LOG_PATH"


  done
done