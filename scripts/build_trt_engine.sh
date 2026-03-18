#!/bin/bash

MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: bash build_trt_engine.sh <model_name>"
    exit 1
fi

CONFIG_PATH="config.yaml"
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

DEVICES=($(yq '.build.devices[]' $CONFIG_PATH))
PRECISIONS=($(yq '.build.precisions[]' $CONFIG_PATH))

DLA_CORE=$(yq '.build.dla_core' $CONFIG_PATH)
ALLOW_FALLBACK=$(yq '.build.allow_gpu_fallback' $CONFIG_PATH)
WORKSPACE=$(yq '.build.workspace' $CONFIG_PATH)
OPT_LEVEL=$(yq '.build.opt_level' $CONFIG_PATH)
USE_CUDA_GRAPH=$(yq '.build.use_cuda_graph' $CONFIG_PATH)

for DEVICE in "${DEVICES[@]}"
do
  for PRECISION in "${PRECISIONS[@]}"
  do
      DIR_NAME="${DEVICE}_${PRECISION}"
      TARGET_DIR="${MODEL_DIR}/${DIR_NAME}"
      mkdir -p "$TARGET_DIR"

      ENGINE_PATH="${TARGET_DIR}/model.engine"
      LOG_PATH="${TARGET_DIR}/build_verbose.log"
      PROFILE_PATH="${TARGET_DIR}/layer_profile.json"

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
        --onnx=$ONNX_PATH \
        --saveEngine=$ENGINE_PATH \
        $PREC_FLAG \
        $DEVICE_FLAG \
        --memPoolSize=workspace:${WORKSPACE} \
        --builderOptimizationLevel=${OPT_LEVEL} \
        --verbose \
        --profilingVerbosity=detailed \
        --dumpProfile \
        --separateProfileRun \
        --exportProfile=$PROFILE_PATH \
        $CUDA_GRAPH_FLAG \
        2>&1 | tee $LOG_PATH

  done
done
