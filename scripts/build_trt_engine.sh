#!/bin/bash

MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: ./export/trt_build.sh <model_name>"
    exit 1
fi

MODEL_DIR="artifacts/${MODEL_NAME}"
ONNX_PATH="${MODEL_DIR}/model.onnx"

if [ ! -f "$ONNX_PATH" ]; then
    echo "ONNX file not found: $ONNX_PATH"
    exit 1
fi

PRECISIONS=("fp32" "fp16" "int8")

for PRECISION in "${PRECISIONS[@]}"
do
    PRECISION_DIR="${MODEL_DIR}/${PRECISION}"
    mkdir -p $PRECISION_DIR

    ENGINE_PATH="${PRECISION_DIR}/model.engine"
    LOG_PATH="${PRECISION_DIR}/build_verbose.log"
    PROFILE_PATH="${PRECISION_DIR}/layer_profile.json"

    PRECISION_FLAG=""

    if [ "$PRECISION" = "fp16" ]; then
        PRECISION_FLAG="--fp16"
    elif [ "$PRECISION" = "int8" ]; then
        PRECISION_FLAG="--int8"
    fi

    echo "========================================="
    echo "Building ${MODEL_NAME} (${PRECISION})"
    echo "========================================="

    trtexec \
      --onnx=$ONNX_PATH \
      --saveEngine=$ENGINE_PATH \
      $PRECISION_FLAG \
      --memPoolSize=workspace:2048 \
      --builderOptimizationLevel=5 \
      --verbose \
      --profilingVerbosity=detailed \
      --dumpProfile \
      --separateProfileRun \
      --exportProfile=$PROFILE_PATH \
      --useCudaGraph \
      2>&1 | tee $LOG_PATH

done