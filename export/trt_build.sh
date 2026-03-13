    #!/bin/bash

# trt 빌드 스크립트
# 인자: 변환 수행할 모델 일므
# 주요 옵션:
#   --memPoolSize=workspace:2048: 2GB 작업 메모리
#   --builderOptimizationLevel=5: 최적화 레벨 5 (최고)
#   --useCudaGraph: CUDA Graph 가속
#   --dumpProfile: 성능 프로파일 저장
#   --exportTimes: 타임스탬프 JSON 저장

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
    ENGINE_PATH="${MODEL_DIR}/${PRECISION}.engine"
    JSON_PATH="${MODEL_DIR}/${PRECISION}.json"
    LOG_PATH="${MODEL_DIR}/${PRECISION}_build.log"

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
      --useCudaGraph \
      --dumpProfile \
      --separateProfileRun \
      --exportTimes=$JSON_PATH \
      2>&1 | tee $LOG_PATH

done