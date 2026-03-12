    #!/bin/bash

# trt 빌드 스크립트
# 스크립트 실행 인자들
# 1번째 인자: onnx 모델 경로
# 2번째 인자: 저장할 engine 경로
# 3번쨰 인자: precision 설정, 없으면 자동으로 fp16
# 주요 옵션:
#   --memPoolSize=workspace:2048: 2GB 작업 메모리
#   --builderOptimizationLevel=5: 최적화 레벨 5 (최고)
#   --useCudaGraph: CUDA Graph 가속
#   --dumpProfile: 성능 프로파일 저장
#   --exportTimes: 타임스탬프 JSON 저장

ONNX_PATH=$1
ENGINE_PATH=$2
PRECISION=${3:-fp16}

if [ -z "$ONNX_PATH" ] || [ -z "$ENGINE_PATH" ]; then
    echo "Usage: ./trt_build.sh <onnx_path> <engine_path> [fp16|fp32|int8]"
    exit 1
fi

PRECISION_FLAG=""

if [ "$PRECISION" = "fp16" ]; then
    PRECISION_FLAG="--fp16"
elif [ "$PRECISION" = "int8" ]; then
    PRECISION_FLAG="--int8"
fi

trtexec \
  --onnx=$ONNX_PATH \
  --saveEngine=$ENGINE_PATH \
  $PRECISION_FLAG \
  --memPoolSize=workspace:2048 \
  --builderOptimizationLevel=5 \
  --useCudaGraph \
  --dumpProfile \
  --separateProfileRun \
  --exportTimes=${ENGINE_PATH%.engine}.json
