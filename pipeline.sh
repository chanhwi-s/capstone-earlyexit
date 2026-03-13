#!/bin/bash

MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
  echo "Usage: bash pipeline.sh <model_name>"
  exit 1
fi

# 이 스크립트가 있는 디렉토리를 프로젝트 루트로 사용
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "====================================="
echo "Step 1: Export Pytorch model to .pt, .onnx"
echo "====================================="
python ${PROJECT_ROOT}/scripts/export_pt_onnx.py $MODEL_NAME || exit 1

echo ""
echo "====================================="
echo "Step 2: Build TensorRT Engines"
echo "====================================="
bash ${PROJECT_ROOT}/scripts/build_trt_engine.sh $MODEL_NAME || exit 1

echo ""
echo "====================================="
echo "Step 3: Run Engines"
echo "====================================="
bash ${PROJECT_ROOT}/scripts/run_trt_engine.sh $MODEL_NAME || exit 1

echo ""
echo "====================================="
echo "Pipeline completed successfully"
echo "====================================="