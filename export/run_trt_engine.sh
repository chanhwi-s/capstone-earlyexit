#!/bin/bash

ENGINE_PATH=$1
NAME=$2

if [ -z "$ENGINE_PATH" ]; then
  echo "Usage: bash run_engine.sh <engine_path> <run_name>"
  exit 1
fi

if [ -z "$NAME" ]; then
  NAME="run"
fi

OUT_DIR=$(dirname $ENGINE_PATH)

LOG_PATH=${OUT_DIR}/${NAME}.log
JSON_PATH=${OUT_DIR}/${NAME}.json

echo "Running TensorRT engine..."
echo "Engine: $ENGINE_PATH"
echo "Log   : $LOG_PATH"
echo "JSON  : $JSON_PATH"

trtexec \
  --loadEngine=$ENGINE_PATH \
  --iterations=200 \
  --warmUp=20 \
  --exportTimes=$JSON_PATH \
  > $LOG_PATH 2>&1

echo "Done."
