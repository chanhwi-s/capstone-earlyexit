# Early-Exit ResNet-18 for Efficient Edge Inference on Jetson AGX Orin

## Project Structure
```text
CAPSTONEDESIGN/
├── artifacts/             # .pt, .onnx, trt 변환 결과 저장
│
├── export/
│   ├── export_pt_onnx.py  # PyTorch → ONNX export script
│   └── trt_build.sh       # ONNX → TensorRT engine build script
│
├── src/
│   ├── models/
│   │   ├── resnet18.py
│   │   └── resnet18_pt_ee.py
│   ├── datasets/
│   ├── engine/
│   ├── configs/        
│   ├── experiments/
│   ├── utils/
│   └── train.py
│
├── main.py
├── requirements.txt
└── README.md
```
---
## run command
### 1. .pt, .onnx 변환
python -m export.export_pt_onnx

### 2. tensorrt 변환
./export/trt_build.sh {model_name} {precision}

### 3. tensorrt engine(runtime) 실행
export/run_trt_engine.sh artifacts/{model_name}/{precision}.engine {precision}_runtime
