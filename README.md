# Early-Exit ResNet-18 for Efficient Edge Inference on Jetson AGX Orin

## Project Structure
```text
CAPSTONE-EARLYEXIT/
├── artifacts/      
├── export/
│   ├── build_trt_engine.sh
│   ├── export_pt_onnx.py
│   └── run_trt_engine.sh
│
├── results/
│
├── src/
│   ├── configs/
│   ├── datasets/
│   ├── engine/
│   ├── models/
│   │   ├── ee_resnet18.py
│   │   └── plain_resnet18.py
│   ├── utils/
│   └── train.py
│
├── main.py
├── pipeline.sh
├── requirements.txt
├── README.md
└── .gitignore
```


---
## 환경 셋팅
```bash
python3.10 -m venv {환경이름}
source {환경이름}/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
---
## run command
### 전체 파이프라인 한번에 실행하는 명령어
- pt, .onnx 변환 -> build trt engind -> tensorrt engine 실행 및 로그 저장
```bash
./pipeline.sh {model_name}
```

### 개별 실행 방법은....  
- pt, .onnx 변환
```bash
python -m export.export_pt_onnx {model_name}
```

- tensorrt 변환
```bash
./export/trt_build.sh {model_name}
```

- tensorrt engine(runtime) 실행
```bash
./export/run_trt_engine.sh {model_name}
```
