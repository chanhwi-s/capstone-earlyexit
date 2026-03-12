# Early-Exit ResNet-18 for Efficient Edge Inference on Jetson AGX Orin

## Project Structure
```text
CAPSTONEDESIGN
│
├── artifacts/                  # 학습 및 변환 결과물 저장
│   ├── onnx/                   # ONNX 모델 파일
│   │   └── plain_resnet18.onnx
│   ├── pytorch/                # PyTorch 모델 (.pt)
│   │   └── plain_resnet18.pt
│   └── tensorrt/               # TensorRT engine 파일 (생성 예정)
│
├── export/                     # 모델 변환 스크립트
│   └── export_to_onnx.py
│
├── src/                        # 학습 및 모델 소스 코드
│   ├── configs/                # 학습 설정 파일
│   │   └── train.yaml
│   │
│   ├── data/                   # 데이터 저장 디렉토리
│   │
│   ├── datasets/               # 데이터 로더 정의
│   │   └── dataloader.py
│   │
│   ├── engine/                 # 학습 루프 로직
│   │   └── trainer.py
│   │
│   ├── experiments/            # 실험 로그 및 체크포인트 저장
│   │
│   ├── models/                 # 모델 정의
│   │   ├── resnet18.py
│   │   └── resnet18_pt_ee.py
│   │
│   ├── utils/                 
│   │   ├── config.py
│   │   ├── experiment.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── seed.py
│   │
│   └── train.py                # 학습 실행 스크립트
│
├── main.py                  
├── requirements.txt
└── README.md
