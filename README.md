# Early-Exit ResNet-18 for Efficient Edge Inference on Jetson AGX Orin

현재 readme는 gpt가 작성한걸 단순 복붙하였습니다.

## Project Structure
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


## Overview

This project explores inference optimization techniques for convolutional neural networks on edge devices.
The main objective is to reduce inference latency and computational cost while maintaining classification accuracy.

To achieve this, we implement **Early-Exit mechanisms on ResNet-18**, allowing the network to terminate inference early for easy samples instead of forwarding through all layers.

The optimized model is later deployed on **NVIDIA Jetson AGX Orin** using **TensorRT acceleration**.

Key goals of the project:

* Implement a baseline ResNet-18 training pipeline in PyTorch
* Introduce Early-Exit branches to reduce unnecessary computation
* Evaluate accuracy–latency trade-offs
* Deploy optimized models on Jetson AGX Orin
* Apply TensorRT optimization for edge inference


---

## Method

### Baseline Model

The baseline architecture is **ResNet-18**, a residual convolutional neural network composed of:

* 4 residual stages
* 8 residual blocks
* global average pooling
* final fully connected classifier

The model is first trained without any early exit branches to establish a performance baseline.

---

### Early-Exit Mechanism

Early-Exit networks attach auxiliary classifiers to intermediate layers of the model.

During inference:

1. Input propagates through the network
2. At intermediate exit points, the model evaluates prediction confidence
3. If confidence exceeds a threshold, inference terminates early
4. Otherwise, computation continues to deeper layers

Benefits:

* Reduced average inference latency
* Lower computational cost
* Energy efficiency on edge hardware

Typical exit locations in ResNet-18:

* After **layer2**
* After **layer3**

---

### Edge Deployment Optimization

For deployment on edge hardware, the trained model will be optimized using:

* **ONNX export**
* **TensorRT engine conversion**
* **FP16 / INT8 inference**

These optimizations allow the model to achieve significantly lower latency on Jetson devices.

---

## Project Structure

```
capstonedesign/

configs/
    train.yaml

datasets/
    dataloader.py

engine/
    trainer.py

models/
    resnet18.py

utils/
    config.py
    experiment.py
    logger.py
    metrics.py
    seed.py

train.py
requirements.txt
README.md
```

---

## Training

### Install dependencies

```
pip install -r requirements.txt
```

### Configure experiment

Edit:

```
configs/train.yaml
```

Example configuration:

```yaml
dataset:
  name: cifar10
  data_root: ./data
  num_workers: 4

train:
  batch_size: 128
  epochs: 10
  seed: 42

optimizer:
  lr: 0.01
  momentum: 0.9
  weight_decay: 5e-4

model:
  name: resnet18
```

### Run training

```
python train.py
```

Training outputs are stored in:

```
experiments/
```

including:

* model checkpoints
* training logs
* TensorBoard logs

---

## Deployment on Jetson AGX Orin

The deployment pipeline is:

```
PyTorch
   ↓
ONNX Export
   ↓
TensorRT Engine
   ↓
Jetson AGX Orin Inference
```

Typical workflow:

1. Train the model on a workstation
2. Export the model to ONNX
3. Convert ONNX to TensorRT engine
4. Deploy optimized model to Jetson

---

## Reproducibility

To ensure reproducible experiments:

* random seeds are fixed
* deterministic dataloader workers are used
* configuration files store experiment parameters

Seed control includes:

* Python random
* NumPy
* PyTorch
* DataLoader workers

---

## Future Work

Planned extensions include:

* Early-Exit ResNet-18 implementation
* Dynamic inference thresholds
* latency / accuracy trade-off analysis
* TensorRT engine benchmarking
* evaluation on Jetson AGX Orin hardware

---

## License

This project is developed for academic and research purposes.
