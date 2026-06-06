# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CIFAR-10 분류 실험을 위한 딥러닝 과제 프레임워크. 학생이 `src/models.py`의 `# TODO` 스켈레톤을 채워 Linear → MLP → ConvNetV1 → ConvNetV2 → ResNet50 순서로 모델을 구현하고 실험한다.

## Commands

### 설치
```bash
pip install -r "lab2 (problem)/requirements.txt"
```

### 학습 실행
```bash
# 기본 실행 (configs/ 아래 yaml 파일 지정)
python "lab2 (problem)/train.py" --config "lab2 (problem)/configs/linear.yaml"

# 커맨드라인으로 설정 오버라이드 (dotted.key=value 문법)
python "lab2 (problem)/train.py" --config "lab2 (problem)/configs/convnet_v2.yaml" --opts train.epochs=10 optimizer.lr=0.0005
```

### 빠른 동작 확인 (단일 에포크)
```bash
python "lab2 (problem)/train.py" --config "lab2 (problem)/configs/linear.yaml" --opts train.epochs=1
```

## Architecture

### 학습 파이프라인 흐름
`train.py` → `src/config.py` (YAML 로드 + override) → `src/data.py` (DataLoader) → `src/models.py` (모델 생성) → `src/engine.py` (train/eval loop) → `src/loggers.py` (파일 로그 + W&B)

### 모델 계층 구조 (`src/models.py`)
| 문제 | 클래스 | 핵심 구조 |
|------|--------|----------|
| Problem 1 | `LinearClassifier` | 이미지를 flatten → `nn.Linear` 1개 |
| Problem 2 | `MLP` / `MLPBlock` | flatten → 2개의 `MLPBlock`(Linear+ReLU) → head |
| Problem 3 | `ConvNetV1` / `ConvNetV1Block` | 4-stage Conv(3×3, pad=1) + Global Avg Pool → head |
| Problem 4 | `ConvNetV2` / `ConvNetV2Block` | 4-stage Residual block + DownsampleBlock (2×2 Conv + BN) + GAP + BN + Dropout → head |
| Problem 6 | `build_resnet50` | `timm` ResNet-50, stride 수정 (conv1/maxpool stride=1) |

`build_model(model_cfg)` 함수가 config의 `model.name`을 읽어 해당 클래스를 인스턴스화한다. 모델 이름 별칭 목록은 `models.py:216` 참고.

### Config 시스템 (`src/config.py`)
YAML 파일이 단일 소스이며 모든 하이퍼파라미터를 포함한다. `--opts` 플래그로 dotted key(`train.epochs`, `model.dropout`, `augment.train.random_crop_padding` 등)를 런타임에 오버라이드할 수 있다. 실행 시 config 사본이 `outputs/<run_name>_<timestamp>/config.yaml`에 자동 저장된다.

### 데이터 증강 (`src/data.py`)
`build_transform()`의 `# Fill this` 블록이 Problem 5/6에서 채워야 할 부분이다. 순서: `RandomCrop` → `RandomHorizontalFlip` → `RandAugment` → `PILToTensor` → `Normalize` → `RandomErasing`. 증강 파라미터는 config의 `augment.train.*` 키로 제어.

### 출력 구조
```
outputs/<run_name>_<timestamp>/
  config.yaml          # 실행 설정 스냅샷
  train.log            # Python logging 출력
  checkpoints/
    best.pt            # 최고 val accuracy 체크포인트
    last.pt            # 마지막 에포크 체크포인트
```

체크포인트에는 `model`, `optimizer`, `scheduler` state_dict와 `metrics`, `config`가 함께 저장된다.

### W&B 로깅
config의 `logging.wandb.enabled: true`로 활성화. 기본값은 `false`이므로 오프라인 실험 시 별도 설정 불필요.

## 구현 가이드라인

- **`NotImplementedError` 위치**: `LinearClassifier.__init__`, `MLPBlock.forward`, `MLP.__init__/forward`, `ConvNetV1.__init__`, `ConvNetV2Block.__init__`, `ConvNetV2.__init__`(downsamples/stages), `data.py`의 증강 블록
- **ConvNetV2 잔차 연결**: `ConvNetV2Block.forward`는 이미 `x + self.droppath(h)` 형태로 작성되어 있으므로, `__init__`에서 conv1/norm1/conv2/norm2만 정의하면 된다.
- **파라미터 수 계산**: `src/utils.py`의 `count_parameters(model)` 함수로 확인 가능. 학습 시작 시 자동으로 로그에 출력된다.
- **디바이스**: `train.device: auto`로 설정하면 GPU가 있으면 CUDA, 없으면 CPU를 자동 선택한다.
