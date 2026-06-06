from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth per sample.

    This class is provided so students do not need to depend on timm internals
    for ConvNetV3 regularization experiments.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# -----------------------------------------------------------------------------
# Problem 1: Linear Classifier
# -----------------------------------------------------------------------------
class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int = 3072, num_classes: int = 10):
        super().__init__()
        # Fill this: define a linear classifier head.
        # Hint: CIFAR-10 image input is flattened in forward(). # 레이어 
        self.head = nn.Linear(in_channels, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: # 데이터 흐름
        x = x.reshape((x.shape[0], -1))
        return self.head(x)


# -----------------------------------------------------------------------------
# Problem 2: Multilayer Perceptron (MLP)
# -----------------------------------------------------------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_channels: int = 3072, hidden_dim: int = 512):
        super().__init__()
        self.linear = nn.Linear(in_channels, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [수정 위치] MLPBlock.forward (56번째 줄 근처)
        # [수정 이유] 과제 요구사항: "Each hidden layer consists of Linear and ReLU layers"
        #   - Linear 변환(self.linear)으로 특징을 선형 변환한 뒤,
        #   - ReLU 활성화 함수로 비선형성을 도입해야 한다.
        #   - raise NotImplementedError 를 실제 연산으로 교체.
        return F.relu(self.linear(x))  # Linear → ReLU 순서로 적용


class MLP(nn.Module):
    def __init__(self, in_channels: int = 3072, hidden_dim: int = 512, num_classes: int = 10):
        super().__init__()
        # [수정 위치] MLP.__init__ — hidden layer 정의 블록 (63번째 줄 근처)
        # [수정 이유] 과제 요구사항:
        #   1. "MLP class has two hidden layers and one output layer"
        #      → MLPBlock 을 2개 생성한다.
        #   2. "Hidden layers should be grouped by nn.ModuleList"
        #      → nn.ModuleList 로 묶어야 PyTorch 가 파라미터를 올바르게 추적한다.
        #   채널 흐름:
        #     1st block: in_channels(3072) → hidden_dim  (이미지 차원 압축)
        #     2nd block: hidden_dim        → hidden_dim  (동일 차원 유지)
        self.hidden = nn.ModuleList([
            MLPBlock(in_channels, hidden_dim),  # 1st hidden layer: flatten 이미지 → hidden
            MLPBlock(hidden_dim, hidden_dim),   # 2nd hidden layer: hidden → hidden
        ])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) 형태의 이미지를 (B, 3072) 로 평탄화
        x = x.reshape((x.shape[0], -1))
        # [수정 위치] MLP.forward — hidden layer 순전파 블록 (69번째 줄 근처)
        # [수정 이유] self.hidden 의 각 MLPBlock 을 순서대로 통과시켜야 한다.
        #   - for 루프로 리스트 내 레이어를 순차적으로 실행한다.
        #   - raise NotImplementedError 를 실제 순전파 루프로 교체.
        for layer in self.hidden:
            x = layer(x)  # MLPBlock: Linear → ReLU
        return self.head(x)  # 최종 분류 헤드: hidden_dim → num_classes


# -----------------------------------------------------------------------------
# Problem 3: Convolutional Neural Network (ConvNetV1)
# -----------------------------------------------------------------------------
class ConvNetV1Block(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, dim, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x))


class ConvNetV1(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dims: Sequence[int] = (64, 128, 256, 512),
        strides: Sequence[int] = (1, 2, 2, 1),
        num_classes: int = 10,
    ):
        super().__init__()
        if len(dims) != len(strides):
            raise ValueError("dims and strides must have the same length.")

        # [수정 위치] ConvNetV1.__init__ — hidden layer 정의 블록
        # [수정 이유] 과제 요구사항:
        #   1. "four hidden layers" → dims 길이만큼 ConvNetV1Block 을 생성한다.
        #   2. "grouped by nn.ModuleList" → PyTorch 가 파라미터를 추적하도록 ModuleList 사용.
        #   3. "3x3 kernels, padding 1" → ConvNetV1Block 기본값(kernel_size=3, padding=1) 그대로.
        #   4. "channels from dims, strides from strides" → 인접 stage 간 채널 수를 이어준다.
        #
        #   채널 흐름 (기본 설정 dims=(64,128,256,512), strides=(1,2,2,1)):
        #     Block 0: in_channels(3)   → dims[0](64),  stride=1  (해상도 유지)
        #     Block 1: dims[0](64)      → dims[1](128), stride=2  (해상도 1/2 다운샘플)
        #     Block 2: dims[1](128)     → dims[2](256), stride=2  (해상도 1/4 다운샘플)
        #     Block 3: dims[2](256)     → dims[3](512), stride=1  (해상도 유지)
        #
        #   [in_channels] + list(dims)[:-1] 로 각 블록의 입력 채널을 구성한다.
        in_dims = [in_channels] + list(dims[:-1])  # [3, 64, 128, 256]
        self.layers = nn.ModuleList([
            ConvNetV1Block(in_dim=in_d, dim=d, stride=s)
            for in_d, d, s in zip(in_dims, dims, strides)
        ])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=(-1, -2))
        return self.head(x)


# -----------------------------------------------------------------------------
# Problem 4: Advanced Convolutional Neural Network (ConvNetV2)
# -----------------------------------------------------------------------------
class ConvNetV2DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, dim, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class ConvNetV2Block(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, droppath: float = 0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # Fill this: define conv1/norm1/conv2/norm2 for a residual block.
        raise NotImplementedError("Problem 4: implement ConvNetV2Block.__init__")
        self.droppath = DropPath(droppath) if droppath > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(x)), inplace=True)
        h = F.relu(self.norm2(self.conv2(h)), inplace=True)
        return x + self.droppath(h)


class ConvNetV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        blocks: Sequence[int] = (1, 2, 6, 1),
        dims: Sequence[int] = (96, 192, 384, 768),
        strides: Sequence[int] = (1, 2, 2, 1),
        num_classes: int = 10,
        droppath: float = 0.0,
        dropout: float = 0.0,
        droprate: float | None = None,
    ):
        super().__init__()
        if droprate is not None:
            dropout = droprate
        if not (len(blocks) == len(dims) == len(strides)):
            raise ValueError("blocks, dims, and strides must have the same length.")

        self.downsamples = nn.ModuleList()
        # Fill this: add one ConvNetV2DownsampleBlock per stage.
        raise NotImplementedError("Problem 4: implement ConvNetV2 downsamples")

        self.layers = nn.ModuleList()
        # Fill this: add the requested number of ConvNetV2Block layers per stage.
        # Hint: use nn.Sequential for each stage and optionally vary droppath per block.
        raise NotImplementedError("Problem 4: implement ConvNetV2 stages")

        self.norm = nn.BatchNorm1d(dims[-1])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for downsample, stage in zip(self.downsamples, self.layers):
            x = downsample(x)
            x = stage(x)
        x = x.mean(dim=(-1, -2))
        x = self.dropout(self.norm(x))
        return self.head(x)


# Backward-compatible aliases for last year's naming.
ConvNet = ConvNetV1
ResNet = ConvNetV2


# -----------------------------------------------------------------------------
# Problem 6: ResNet-50 experiments
# -----------------------------------------------------------------------------
def build_resnet50(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "ResNet50 experiments require timm. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    model = timm.create_model(
        model_name="resnet50.a1_in1k",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    model.conv1.stride = (1, 1)
    model.maxpool.stride = (1, 1)
    return model


def build_model(model_cfg: Dict) -> nn.Module:
    name = str(model_cfg.get("name", "linear")).lower()
    num_classes = int(model_cfg.get("num_classes", 10))

    if name in {"linear", "linear_classifier", "linearclassifier"}:
        return LinearClassifier(
            in_channels=int(model_cfg.get("in_channels", 3072)),
            num_classes=num_classes,
        )
    if name in {"mlp", "mlp-512", "mlp-2048"}:
        return MLP(
            in_channels=int(model_cfg.get("in_channels", 3072)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            num_classes=num_classes,
        )
    if name in {"convnet_v1", "convnet", "convnetv1"}:
        return ConvNetV1(
            in_channels=int(model_cfg.get("in_channels", 3)),
            dims=model_cfg.get("dims", [64, 128, 256, 512]),
            strides=model_cfg.get("strides", [1, 2, 2, 1]),
            num_classes=num_classes,
        )
    if name in {"convnet_v2", "convnet_v3_reg", "convnet_v3_aug", "resnet", "convnetv2"}:
        return ConvNetV2(
            in_channels=int(model_cfg.get("in_channels", 3)),
            blocks=model_cfg.get("blocks", [1, 2, 6, 1]),
            dims=model_cfg.get("dims", [96, 192, 384, 768]),
            strides=model_cfg.get("strides", [1, 2, 2, 1]),
            num_classes=num_classes,
            droppath=float(model_cfg.get("droppath", 0.0)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if name in {"resnet50_scratch", "resnet50_finetune", "resnet50"}:
        return build_resnet50(
            num_classes=num_classes,
            pretrained=bool(model_cfg.get("pretrained", name == "resnet50_finetune")),
        )
    raise ValueError(f"Unknown model name: {name}")
