# src/defectvad/common/preprocessor.py

import torch
from typing import Dict, Any, Optional, Tuple, Union
from torchvision import transforms as T
from pathlib import Path
import numpy as np
from PIL import Image


class Preprocessor:
    """
    Anomalib 스타일의 전처리기 (Preprocessor)
    - 구성 기반 설정 (config-driven)
    - train/inference 모드 지원
    - PIL → Tensor → Normalize 자동 처리
    - Tiling 옵션 (Tiler 통합 가능)
    - 단일 이미지 또는 배치 처리
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_size = config.get("image_size", 256)
        self.crop_size = config.get("crop_size", None)
        self.normalization = config.get("normalization", "imagenet")
        self.tiling = config.get("tiling", None)  # {'tile_size': int, 'stride': int}

        # 정규화 파라미터
        self.mean, self.std = self._get_normalization_params()

        # 변환 파이프라인 생성
        self.transform = self._build_transform()

        # Tiler (옵션)
        self.tiler = None
        if self.tiling:
            from defectvad.components.tiler import Tiler
            self.tiler = Tiler(
                tile_size=self.tiling["tile_size"],
                stride=self.tiling.get("stride", self.tiling["tile_size"]),
                pad_if_needed=True,
            )

    def _get_normalization_params(self) -> Tuple[tuple, tuple]:
        """정규화 파라미터 조회"""
        norm = self.normalization
        if isinstance(norm, dict):
            return tuple(norm["mean"]), tuple(norm["std"])

        if norm == "imagenet":
            return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif norm == "clip":
            return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        elif norm == "none" or norm is None:
            return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        else:
            raise ValueError(f"Unknown normalization: {norm}")

    def _build_transform(self) -> T.Compose:
        """전처리 파이프라인 구성"""
        transform_list = []

        # 1. 리사이즈 또는 리사이즈+센터크롭
        if self.crop_size:
            transform_list.append(T.Resize(max(self.image_size, self.crop_size), interpolation=T.InterpolationMode.BICUBIC))
            transform_list.append(T.CenterCrop(self.crop_size))
        else:
            transform_list.append(T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC))

        # 2. 텐서 변환 및 정규화
        transform_list.append(T.ToTensor())
        if self.normalization != "none":
            transform_list.append(T.Normalize(mean=self.mean, std=self.std))

        return T.Compose(transform_list)

    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor, str, Path]) -> torch.Tensor:
        """
        단일 이미지 전처리 (호출 가능 객체)
        Args:
            image: PIL Image, np.ndarray, 파일 경로, 또는 텐서
        Returns:
            processed_tensor: (C, H, W) 또는 (N, C, H, W) 텐서
        """
        # 입력 통일
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            return self._preprocess_tensor(image)

        # 변환 적용
        tensor = self.transform(image)  # (C, H, W)

        # Tiling 적용 (옵션)
        if self.tiler:
            tensor = self.tiler.tile(tensor)  # (N, C, H, W)

        return tensor

    def _preprocess_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """이미 텐서인 경우 전처리 (배치 감지)"""
        if tensor.dim() == 3:  # (C, H, W)
            tensor = tensor.unsqueeze(0)  # (1, C, H, W)
        elif tensor.dim() != 4:
            raise ValueError(f"Invalid tensor shape: {tensor.shape}, expected (B, C, H, W) or (C, H, W)")

        # 크기 조정 및 정규화
        if tensor.shape[-1] != self.image_size or tensor.shape[-2] != self.image_size:
            tensor = T.functional.resize(tensor, (self.image_size, self.image_size), antialias=True)

        # 정규화
        if self.normalization != "none":
            tensor = T.functional.normalize(tensor, mean=self.mean, std=self.std)

        # Tiling
        if self.tiler:
            b, c, h, w = tensor.shape
            tiles = []
            for i in range(b):
                tile = self.tiler.tile(tensor[i])  # (N, C, H, W)
                tiles.append(tile)
            tensor = torch.cat(tiles, dim=0)  # (B*N, C, H, W)

        return tensor

    def inverse_transform(self, tensor: torch.Tensor) -> Image.Image:
        """
        텐서를 원본 이미지로 복원 (시각화용)
        """
        if tensor.dim() == 3:
            tensor = tensor.clone().detach()
        elif tensor.dim() == 4:
            tensor = tensor[0].clone().detach()  # 첫 번째 샘플
        else:
            raise ValueError(f"Invalid tensor dim: {tensor.dim()}")

        # 역정규화
        if self.normalization != "none":
            inv_mean = [-m / s for m, s in zip(self.mean, self.std)]
            inv_std = [1.0 / s for s in self.std]
            tensor = T.functional.normalize(tensor, mean=inv_mean, std=inv_std)

        tensor = torch.clamp(tensor, 0, 1)
        return TF.to_pil_image(tensor)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Preprocessor':
        """config에서 Preprocessor 생성"""
        return cls(config)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 전처리 설정 반환"""
        return {
            "image_size": 256,
            "crop_size": None,
            "normalization": "imagenet",
            "tiling": None,
        }
