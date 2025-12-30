# src/defectvad/common/visualizer.py

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

import torch
import torchvision.transforms.functional as TF


class Visualizer:
    """
    이상 탐지 결과 시각화 도구
    - 입력 이미지, 이상 맵, 마스크, 예측 점수 등을 하나의 플롯으로 출력
    - 배치 또는 단일 샘플 지원
    - 파일 저장 또는 인라인 플롯 (Jupyter)
    """

    def __init__(self, save_dir=None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot(
        self,
        image: torch.Tensor,
        anomaly_map: torch.Tensor,
        pred_score: float,
        mask: Optional[torch.Tensor] = None,
        label: Optional[int] = None,
        image_path: Optional[str] = None,
        show: bool = True,
        save: bool = False,
        filename: str = None,
        threshold: float = 0.5,
    ):
        """
        단일 샘플 시각화
        """
        # 텐서를 numpy로 변환
        img_np = self._tensor_to_image(image)  # (H, W, 3)
        map_np = self._tensor_to_anomaly_map(anomaly_map)  # (H, W)

        has_mask = mask is not None
        if has_mask:
            mask_np = self._tensor_to_mask(mask)  # (H, W)

        # 제목 생성
        title = f"Pred: {pred_score:.3f}"
        if label is not None:
            title += f" | Label: {'Anomalous' if label else 'Normal'}"
        title += f" | {'ANOMALY' if pred_score >= threshold else 'NORMAL'}"

        # 플롯 구성
        fig, axes = plt.subplots(1, 3 if has_mask else 2, figsize=(12 if has_mask else 9, 4))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # 1. 입력 이미지
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # 2. 이상 맵
        im1 = axes[1].imshow(map_np, cmap="plasma", vmin=0, vmax=np.max(map_np))
        axes[1].set_title(f"Anomaly Map\n(Score: {pred_score:.3f})")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        # 3. 마스크 (있는 경우)
        if has_mask:
            im2 = axes[2].imshow(mask_np, cmap="gray")
            axes[2].set_title("Ground Truth Mask")
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2], shrink=0.8)

        fig.suptitle(title, fontsize=12, y=0.98)
        plt.tight_layout()

        if save and self.save_dir:
            fname = filename or Path(image_path or "unknown").stem
            save_path = self.save_dir / f"{fname}_vis.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f" > Visualization saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_batch(
        self,
        images: torch.Tensor,
        anomaly_maps: torch.Tensor,
        pred_scores: List[float],
        masks: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_paths: Optional[List[str]] = None,
        show: bool = True,
        save: bool = False,
        max_samples: int = 8,
        threshold: float = 0.5,
    ):
        """
        배치 시각화 (최대 max_samples까지)
        """
        n_samples = min(len(images), max_samples)
        cols = 4
        rows = (n_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.5 * rows))
        if rows == 1:
            axes = axes[None, :]  # 2D 인덱스 통일
        elif cols == 1:
            axes = axes[:, None]

        for i in range(n_samples):
            r, c = i // cols, i % cols
            ax = axes[r, c]

            # 데이터 추출
            img = images[i]
            am = anomaly_maps[i]
            score = pred_scores[i]
            mask = masks[i] if masks is not None else None
            label = labels[i].item() if labels is not None else None
            path = image_paths[i] if image_paths else f"sample_{i}"

            # 시각화
            self._plot_to_axis(ax, img, am, score, mask, label, threshold)

        # 빈 축 숨기기
        for i in range(n_samples, rows * cols):
            r, c = i // cols, i % cols
            axes[r, c].axis("off")

        fig.suptitle(f"Anomaly Detection Results (n={n_samples}, threshold={threshold})", fontsize=14, y=0.98)
        plt.tight_layout()

        if save and self.save_dir:
            save_path = self.save_dir / "batch_visualization.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f" > Batch visualization saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def _plot_to_axis(
        self,
        ax,
        image: torch.Tensor,
        anomaly_map: torch.Tensor,
        pred_score: float,
        mask: Optional[torch.Tensor] = None,
        label: Optional[int] = None,
        threshold: float = 0.5,
    ):
        """단일 축에 이미지 + anomaly map 오버레이"""
        img_np = self._tensor_to_image(image)
        map_np = self._tensor_to_anomaly_map(anomaly_map)

        # 오버레이: 입력 이미지에 anomaly map 투명도로 덮기
        ax.imshow(img_np)
        im = ax.imshow(map_np, cmap="jet", alpha=0.5, vmin=0, vmax=np.max(map_np))

        # 제목
        title = f"Score: {pred_score:.3f}"
        if label is not None:
            title += f"\n{'Anom' if label else 'Norm'}"
        title += f"\n{'🚨' if pred_score >= threshold else ' >'}"
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        return im

    @staticmethod
    def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        """(C, H, W) 텐서를 (H, W, 3) numpy 이미지로 변환"""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        img = TF.to_pil_image(tensor.cpu().clamp(0, 1))
        return np.array(img)

    @staticmethod
    def _tensor_to_anomaly_map(tensor: torch.Tensor) -> np.ndarray:
        """(H, W) anomaly map을 numpy로 변환"""
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        return tensor.cpu().numpy()

    @staticmethod
    def _tensor_to_mask(tensor: torch.Tensor) -> np.ndarray:
        """(1, H, W) 또는 (H, W) 마스크를 numpy로 변환"""
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        return tensor.cpu().numpy()

    def save_anomaly_map(self, anomaly_map: torch.Tensor, filepath: Union[str, Path]):
        """이상 맵만 파일로 저장 (numpy 형식)"""
        np.save(filepath, self._tensor_to_anomaly_map(anomaly_map))
        print(f"Anomaly map saved to {filepath}")
