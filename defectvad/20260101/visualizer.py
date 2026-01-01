# common/predictor.py

import torch
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, denormalize=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), cmap="jet", alpha=0.5):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.cmap = cmap
        self.alpha = alpha
        self.denormalize = denormalize

    def denormalize_imagenet(self, images):
        mean = self.mean.to(images.device)
        std = self.std.to(images.device)
        images = images * std + mean
        return images.clamp(0, 1)

    def overlay(self, image: np.ndarray, anomaly_map: np.ndarray):
        if anomaly_map.ndim == 3 and anomaly_map.shape[0] == 1:
            anomaly_map = anomaly_map.squeeze(0)

        if anomaly_map.ndim != 2:
            raise ValueError(f"Invalid anomaly_map shape: {anomaly_map.shape}")

        heatmap = plt.get_cmap(self.cmap)(anomaly_map)[..., :3]
        overlay = (1 - self.alpha) * image + self.alpha * heatmap
        return np.clip(overlay, 0, 1)

    def __call__(self, preds, indices=None, max_samples=4, show_gt=True):
        self.visualize(preds, indices, max_samples, show_gt)

    def visualize(self, preds, indices=None, max_samples=4, show_gt=True):
        images = preds["image"]
        labels = preds["label"]
        anomaly_maps = preds["anomaly_map"]
        gt_masks = preds.get("mask", None)
        pred_masks = preds.get("pred_mask", None)
        pred_labels = preds.get("pred_label", None)

        N = images.size(0)
        if indices is None:
            indices = list(range(min(N, max_samples)))

        for idx in indices:
            self._visualize_image(
                image=images[idx],
                label=labels[idx],
                anomaly_map=anomaly_maps[idx],
                gt_mask=None if gt_masks is None else gt_masks[idx],
                pred_mask=None if pred_masks is None else pred_masks[idx],
                pred_label=None if pred_labels is None else pred_labels[idx],
                show_gt=show_gt,
            )

    def _visualize_image(self, image, label, anomaly_map, 
                         gt_mask=None, pred_mask=None, pred_label=None, show_gt=True):
        img = self.to_numpy_image(image, denormalize=self.denormalize)
        amap = self.normalize_anomaly_map(anomaly_map)
        overlay = self.overlay(img, amap)

        gt = None
        if show_gt and gt_mask is not None:
            gt = self.to_numpy_mask(gt_mask)
            if gt.sum() == 0:
                gt = None

        pm = None
        if pred_mask is not None:
            pm = self.to_numpy_mask(pred_mask)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 1. Image
        title = "Image"
        if pred_label is not None:
            title += f" GT={int(label)} | Pred={int(pred_label)}"

        axes[0].imshow(img)
        axes[0].set_title(title)
        axes[0].axis("off")

        # 2. Anomaly Overlay
        axes[1].imshow(overlay)
        axes[1].set_title("Anomaly Overlay")
        axes[1].axis("off")

        # 3. GT Mask
        if gt is not None:
            axes[2].imshow(gt, cmap="gray")
            axes[2].set_title("GT Mask")
        else:
            axes[2].set_axis_off()

        # 4. Pred Mask
        if pm is not None:
            axes[3].imshow(pm, cmap="gray", vmin=0, vmax=1)
            axes[3].set_title("Pred Mask")
            axes[3].axis("off")
        else:
            axes[3].set_axis_off()

        fig.tight_layout()
        plt.show()

    def normalize_anomaly_map(self, amap: torch.Tensor) -> np.ndarray:
        min_v = amap.min()
        max_v = amap.max()
        if max_v > min_v:
            amap = (amap - min_v) / (max_v - min_v)
        else:
            amap = torch.zeros_like(amap)
        return amap.cpu().numpy()

    def to_numpy_image(self, image, denormalize=True):
        if not torch.is_tensor(image):
            raise TypeError(f"Expected torch.Tensor, got {type(image)}")

        if image.dim() == 4 and image.shape[0] == 1:
            image = image.squeeze(0)

        if image.dim() != 3:
            raise ValueError(f"Invalid image shape: {tuple(image.shape)}")

        if denormalize:
            image = self.denormalize_imagenet(image.unsqueeze(0)).squeeze(0)

        image = image.permute(1, 2, 0)

        if image.shape[-1] != 3:
            raise ValueError(f"Expected 3-channel image, got {image.shape}")

        return image.cpu().numpy()


    def to_numpy_mask(self, mask):
        if not torch.is_tensor(mask):
            raise TypeError(f"Expected torch.Tensor, got {type(mask)}")

        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        assert mask.dim() == 2, f"Invalid mask shape: {mask.shape}"
        return mask.cpu().numpy()
    
    


