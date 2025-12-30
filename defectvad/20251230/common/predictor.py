# src/defectvad/common/predictor.py

import torch
from typing import Dict, List, Union, Optional
from pathlib import Path


class Predictor:
    """
    'pred_score': (B,) or scalar,
    'anomaly_map': (B, H, W) or (H, W)
    """

    def __init__(self, model: torch.nn.Module, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()

        self.threshold = 0.5

    def load_checkpoint(self, ckpt_path: Union[str, Path], strict: bool = True):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location=self.device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.model.load_state_dict(state_dict, strict=strict)
        print(f" > Loaded checkpoint from {ckpt_path}")

    @torch.no_grad()
    def predict(self, image):
        """
            dict: {
                'anomaly_map': (H, H),
                'pred_score': float,
                'is_anomalous': bool
            }
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)
        image = image.to(self.device)

        outputs = self.model(image)
        anomaly_map = outputs["anomaly_map"][0].cpu()
        pred_score = outputs["pred_score"][0].cpu().item()

        return {
            "anomaly_map": anomaly_map,
            "pred_score": pred_score,
            "is_anomalous": pred_score >= self.threshold,
        }

    @torch.no_grad()
    def predict_batch(self, batch):
        """
            dict: {
                'anomaly_maps': [(H,H), ...],
                'pred_scores': [float, ...],
                'is_anomalous': [bool, ...]
            }
        """
        batch = batch.to(self.device)
        outputs = self.model(batch)

        anomaly_maps = outputs["anomaly_map"].cpu()
        pred_scores = outputs["pred_score"].cpu().numpy().tolist()

        return {
            "anomaly_maps": [am for am in anomaly_maps],
            "pred_scores": pred_scores,
            "is_anomalous": [score >= self.threshold for score in pred_scores],
        }

    def set_threshold(self, threshold: float):
        self.threshold = float(threshold)
        print(f"Threshold set to {self.threshold}")

    def get_threshold(self) -> float:
        return self.threshold

    @torch.no_grad()
    def calibrate_threshold(self, dataloader, percentile=95):
        self.model.eval()
        scores = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            outputs = self.model(images)
            scores.append(outputs["pred_score"].cpu())

        scores = torch.cat(scores)
        threshold = torch.quantile(scores, percentile / 100.0).item()

        self.set_threshold(threshold)
        print(f" > Calibrated threshold ({percentile}%) = {threshold:.4f}")
        return threshold
