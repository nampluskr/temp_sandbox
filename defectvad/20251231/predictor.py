# common/predictor.py

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


class Predictor:
    def __init__(self, model, device=None, image_threshold=None, pixel_threshold=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.pixel_level = pixel_level

        self.image_threshold = float(image_threshold) if image_threshold is not None else None
        self.pixel_threshold = float(pixel_threshold) if pixel_threshold is not None else None

    def set_image_threshold(self, threshold):
        self.image_threshold = float(threshold)

    def set_pixel_threshold(self, threshold):
        self.pixel_threshold = float(threshold)

    def __call__(self, dataloader):
        return self.predict(dataloader)

    # TOTO: predict + predict_step 합치기: isinstance(data, DataLoader) or isinstance(data, torch.Tensor)
    @torch.no_grad()
    def predict(self, dataloader):
        self.model.eval()

        outputs = {"image": [], "label": [], "mask": [], "anomaly_map": [], "pred_score": []}

        with tqdm(dataloader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(">> Predicting")

            for batch in progress_bar:
                images = batch["image"].to(self.device)
                predictions = self.model(images)

                outputs["image"].append(batch["image"])                                     # (B, C, H, W)
                outputs["label"].append(batch["label"])                                     # (B,)
                outputs["anomaly_map"].append(predictions["anomaly_map"].cpu())             # (B, H, W)
                outputs["pred_score"].append(predictions["pred_score"].cpu().flatten())     # (B,)

                if batch["mask"] is not None:
                    outputs["mask"].append(batch["mask"])                                   # (B, H, W)
                else:
                    masks = torch.zeros_like(predictions["anomaly_map"]).cpu()
                    outputs["mask"].append(masks)

        if self.image_threshold is not None:
            pred_scores = outputs["pred_score"].flatten()
            outputs["pred_label"] = (pred_scores >= self.image_threshold).long()

        if self.pixel_threshold is not None:
            anomaly_maps = outputs["anomaly_map"]
            outputs["pred_mask"] = (anomaly_maps >= self.pixel_threshold).long()

        for key, value in outputs.items():
            if value and isinstance(value[0], torch.Tensor):
                outputs[key] = torch.cat(value, dim=0)

        return outputs

    @torch.no_grad()
    def predict_step(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)    # (1, C, H, W)

        self.model.eval()
        images = images.to(self.device)
        predictions = self.model(images)

        outputs = {
            "image": images.cpu(),
            "anomaly_map": predictions["anomaly_map"].cpu(),
            "pred_score": predictions["pred_score"].cpu().flatten(),
        }

        if self.image_threshold is not None:
            pred_scores = outputs["pred_score"].flatten()
            outputs["pred_label"] = (pred_scores >= self.image_threshold).long()

        if self.pixel_threshold is not None:
            anomaly_maps = outputs["anomaly_map"]
            outputs["pred_mask"] = (anomaly_maps >= self.pixel_threshold).long()

        return outputs
