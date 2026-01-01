# common/predictor.py

import os
from tqdm import tqdm
import torch


class Predictor:
    def __init__(self, model, device=None, image_threshold=None, pixel_threshold=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold

    def set_image_threshold(self, threshold):
        self.image_threshold = threshold

    def set_pixel_threshold(self, threshold):
        self.pixel_threshold = threshold

    def __call__(self, dataloader):
        return self.predict(dataloader)

    @torch.no_grad()
    def predict_step(self, batch):
        images = batch["image"].to(self.device)
        preds = self.model(images)
        outputs = {**batch, **preds}

        if self.image_threshold is not None:
            outputs["pred_label"] = (preds["pred_score"] >= self.image_threshold).long()

        if self.pixel_threshold is not None:
            outputs["pred_mask"] = (preds["anomaly_map"] >= self.pixel_threshold).long()

        return outputs

    @torch.no_grad()
    def predict(self, dataloader):
        self.model.eval()

        outputs = {}
        with tqdm(dataloader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(">> Predicting")

            for batch in progress_bar:
                batch_outputs = self.predict_step(batch)

                for key, value in batch_outputs.items():
                    if torch.is_tensor(value):
                        outputs.setdefault(key, []).append(value.cpu())

        for key, value in outputs.items():
            outputs[key] = torch.cat(value, dim=0)

        return outputs


# helper functions
def save_predictions(outputs, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(outputs, save_path)


def load_predictions(load_path, map_location="cpu"):
    outputs = torch.load(load_path, map_location=map_location)
    return outputs
