# common/evaluator.py

import torch
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
from torchmetrics.classification import BinaryF1Score, BinaryPrecisionRecallCurve
from torchmetrics import MetricCollection


class Evaluator:
    def __init__(self, model, device=None, pixel_level=False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.pixel_level = pixel_level

        self.image_metrics = MetricCollection({
            "auroc": BinaryAUROC(),
            "aupr": BinaryAveragePrecision(),
            "f1": BinaryF1Score(threshold=0.5),     # default threshold value
        }).to(self.device)

        self.pixel_metrics = MetricCollection({
            "auroc": BinaryAUROC(),
            "aupr": BinaryAveragePrecision(),
            "f1": BinaryF1Score(threshold=0.5),     # default threshold value
        }).to(self.device)

        # thresholds
        self.image_threshold = None
        self.pixel_threshold = None

    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        predictions = self.model(images)
        return {**batch, **predictions}

    def set_pixel_level(self, value=True):
        self.pixel_level = value

    def set_image_threshold(self, threshold):
        self.image_threshold = float(thershold)

    def set_pixel_threshold(self, threshold):
        self.pixel_threshold = float(thershold)

    def __call__(self, dataloader, pixel_level=None, image_threshold=None, pixel_threshold=None):
        return self.evaluate(dataloader, pixel_level, image_threshold, pixel_threshold)

    @torch.no_grad()
    def evaluate(self, dataloader, pixel_level=None, image_threshold=None, pixel_threshold=None):
        pixel_level = pixel_level if pixel_level is not None else self.pixel_level

        self.model.eval()

        all_image_scores, all_image_labels = [], []
        self.image_metrics.reset()

        all_pixel_scores, all_pixel_labels = [], []
        if pixel_level:
            self.pixel_metrics.reset()

        with tqdm(dataloader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f">> Evaluation")
            for batch in progress_bar:
                predictions = self.validation_step(batch)

                image_scores = predictions["pred_score"].cpu().flatten()
                image_labels = predictions["label"].cpu().flatten().long()

                all_image_scores.append(image_scores)
                all_image_labels.append(image_labels)

                if pixel_level:
                    pixel_scores = predictions["anomaly_map"]
                    pixel_labels = predictions["mask"]

                    if pixel_labels is not None:
                        valid = pixel_labels >= 0       # (0: normal, 1: anomaly, -1: invalid)

                        pixel_scores = pixel_scores[valid].cpu().flatten()
                        pixel_labels = pixel_labels[valid].cpu().flatten().long()

                        all_pixel_scores.append(pixel_scores)
                        all_pixel_labels.append(pixel_labels)

        all_image_scores = torch.cat(all_image_scores)
        all_image_labels = torch.cat(all_image_labels)
        self.image_metrics.update(all_image_scores, all_image_labels)

        # Priority: image_threshold > self.image_threshold > _find_f1_threshold
        _threshold = image_threshold if image_threshold is not None else self.image_threshold

        if _threshold is not None:
            f1_metric = BinaryF1Score(threshold=_threshold).to(self.device)
            image_f1 = f1_metric(all_image_scores, all_image_labels).item()
            image_th = float(_threshold)
        else:
            image_f1, image_th = self._find_f1_threshold(all_image_scores, all_image_labels)
            image_f1, image_th = image_f1.item(), image_th.item()
        
        results = {
                "auroc": self.image_metrics["auroc"].compute().item(),
                "aupr": self.image_metrics["aupr"].compute().item(),
                "f1": image_f1,
                "th": image_th,
        }

        if pixel_level and all_pixel_scores and all_pixel_labels:
            all_pixel_scores = torch.cat(all_pixel_scores)
            all_pixel_labels = torch.cat(all_pixel_labels)
            self.pixel_metrics.update(all_pixel_scores, all_pixel_labels)

            # Priority: pixel_threshold > self.pixel_threshold > _find_f1_threshold
            _threshold = pixel_threshold if pixel_threshold is not None else self.pixel_threshold

            if _threshold is not None:
                f1_metric = BinaryF1Score(threshold=_threshold).to(self.device)
                pixel_f1 = f1_metric(all_pixel_scores, all_pixel_labels).item()
                pixel_th = float(_threshold)
            else:
                pixel_f1, pixel_th = self._find_f1_threshold(all_pixel_scores, all_pixel_labels)
                pixel_f1, pixel_th = pixel_f1.item(), pixel_th.item()

            results.update({
                "pix_auroc": self.pixel_metrics["auroc"].compute().item(),
                "pix_aupr": self.pixel_metrics["aupr"].compute().item(),
                "pix_f1": pixel_f1,
                "pix_th": pixel_th,
            })
        return results

    def _find_f1_threshold(self, scores, labels):
        curve = BinaryPrecisionRecallCurve().to(self.device)
        precisions, recalls, thresholds = curve(scores, labels)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        max_idx = torch.argmax(f1_scores)
        return f1_scores[max_idx], thresholds[max_idx]

    @torch.no_grad()
    def calibrate_thresholds(self, dataloader):
        self.model.eval()
        all_scores = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"]

            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue

            normal_images = images[normal_mask]
            predictions = self.model(normal_images)
            pred_scores = predictions["pred_score"].cpu().flatten()

            all_scores.append(pred_scores)

        if len(all_scores) == 0:
            raise ValueError(" > No normal samples found!")

        all_scores = torch.cat(all_scores)
        mean, std = all_scores.mean().item(), all_scores.std().item()

        return {
            "99%": torch.quantile(all_scores, 0.99).item(),
            "97%": torch.quantile(all_scores, 0.97).item(),
            "95%": torch.quantile(all_scores, 0.95).item(),
            "3sig": mean + 3 * std,
            "2sig": mean + 2 * std,
            "1sig": mean + 1 * std,
        }
