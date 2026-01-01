# common/evaluator.py

import torch
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
from torchmetrics.classification import BinaryF1Score, BinaryPrecisionRecallCurve
from torchmetrics import MetricCollection


class Evaluator:
    def __init__(self, model, device=None, pixel_level=False, image_threshold=None, pixel_threshold=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.pixel_level = pixel_level
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold

        self.image_metrics = MetricCollection({
            "auroc": BinaryAUROC(),
            "aupr": BinaryAveragePrecision(),
        }).to(self.device)

        self.pixel_metrics = MetricCollection({
            "auroc": BinaryAUROC(),
            "aupr": BinaryAveragePrecision(),
        }).to(self.device)

    def set_pixel_level(self, value=True):
        self.pixel_level = value

    def set_image_threshold(self, threshold):
        self.image_threshold = float(threshold)

    def set_pixel_threshold(self, threshold):
        self.pixel_threshold = float(threshold)

    def __call__(self, dataloader, pixel_level=False, image_threshold=None, pixel_threshold=None):
        return self.evaluate(dataloader, pixel_level, image_threshold, pixel_threshold)

    @torch.no_grad()
    def evaluate(self, dataloader, pixel_level=False, image_threshold=None, pixel_threshold=None):
        self.model.eval()
        pixel_level = pixel_level or self.pixel_level
        image_threshold = image_threshold or self.image_threshold
        pixel_threshold = pixel_threshold or self.pixel_threshold

        self.image_metrics.reset()
        self.pixel_metrics.reset()

        image_scores, image_labels = [], []
        pixel_scores, pixel_labels = [], []

        with tqdm(dataloader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(" > Evaluation")

            for batch in progress_bar:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device).long()
                preds = self.model(images)  # pred_score, anomaly_map

                # image-level
                batch_image_scores = preds["pred_score"].flatten()
                batch_image_labels = labels.flatten()

                image_scores.append(batch_image_scores.cpu())
                image_labels.append(batch_image_labels.cpu())
                self.image_metrics.update(batch_image_scores, batch_image_labels)

                # pixel-level
                if pixel_level and "anomaly_map" in preds and "mask" in batch:
                    batch_pixel_scores = preds["anomaly_map"].squeeze(1)
                    batch_pixel_labels = batch["mask"].to(self.device).squeeze(1)

                    valid_mask = batch_pixel_labels >= 0
                    batch_pixel_scores = batch_pixel_scores[valid_mask].flatten()
                    batch_pixel_labels = batch_pixel_labels[valid_mask].flatten().long()

                    pixel_scores.append(batch_pixel_scores.cpu())
                    pixel_labels.append(batch_pixel_labels.cpu())
                    self.pixel_metrics.update(batch_pixel_scores, batch_pixel_labels)

        # image-level metrics
        image_scores = torch.cat(image_scores)
        image_labels = torch.cat(image_labels)
        _image_f1_scores, _image_threshold = self.resolve_f1_threshold(
            image_scores, image_labels, image_threshold)
        _image_metrics = self.image_metrics.compute()

        results = {
            "img_auroc": _image_metrics["auroc"].item(),
            "img_aupr": _image_metrics["aupr"].item(),
            "img_f1": _image_f1_scores.item(),
            "img_th": _image_threshold.item(),
        }

        # pixel-level metrics
        if pixel_level and pixel_scores:
            pixel_scores = torch.cat(pixel_scores)
            pixel_labels = torch.cat(pixel_labels)
            _pixel_f1_scores, _pixel_threshold = self.resolve_f1_threshold(
                pixel_scores, pixel_labels, pixel_threshold)
            _pixel_metrics = self.pixel_metrics.compute()

            results.update({
                "pix_auroc": _pixel_metrics["auroc"].item(),
                "pix_aupr": _pixel_metrics["aupr"].item(),
                "pix_f1": _pixel_f1_scores.item(),
                "pix_th": _pixel_threshold.item(),
            })

        return results

    def resolve_f1_threshold(self, scores, labels, threshold):
        if threshold is not None:
            f1_metric = BinaryF1Score(threshold=threshold).to(self.device)
            return f1_metric(scores, labels), torch.tensor(threshold).float()
        else:
            fi_curve = BinaryPrecisionRecallCurve().to(self.device)
            precisions, recalls, thresholds = fi_curve(scores, labels)
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
            max_idx = torch.argmax(f1_scores)
            return f1_scores[max_idx], thresholds[max_idx]

    @torch.no_grad()
    def calibrate_image_thresholds(self, dataloader):
        self.model.eval()

        scores = []
        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"]

            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue

            images = images[normal_mask]
            preds = self.model(images)
            batch_scores = preds["pred_score"].flatten()
            scores.append(batch_scores.cpu())

        if len(scores) == 0:
            raise ValueError(" > No normal samples found!")

        scores = torch.cat(scores)
        mean, std = scores.mean().item(), scores.std().item()

        return {
            "99%": torch.quantile(scores, 0.99).item(),
            "97%": torch.quantile(scores, 0.97).item(),
            "95%": torch.quantile(scores, 0.95).item(),
            "3sig": mean + 3 * std,
            "2sig": mean + 2 * std,
            "1sig": mean + 1 * std,
        }

    @torch.no_grad()
    def calibrate_pixel_thresholds(self, dataloader, topk=20):
        self.model.eval()

        scores = []
        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"]

            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue

            images = images[normal_mask]
            preds = self.model(images)

            if "anomaly_map" not in preds or "mask" not in batch:
                continue

            pixel_scores = preds["anomaly_map"][normal_mask]
            pixel_labels = batch["mask"][normal_mask].to(self.device)

            pixel_scores = pixel_scores.squeeze(1)
            pixel_labels = pixel_labels.squeeze(1)

            for _scores, _labels in zip(pixel_scores, pixel_labels):
                _scores = _scores[_labels >= 0]

                if _scores.numel() > topk:
                    topk_valuess, _ = torch.topk(_scores, k=topk, largest=True)
                else:
                    topk_valuess = _scores

                scores.append(topk_valuess.mean().cpu())

        if len(scores) == 0:
            raise ValueError(" > No valid pixel-level normal samples found!")

        scores = torch.stack(scores)
        mean, std = scores.mean().item(), scores.std().item()

        return {
            "99%": torch.quantile(scores, 0.99).item(),
            "97%": torch.quantile(scores, 0.97).item(),
            "95%": torch.quantile(scores, 0.95).item(),
            "3sig": mean + 3 * std,
            "2sig": mean + 2 * std,
            "1sig": mean + 1 * std,
        }
