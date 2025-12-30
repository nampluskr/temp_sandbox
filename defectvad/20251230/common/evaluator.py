# defectvad/common/evaluator.py

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
from torchmetrics.classification import BinaryPrecisionRecallCurve


class Evaluator:
    def __init__(self, model, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.image_metrics = MetricCollection({
            "image_auroc": BinaryAUROC(),
            "image_aupr": BinaryAveragePrecision(),
            "img_f1": BinaryF1Score(threshold=0.5),
        }).to(self.device)

        self.pixel_metrics = MetricCollection({
            "pix_auroc": BinaryAUROC(),
            "pix_aupr": BinaryAveragePrecision(),
            "pix_f1": BinaryF1Score(search_threshold=False),
        }).to(self.device)

    def setup_output_dir(self, dataset_name, category_name, model_name):
        project_dir = os.environ["PROJECT_DIR"]
        self.output_dir = os.path.join(project_dir, "outputs", dataset_name, category_name, model_name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f" > Evaluation results will be saved to: {self.output_dir}")

    def evaluate(self, test_loader, dataset_name, category_name, model_name, pixel_level=True):
        self.model.eval()
        self.setup_output_dir(dataset_name, category_name, model_name)

        image_scores, image_labels, pixel_scores, pixel_labels = self._forward_all(test_loader)
        image_metrics = self._compute_image_metrics(image_scores, image_labels)

        pixel_metrics = {}
        if pixel_level and pixel_scores is not None:
            pixel_metrics = self._compute_pixel_metrics(pixel_scores, pixel_labels)

        metrics = {**image_metrics, **pixel_metrics}
        # self._save_results(metrics, image_scores, image_labels, pixel_scores, pixel_labels)
        return metrics

    @torch.no_grad()
    def _forward_all(self, test_loader):
        all_image_scores, all_image_labels = [], []
        all_pixel_scores, all_pixel_labels = [], []

        for batch in test_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].cpu()
            masks = batch.get("mask")

            outputs = self.model(images)
            pred_scores = outputs["pred_score"].cpu()
            anomaly_maps = outputs["anomaly_map"].cpu()

            all_image_scores.append(pred_scores)
            all_image_labels.append(labels)

            if masks is not None:
                valid_mask = masks.cpu() >= 0
                pixel_labels = masks[valid_mask].flatten()
                pixel_scores = anomaly_maps[valid_mask].flatten()
                all_pixel_scores.append(pixel_scores)
                all_pixel_labels.append(pixel_labels)

        image_scores = torch.cat(all_image_scores)
        image_labels = torch.cat(all_image_labels)
        pixel_scores = torch.cat(all_pixel_scores) if all_pixel_scores else None
        pixel_labels = torch.cat(all_pixel_labels) if all_pixel_labels else None

        return image_scores, image_labels, pixel_scores, pixel_labels

    @torch.no_grad()
    def _compute_image_metrics(self, image_scores, image_labels):
        image_metrics = self.image_metrics(image_scores, image_labels)
        image_auroc = image_metrics["img_auroc"].item()
        image_aupr = image_metrics["img_aupr"].item()

        image_th, image_f1 = self._find_f1_threshold(image_scores, image_labels)
        return {
            "img_auroc": round(image_auroc, 4),
            "img_aupr": round(image_aupr, 4),
            "img_f1": round(image_f1, 4),
            "img_th": round(image_th, 4),
        }

    @torch.no_grad()
    def _compute_pixel_metrics(self, pixel_scores, pixel_labels):
        pixel_metrics = self.pixel_metrics(pixel_scores, pixel_labels)
        pixel_auroc = pixel_metrics["pix_auroc"].item()
        pixel_aupr = pixel_metrics["pix_aupr"].item()

        pixel_th, pixel_f1 = self._find_f1_threshold(pixel_scores, pixel_labels)
        return {
            "pix_auroc": round(pixel_auroc, 4),
            "pix_aupr": round(pixel_aupr, 4),
            "pix_f1": round(pixel_f1, 4),
            "pix_th": round(pixel_th, 4),
        }

    def _find_f1_threshold(self, scores, labels, num_thresholds=100):
        curve = BinaryPrecisionRecallCurve(num_thresholds=num_thresholds).to(self.device)
        precisions, recalls, thresholds = curve(scores, labels)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        max_idx = torch.argmax(f1_scores)
        return thresholds[max_idx].item(), f1_scores[max_idx].item()

    def _save_results(self, metrics: Dict[str, float], pred_scores, labels, masks, paths):
        # 1. 메트릭 저장
        logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # 2. 예측 결과 저장
        preds_dir = os.path.join(self.output_dir, "predictions")
        os.makedirs(preds_dir, exist_ok=True)
        results = {
            "pred_scores": pred_scores.numpy(),
            "labels": labels.numpy(),
            "masks": masks.numpy() if masks is not None else None,
            "image_paths": paths,
        }
        with open(os.path.join(preds_dir, "test_results.pkl"), "wb") as f:
            pickle.dump(results, f)

        print(f" > Test results saved to {self.output_dir}")
