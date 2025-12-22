# src/vad_mini/models/components/base_trainer.py

from abc import ABC, abstractmethod
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.functional.classification import binary_roc

import torch


class BaseTrainer(ABC):
    def __init__(self, model, loss_fn, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, torch.nn.Module) else loss_fn
        self.optimizer = self.configure_optimizers()

        self.global_epoch = 0
        self.global_step = 0

        self.aucroc = BinaryAUROC().to(self.device)
        self.aupr = BinaryAveragePrecision().to(self.device)

    def fit(self, train_loader, num_epochs, valid_loader=None):
        self.on_fit_start(valid_loader)

        for epoch in range(1, num_epochs + 1):

            self.on_train_start(epoch, num_epochs)
            train_outputs = self.train(train_loader)
            self.on_train_end(train_outputs)

            if valid_loader is not None:
                self.on_validation_start()
                valid_outputs = self.evaluate(valid_loader)
                self.on_validation_end(valid_outputs)

        self.on_fit_end()

    #######################################################
    # Hooks
    #######################################################

    def on_fit_start(self, valid_loader):
        self.has_valid_loader = valid_loader is not None

    def on_train_start(self, epoch, num_epochs):
        self.global_epoch += 1
        self.epoch_info = f"[{epoch:3d}/{num_epochs}]"

    @abstractmethod
    def training_step(self, batch):
        raise NotImplementedError

    def on_train_end(self, outputs):
        self.train_info = ", ".join([f"{k}:{v:.3f}" for k, v in outputs.items()])
        if not self.has_valid_loader:
            print(f"{self.epoch_info} {self.train_info}")

    def on_train_batch_start(self, batch, batch_idx):
        self.global_step += 1

    def on_train_batch_end(self, outputs, batch, batch_idx): pass

    def on_validation_start(self): pass

    def on_validation_end(self, outputs):
        valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in outputs.items()])
        print(f"{self.epoch_info} {self.train_info} | (val) {valid_info}")

    def on_validation_batch_start(self, batch, batch_idx): pass

    @abstractmethod
    def validation_step(self, batch):
        raise NotImplementedError

    def on_validation_batch_end(self, outputs, batch, batch_idx): pass

    def on_fit_end(self): pass

    #######################################################
    # Train one epoch
    #######################################################

    @torch.enable_grad()
    def train(self, dataloader):
        self.model.train()
        outputs = {}
        num_images = 0

        with tqdm(dataloader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f">> Training")
            for batch_idx, batch in enumerate(progress_bar):
                self.on_train_batch_start(batch_idx, batch)

                batch_size = batch["image"].shape[0]
                num_images += batch_size

                self.optimizer.zero_grad()
                batch_outputs = self.training_step(batch)
                loss = batch_outputs["loss"]
                loss.backward()
                self.optimizer.step()

                for name, value in batch_outputs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    outputs.setdefault(name, 0.0)
                    outputs[name] += value * batch_size

                progress_bar.set_postfix({
                    name: f"{value / num_images:.3f}"
                    for name, value in outputs.items()
                })
                self.on_train_batch_end(batch_outputs, batch_idx, batch)

        return {name: value / num_images for name, value in outputs.items()}

    #######################################################
    # Validate one epoch
    #######################################################

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        self.aucroc.reset()
        self.aupr.reset()

        all_pred_scores = []
        all_labels = []

        with tqdm(dataloader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f">> Validation")
            for batch in progress_bar:
                images = batch["image"].to(self.device)
                labels = batch["label"]

                predictions = self.model(images)
                pred_scores = predictions["pred_score"].squeeze()

                all_pred_scores.append(pred_scores.cpu())
                all_labels.append(labels)

        all_pred_scores = torch.cat(all_pred_scores)
        all_labels = torch.cat(all_labels)

        self.aucroc.update(all_pred_scores, all_labels)
        self.aupr.update(all_pred_scores, all_labels)

        fpr, tpr, thresholds = binary_roc(all_pred_scores, all_labels)
        j_scores = tpr - fpr

        return {
            "aucroc": self.aucroc.compute().item(),
            "aupr": self.aupr.compute().item(),
            "threshold": thresholds[torch.argmax(j_scores)].item()
        }

    @torch.no_grad()
    def calibrate_threshold(self, dataloader):
        self.model.eval()
        all_scores = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            outputs = self.model(images)
            pred_scores = outputs["pred_score"].squeeze().cpu()
            all_scores.append(pred_scores)

        all_scores = torch.cat(all_scores)

        thresholds = {}
        thresholds["99%"] = torch.quantile(all_scores, 0.99).item()
        thresholds["97%"] = torch.quantile(all_scores, 0.97).item()
        thresholds["95%"] = torch.quantile(all_scores, 0.95).item()
        thresholds["3-sigma"] = (all_scores.mean() + 3 * all_scores.std()).item()
        thresholds["2-sigma"] = (all_scores.mean() + 2 * all_scores.std()).item()
        thresholds["1-sigma"] = (all_scores.mean() + 1 * all_scores.std()).item()

        return thresholds
