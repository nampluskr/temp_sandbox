# common/base_trainer.py

from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
from torch.nn.utils import clip_grad_norm_
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.functional.classification import binary_roc

from .early_stopper import EarlyStopper


class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, torch.nn.Module) else loss_fn

        self.train_loader = None
        self.valid_loader = None

        # configure optimizers
        self.optimizer = None
        self.scheduler = None
        self.gradient_clip_val = None

        # configure early stoppers
        self.train_early_stopper = None
        self.valid_early_stopper = None

        # training epochs and steps
        self.global_epoch = 0
        self.global_step = 0
        self.max_epochs = 1
        self.max_steps = 1

        # validation metrics
        self.aucroc = BinaryAUROC().to(self.device)
        self.aupr = BinaryAveragePrecision().to(self.device)

    #######################################################
    # setup for anomaly detection models
    #######################################################

    def configure_optimizers(self):
        pass

    def configure_early_stoppers(self):
        self.train_early_stopper = None
        self.valid_early_stopper = EarlyStopper(patience=5, min_delta=1e-3, monitor="auroc")

    @abstractmethod
    def training_step(self, batch):
        raise NotImplementedError
    
    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        predictions = self.model(images)
        return {**batch, **predictions}
    
    #######################################################
    # fit: train model for max_epochs or max_steps
    #######################################################

    def fit(self, train_loader, max_epochs=1, valid_loader=None):
        self.max_epochs = max_epochs
        self.max_steps = max_epochs * len(train_loader)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.configure_optimizers()
        self.configure_early_stoppers()

        self.on_fit_start()
        self.on_train_start()

        for _ in range(self.max_epochs):
            self.on_train_epoch_start()
            train_outputs = self.train(self.train_loader)
            self.on_train_epoch_end(train_outputs)

            if self.valid_loader is not None:
                self.on_validation_epoch_start()
                valid_outputs = self.validate(self.valid_loader)
                self.on_validation_epoch_end(valid_outputs)

            if self.train_early_stop or self.valid_early_stop:
                break

        self.on_train_end()
        self.on_fit_end()

    #######################################################
    # Hooks
    #######################################################

    def backward(self, loss):
        loss.backward()

    def on_fit_start(self): pass

    def on_train_start(self):
        self.early_stop_str = ""
        self.train_early_stop = False
        self.valid_early_stop = False
        self.current_epoch = 0
        self.current_step = 0
        print("\n*** Training start...")

    def on_train_epoch_start(self):
        self.global_epoch += 1
        self.current_epoch += 1

    def on_train_batch_start(self, batch, batch_idx): pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.global_step += 1
        self.current_step += 1

    def on_train_epoch_end(self, outputs):
        self.epoch_info = f"[{self.current_epoch:3d}/{self.max_epochs}]"
        self.train_info = ", ".join([f"{k}:{v:.3f}" for k, v in outputs.items()])
        if self.valid_loader is None:
            print(f"{self.epoch_info} {self.train_info}")

        if self.train_early_stopper is not None:
            metric_name = self.train_early_stopper.monitor
            self.train_early_stop = self.train_early_stopper.step(outputs[metric_name])

            if self.train_early_stopper.target_reached:
                self.early_stop_str += f"Training target readched! {self.train_early_stopper.get_info()}"
            elif self.train_early_stopper.early_stop:
                self.early_stop_str += f"Training Early Stopped! {self.train_early_stopper.get_info()}"

        # if self.max_steps is not None:
        #     if self.current_step >= self.max_steps:
        #         self.train_early_stop = True
        #         self.early_stop_str += f"Max training step reached! {self.current_step} steps"

    def on_validation_epoch_start(self): pass

    def on_validation_batch_start(self, batch, batch_idx): pass

    def on_validation_batch_end(self, outputs, batch, batch_idx): pass

    def on_validation_epoch_end(self, outputs):
        valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in outputs.items()])
        print(f"{self.epoch_info} {self.train_info} | (val) {valid_info}")

        if self.valid_early_stopper is not None:
            metric_name = self.valid_early_stopper.monitor
            self.valid_early_stop = self.valid_early_stopper.step(outputs[metric_name])

            if self.valid_early_stopper.target_reached:
                self.early_stop_str += f"Validation target readched! {self.valid_early_stopper.get_info()}"
            elif self.valid_early_stopper.early_stop:
                self.early_stop_str += f"Validation Early Stopped! {self.valid_early_stopper.get_info()}"

    def on_train_end(self):
        if self.train_early_stop or self.valid_early_stop:
            print(f">> {self.early_stop_str}")
        print("*** Training completed!")

    def on_fit_end(self): pass

    #######################################################
    # Train one epoch
    #######################################################

    @torch.enable_grad()
    def train(self, train_loader):
        self.model.train()
        outputs = {}
        num_images = 0

        with tqdm(train_loader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f">> Training")
            for batch_idx, batch in enumerate(progress_bar):
                self.on_train_batch_start(batch, batch_idx)

                batch_size = batch["image"].shape[0]
                num_images += batch_size
                batch_outputs = self.training_step(batch)

                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    loss = batch_outputs["loss"]
                    self.backward(loss)

                    if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                        clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_val)

                    self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step()

                for name, value in batch_outputs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    outputs.setdefault(name, 0.0)
                    outputs[name] += value * batch_size

                progress_bar.set_postfix({name: f"{value / num_images:.3f}" for name, value in outputs.items()})
                self.on_train_batch_end(batch_outputs, batch, batch_idx)

        return {name: value / num_images for name, value in outputs.items()}

    #######################################################
    # Validate one epoch
    #######################################################

    @torch.no_grad()
    def validate(self, valid_loader):
        self.model.eval()
        all_pred_scores = []
        all_labels = []

        self.aucroc.reset()
        self.aupr.reset()

        with tqdm(valid_loader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f">> Validation")
            for batch_idx, batch in enumerate(progress_bar):
                self.on_validation_batch_start(batch, batch_idx)

                predictions = self.validation_step(batch)
                pred_scores = predictions["pred_score"].flatten()
                labels = predictions["label"]

                all_pred_scores.append(pred_scores.cpu())
                all_labels.append(labels.cpu())

                self.on_validation_batch_end(predictions, batch, batch_idx)

        all_pred_scores = torch.cat(all_pred_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        self.aucroc.update(all_pred_scores, all_labels)
        self.aupr.update(all_pred_scores, all_labels)

        fpr, tpr, thresholds = binary_roc(all_pred_scores, all_labels)
        j_scores = tpr - fpr

        return {
            "auroc": self.aucroc.compute().item(),
            "aupr": self.aupr.compute().item(),
            "threshold": thresholds[torch.argmax(j_scores)].item()
        }

    @torch.no_grad()
    def calibrate_threshold(self, dataloader):
        self.model.eval()
        all_scores = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            predictions = self.model(images)
            pred_scores = predictions["pred_score"].flatten()
            all_scores.append(pred_scores.cpu())

        all_scores = torch.cat(all_scores)
        mean, std = all_scores.mean(), all_scores.std()

        return {
            "99%": torch.quantile(all_scores, 0.99).item(),
            "97%": torch.quantile(all_scores, 0.97).item(),
            "95%": torch.quantile(all_scores, 0.95).item(),
            "3-sigma": (mean + 3 * std).item(),
            "2-sigma": (mean + 2 * std).item(),
            "1-sigma": (mean + 1 * std).item(),
        }
