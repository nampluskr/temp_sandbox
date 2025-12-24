import torch
import torch.optim as optim

from vad_mini.models.components.base_trainer import BaseTrainer, EarlyStopper
from .torch_model import SupersimplenetModel
from .loss import SSNLoss


class SupersimplenetTrainer(BaseTrainer):
    def __init__(self, perlin_threshold=0.2, backbone="wide_resnet50_2.tv_in1k",
        layers=["layer2", "layer3"], adapt_cls_features=False, supervised=False):

        self.supervised = supervised
        if supervised:
            stop_grad = False
            self.norm_clip_val = 1
        else:
            stop_grad = True
            self.norm_clip_val = 0

        model = SupersimplenetModel(
            perlin_threshold=perlin_threshold,
            backbone=backbone,
            layers=layers,
            stop_grad=stop_grad,
            adapt_cls_features=adapt_cls_features,
        )
        loss_fn = SSNLoss()
        super().__init__(model, loss_fn=loss_fn)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW([
            {"params": self.model.adaptor.parameters(), "lr": 0.001},
            {"params": self.model.segdec.parameters(), "lr": 0.002, "weight_decay": 0.00001},
        ])
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(self.max_epochs * 0.8), int(self.max_epochs * 0.9)],
            gamma=0.4,
        )

    def configure_early_stoppers(self):
        self.train_early_stopper = None
        self.valid_early_stopper = None

    def training_step(self, batch):
        anomaly_map, anomaly_score, masks, labels = self.model(
            images=batch["image"].to(self.device),
            # masks=batch["mask"].unsqueeze(dim=1).to(self.device),
            masks=None,
            labels=batch["label"].to(self.device),
        )
        loss = self.loss_fn(pred_map=anomaly_map, pred_score=anomaly_score, target_mask=masks, target_label=labels)
        return {"loss": loss}

    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        predictions = self.model(images)
        return {**batch, **predictions}
