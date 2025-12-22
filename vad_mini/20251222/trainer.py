# src/vad_mini/models/stfpm/trainer.py

import torch
import torch.optim as optim

from vad_mini.models.components.base_trainer import BaseTrainer
from .torch_model import STFPMModel
from .loss import STFPMLoss


class STFPMTrainer(BaseTrainer):
    def __init__(self, backbone="resnet50", layers=["layer1", "layer2", "layer3"]):

        model = STFPMModel(backbone=backbone, layers=layers)
        loss_fn = STFPMLoss()
        super().__init__(model, loss_fn)

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        teacher_features, student_features = self.model.forward(images)
        loss = self.loss_fn(teacher_features, student_features)
        return dict(loss=loss)

    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        predictions = self.model(images)
        return predictions

    def configure_optimizers(self):
        return optim.SGD(
            params=self.model.student_model.parameters(),
            lr=0.4,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001,
        )
