# src/defectvad/models/stfpm/trainer.py

import torch
import torch.optim as optim

from defectvad.common.base_model import BaseModel
from defectvad.common.base_trainer import BaseTrainer

from .torch_model import STFPMModel
from .loss import STFPMLoss


class STFPM(BaseModel):
    def __init__(self, backbone="resnet50", layers=["layer1", "layer2", "layer3"]):
        model = STFPMModel(
            backbone=backbone,
            layers=layers
        )
        super().__init__(model)


class STFPMTrainer(BaseTrainer):
    def __init__(self, model):
        if not isinstance(model, STFPM):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        loss_fn = STFPMLoss()
        super().__init__(model, loss_fn=loss_fn)

    def configure_optimizers(self):
        self.optimizer = optim.SGD(
            params=self.model.student_model.parameters(),
            lr=0.4,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001,
        )
        self.gradient_clip_val = 1.0

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        teacher_features, student_features = self.model.forward(images)
        loss = self.loss_fn(teacher_features, student_features)
        return {"loss": loss}
