# src/defectvad/models/reverse_distillaton/trainer.py

import torch
import torch.optim as optim

from defectvad.common.base_model import BaseModel
from defectvad.common.base_trainer import BaseTrainer

from .torch_model import ReverseDistillationModel
from .loss import ReverseDistillationLoss
from .anomaly_map import AnomalyMapGenerationMode


class ReverseDistillation(BaseModel):
    def __init__(self, backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"],
        input_size=(256, 256), anomaly_map_mode="add"):

        model = ReverseDistillationModel(
            backbone=backbone,
            layers=layers,
            input_size=input_size,
            anomaly_map_mode=anomaly_map_mode,
            pre_trained=True,
        )
        super().__init__(model)


class ReverseDistillationTrainer(BaseTrainer):
    def __init__(self, model):
        if not isinstance(model, ReverseDistillation):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")
        
        loss_fn = ReverseDistillationLoss()
        super().__init__(model, loss_fn=loss_fn)

    def configure_optimizers(self):
        self.optimizer = optim.Adam(
            params=list(self.model.decoder.parameters()) + list(self.model.bottleneck.parameters()),
            lr=0.005,                   # default: 0.005
            betas=(0.5, 0.99),
        )
        self.gradient_clip_val = 1.0    # default: None

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        loss = self.loss_fn(*self.model(images))
        return {"loss": loss}
