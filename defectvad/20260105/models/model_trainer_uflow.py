# src/defectvad/models/uflow/model_trainer.py

import torch
import torch.optim as optim

from defectvad.common.base_model import BaseModel
from defectvad.common.base_trainer import BaseTrainer

from .torch_model import UflowModel
from .loss import UFlowLoss


class Uflow(BaseModel):
    def __init__(self, backbone="mcait", flow_steps=4, affine_clamp=2.0,
        affine_subnet_channels_ratio=1.0, permute_soft=False, input_size=(448, 448)):

        model = UflowModel(
            input_size=input_size,
            backbone=backbone,
            flow_steps=flow_steps,
            affine_clamp=affine_clamp,
            affine_subnet_channels_ratio=affine_subnet_channels_ratio,
            permute_soft=permute_soft,
        )
        super().__init__(model)


class UflowTrainer(BaseTrainer):
    def __init__(self, model):

        if not isinstance(model, Uflow):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        loss_fn = UFlowLoss()
        super().__init__(model, loss_fn=loss_fn)

    def configure_optimizers(self):
        self.optimizer = optim.Adam([
            {"params": self.model.parameters(), "initial_lr": 1e-3}],
            lr=1e-3,
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.4,
            total_iters=25000,
        )
        self.gradient_clip_val = None

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        z, ljd = self.model(images)
        loss = self.loss_fn(z, ljd)
        return {"loss": loss}
