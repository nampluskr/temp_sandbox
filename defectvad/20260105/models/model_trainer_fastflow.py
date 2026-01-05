# src/defectvad/models/fastflow/trainer.py

import torch
import torch.optim as optim

from defectvad.common.base_model import BaseModel
from defectvad.common.base_trainer import BaseTrainer

from .torch_model import FastflowModel
from .loss import FastflowLoss


class Fastflow(BaseModel):
    def __init__(self, backbone="wide_resnet50_2",
        flow_steps=8, conv3x3_only=False, hidden_ratio=1.0, input_size=(256, 256)):

        model = FastflowModel(
            input_size=input_size,
            backbone=backbone,
            pre_trained=True,
            flow_steps=flow_steps,
            conv3x3_only=conv3x3_only,
            hidden_ratio=hidden_ratio,
        )
        super().__init__(model)


class FastflowTrainer(BaseTrainer):
    def __init__(self, model):

        if not isinstance(model, Fastflow):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        loss_fn = FastflowLoss()
        super().__init__(model, loss_fn=loss_fn)

    def configure_optimizers(self):
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=0.001,
            weight_decay=0.00001,
        )

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        hidden_variables, jacobians = self.model(images)
        loss = self.loss_fn(hidden_variables, jacobians)
        return {"loss": loss}
