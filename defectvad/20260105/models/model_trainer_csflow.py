# src/defectvad/models/csflow/model_trainer.py

import torch
import torch.optim as optim

from defectvad.common.base_model import BaseModel
from defectvad.common.base_trainer import BaseTrainer

from .torch_model import CsFlowModel
from .loss import CsFlowLoss


class Csflow(BaseModel):
    def __init__(self, cross_conv_hidden_channels=1024, n_coupling_blocks=4, clamp=3,
        num_channels=3, input_size=(256, 256)):

        model = CsFlowModel(
            input_size=input_size,
            cross_conv_hidden_channels=cross_conv_hidden_channels,
            n_coupling_blocks=n_coupling_blocks,
            clamp=clamp,
            num_channels=num_channels,
        )
        super().__init__(model)


class CsflowTrainer(BaseTrainer):
    def __init__(self, model):

        if not isinstance(model, Csflow):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        loss_fn = CsFlowLoss()
        super().__init__(model, loss_fn=loss_fn)
        self.model.feature_extractor.eval()

    def configure_optimizers(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=2e-4,
            eps=1e-04,
            weight_decay=1e-5,
            betas=(0.5, 0.9),
        )
        self.gradient_clip_val = 1.0

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        z_dist, jacobians = self.model(images)
        loss = self.loss_fn(z_dist, jacobians)
        return {"loss": loss}
