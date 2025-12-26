# src/vad_mini/models/anomaly_dino/trainer.py

import torch
import torch.optim as optim

from vad_mini.models._components.base_trainer import BaseTrainer
from .torch_model import AnomalyDINOModel


class AnomalyDINOTrainer(BaseTrainer):
    def __init__(self, encoder_name="dinov2_vit_small_14", num_neighbours=1,
        masking=False, coreset_subsampling=False, sampling_ratio=0.1):

        model = AnomalyDINOModel(
            num_neighbours=num_neighbours,
            encoder_name=encoder_name,
            masking=masking,
            coreset_subsampling=coreset_subsampling,
            sampling_ratio=sampling_ratio,
        )
        super().__init__(model, loss_fn=None)

        # trainer_arguments
        self.max_epochs = 1
        self.gradient_clip_val = 0
        self.num_sanity_val_steps = 0

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        _ = self.model(images)
        return {"loss": torch.tensor(0.0, requires_grad=True, device=self.device)}

    def on_train_epoch_end(self, outputs):
        super().on_train_epoch_end(outputs)
        self.model.fit()
