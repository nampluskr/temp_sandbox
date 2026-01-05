# src/defectvad/models/stfpm/trainer.py

import logging
from typing import Any

import torch
import torch.optim as optim
from torch.nn.init import trunc_normal_
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from defectvad.common.base_model import BaseModel
from defectvad.common.base_trainer import BaseTrainer

from .torch_model import DinomalyModel
from .optimizer import StableAdamW, WarmCosineScheduler


# Training constants
DEFAULT_IMAGE_SIZE = 448
DEFAULT_CROP_SIZE = 392
MAX_STEPS_DEFAULT = 5000

# Default Training hyperparameters
TRAINING_CONFIG: dict[str, Any] = {
    "optimizer": {
        "lr": 2e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
        "amsgrad": True,
        "eps": 1e-8,
    },
    "scheduler": {
        "base_value": 2e-3,
        "final_value": 2e-4,
        "total_iters": MAX_STEPS_DEFAULT,
        "warmup_iters": 100,
    },
    "trainer": {
        "gradient_clip_val": 0.1,
        "num_sanity_val_steps": 0,
        "max_steps": MAX_STEPS_DEFAULT,
    },
}


class Dinomaly(BaseModel):
    def __init__(self, encoder_name="dinov2reg_vit_base_14", bottleneck_dropout=0.2, decoder_depth=8,
            target_layers=None, fuse_layer_encoder=None, fuse_layer_decoder=None, remove_class_token=False):

        model = DinomalyModel(
            encoder_name=encoder_name,
            bottleneck_dropout=bottleneck_dropout,
            decoder_depth=decoder_depth,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
        )
        super().__init__(model)


class DinomalyTrainer(BaseTrainer):
    def __init__(self, model):

        if not isinstance(model, Dinomaly):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        super().__init__(model, loss_fn=None)

        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze bottleneck and decoder
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        self.trainable_modules = torch.nn.ModuleList([self.model.bottleneck, self.model.decoder])
        self._initialize_trainable_modules(self.trainable_modules)

    def configure_optimizers(self):
        optimizer_config = TRAINING_CONFIG["optimizer"]
        assert isinstance(optimizer_config, dict)
        self.optimizer = StableAdamW([{"params": self.trainable_modules.parameters()}], **optimizer_config)

        # Create a scheduler config with dynamically determined total steps
        scheduler_config = TRAINING_CONFIG["scheduler"].copy()
        assert isinstance(scheduler_config, dict)
        scheduler_config["total_iters"] = self.max_steps
        self.scheduler = WarmCosineScheduler(self.optimizer, **scheduler_config)

        self.gradient_clip_val = 0.1

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        loss = self.model(images, global_step=self.current_step)
        return {"loss": loss}

    @staticmethod
    def _initialize_trainable_modules(trainable_modules: torch.nn.ModuleList) -> None:
        for m in trainable_modules.modules():
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
