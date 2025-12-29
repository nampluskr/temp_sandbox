# src/defectvad/models/ganomaly/trainer.py

import torch
import torch.optim as optim

from defectvad.common.base_trainer import BaseTrainer
from defectvad.common.early_stopper import EarlyStopper
from .torch_model import GanomalyModel
from .loss import DiscriminatorLoss, GeneratorLoss


class GanomalyTrainer(BaseTrainer):
    def __init__(self, input_size=(256, 256), num_input_channels=3, n_features=64, latent_vec_size=100,
                 extra_layers=0, add_final_conv_layer=True, batch_size=32, wadv=1, wcon=50, wenc=1,
                 lr=0.0002, beta1=0.5, beta2=0.999):

        model = GanomalyModel(
            input_size=input_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            latent_vec_size=latent_vec_size,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
        )
        super().__init__(model, loss_fn=None)

        self.n_features = n_features
        self.latent_vec_size = latent_vec_size
        self.extra_layers = extra_layers
        self.add_final_conv_layer = add_final_conv_layer

        self.generator_loss = GeneratorLoss(wadv, wcon, wenc)
        self.discriminator_loss = DiscriminatorLoss()

        self.min_scores = torch.tensor(float("inf"), device=self.device)
        self.max_scores = torch.tensor(float("-inf"), device=self.device)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def _reset_min_max(self):
        self.min_scores = torch.tensor(float("inf"), device=self.device)
        self.max_scores = torch.tensor(float("-inf"), device=self.device)

    def configure_optimizers(self):
        self.optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
        self.optimizer_g = optim.Adam(
            self.model.generator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
 
    def configure_early_stoppers(self):
        self.train_early_stopper = None
        self.valid_early_stopper = EarlyStopper(patience=10, min_delta=1e-4, monitor="auroc")

    def training_step(self, batch) -> dict:
        images = batch["image"].to(self.device)
        d_opt, g_opt = self.optimizer_d, self.optimizer_g

        # forward pass
        padded, fake, latent_i, latent_o = self.model(images)
        pred_real, _ = self.model.discriminator(padded)

        # generator update
        pred_fake, _ = self.model.discriminator(fake)
        g_loss = self.generator_loss(latent_i, latent_o, padded, fake, pred_real, pred_fake)

        g_opt.zero_grad()
        g_loss.backward(retain_graph=True)
        g_opt.step()

        # discrimator update
        pred_fake, _ = self.model.discriminator(fake.detach())
        d_loss = self.discriminator_loss(pred_real, pred_fake)

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()
        return {"g_loss": g_loss, "d_loss": d_loss}

    def on_validation_epoch_start(self):
        self._reset_min_max()
        super().on_validation_epoch_start()

    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        with torch.no_grad():
            outputs = self.model(images)
            self.max_scores = torch.max(self.max_scores, torch.max(outputs["pred_score"]))
            self.min_scores = torch.min(self.min_scores, torch.min(outputs["pred_score"]))
        return {**batch, **outputs}

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        return (scores - self.min_scores.to(scores.device)) / (
            self.max_scores.to(scores.device) - self.min_scores.to(scores.device)
        )
