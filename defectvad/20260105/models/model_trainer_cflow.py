# src/defectvad/models/cflow/trainer.py

import einops
import torch
import torch.nn.functional as F
import torch.optim as optim

from defectvad.common.base_model import BaseModel
from defectvad.common.base_trainer import BaseTrainer

from .torch_model import CflowModel
from .utils import get_logp, positional_encoding_2d


class Cflow(BaseModel):
    def __init__(self, backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], 
                 fiber_batch_size=64, decoder="freia-cflow", condition_vector=128, coupling_blocks=8,
                 clamp_alpha=1.9, permute_soft=False):

        model = CflowModel(
            backbone=backbone,
            layers=layers,
            pre_trained=True,
            fiber_batch_size=fiber_batch_size,
            decoder=decoder,
            condition_vector=condition_vector,
            coupling_blocks=coupling_blocks,
            clamp_alpha=clamp_alpha,
            permute_soft=permute_soft,
        )
        super().__init__(model)


class CflowTrainer(BaseTrainer):
    def __init__(self, model):

        if not isinstance(model, Cflow):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        super().__init__(model, loss_fn=None)

    def configure_optimizers(self):
        decoders_parameters = []
        for decoder_idx in range(len(self.model.pool_layers)):
            decoders_parameters.extend(list(self.model.decoders[decoder_idx].parameters()))

        self.decoder_optimizer = optim.Adam(
            params=decoders_parameters,
            lr=0.0001,
        )

    def training_step(self, batch):
        """Perform a training step of the CFLOW model.

        1. Extract features using the encoder
        2. Process features in fiber batches
        3. Apply positional encoding
        4. Train decoders using normalizing flows
        """
        images: torch.Tensor = batch["image"].to(self.device)
        activation = self.model.encoder(images)
        avg_loss = torch.zeros([1], dtype=torch.float64).to(self.device)

        height = []
        width = []
        for layer_idx, layer in enumerate(self.model.pool_layers):
            encoder_activations = activation[layer].detach()  # BxCxHxW

            batch_size, dim_feature_vector, im_height, im_width = encoder_activations.size()
            image_size = im_height * im_width
            embedding_length = batch_size * image_size  # number of rows in the conditional vector

            height.append(im_height)
            width.append(im_width)
            # repeats positional encoding for the entire batch 1 C H W to B C H W
            pos_encoding = einops.repeat(
                positional_encoding_2d(self.model.condition_vector, im_height, im_width).unsqueeze(0),
                "b c h w-> (tile b) c h w",
                tile=batch_size,
            ).to(self.device)
            c_r = einops.rearrange(pos_encoding, "b c h w -> (b h w) c")  # BHWxP
            e_r = einops.rearrange(encoder_activations, "b c h w -> (b h w) c")  # BHWxC
            perm = torch.randperm(embedding_length)  # BHW
            decoder = self.model.decoders[layer_idx].to(self.device)

            fiber_batches = embedding_length // self.model.fiber_batch_size  # number of fiber batches
            if fiber_batches <= 0:
                msg = "Make sure we have enough fibers, otherwise decrease N or batch-size!"
                raise ValueError(msg)

            for batch_num in range(fiber_batches):  # per-fiber processing
                self.decoder_optimizer.zero_grad()
                if batch_num < (fiber_batches - 1):
                    idx = torch.arange(
                        batch_num * self.model.fiber_batch_size,
                        (batch_num + 1) * self.model.fiber_batch_size,
                    )
                else:  # When non-full batch is encountered batch_num * N will go out of bounds
                    idx = torch.arange(batch_num * self.model.fiber_batch_size, embedding_length)
                # get random vectors
                c_p = c_r[perm[idx]]  # NxP
                e_p = e_r[perm[idx]]  # NxC
                # decoder returns the transformed variable z and the log Jacobian determinant
                p_u, log_jac_det = decoder(e_p, [c_p])
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector  # likelihood per dim
                loss = -F.logsigmoid(log_prob).mean()
                loss.backward()
                self.decoder_optimizer.step()
                avg_loss += loss.sum()

        return {"loss": avg_loss.detach()}
