# src/vad_mini/models/efficientad/trainer.py

from pathlib import Path
import tqdm

import torch
import torch.optim as optim
# from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, RandomGrayscale, Resize, ToTensor
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomGrayscale, Resize, ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from vad_mini.models.components.base_trainer import BaseTrainer, EarlyStopper
from .torch_model import EfficientAdModel, EfficientAdModelSize, reduce_tensor_elems


class EfficientAdTrainer(BaseTrainer):
    def __init__(self, teacher_out_channels=384, model_size="small", padding=False, pad_maps=True):
        model = EfficientAdModel(
            teacher_out_channels=teacher_out_channels,
            model_size=model_size,
            padding=padding,
            pad_maps=pad_maps,
        )
        super().__init__(model, loss_fn=None)

        self.model_size = model_size
        self.backbone_dir = Path("/home/namu/myspace/NAMU/backbones")
        self.imagenet_dir = Path("/home/namu/myspace/NAMU/datasets/imagenette2")
        self.batch_size = 1     # imagenet dataloader batch_size is 1

        self.max_epochs = 10
        self.max_steps = -1

    def configure_optimizers(self) -> torch.optim.Optimizer:
        self.optimizer = optim.Adam(
            list(self.model.student.parameters()) + list(self.model.ae.parameters()),
            lr=1e-4,
            weight_decay=1e-5,
        )

        if self.max_epochs < 0 and self.max_steps < 0:
            msg = "A finite number of steps or epochs must be defined"
            raise ValueError(msg)

        # lightning stops training when either 'max_steps' or 'max_epochs' is reached (earliest),
        # so actual training steps need to be determined here
        if self.max_epochs < 0:
            # max_epochs not set
            num_steps = self.max_steps
        elif self.max_steps < 0:
            # max_steps not set -> determine steps as 'max_epochs' * 'steps in a single training epoch'
            num_steps = self.max_epochs * len(self.train_loader)
        else:
            num_steps = min(
                self.max_steps,
                self.max_epochs * len(self.train_loader),
            )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(0.95 * num_steps),
            gamma=0.1
        )

    def configure_early_stoppers(self):
        self.train_early_stopper = None
        self.valid_early_stopper = None

    def on_train_start(self) -> None:
        """Set up model before training begins.

        1. Validates training parameters (batch size=1, no normalization)
        2. Sets up pretrained teacher model
        3. Prepares ImageNette dataset
        4. Calculates channel statistics
        """
        # if self.trainer.datamodule.train_batch_size != 1:
        #     msg = "train_batch_size for EfficientAd should be 1."
        #     raise ValueError(msg)

        # if self.pre_processor and extract_transforms_by_type(self.pre_processor.transform, Normalize):
        #     msg = "Transforms for EfficientAd should not contain Normalize."
        #     raise ValueError(msg)

        sample = next(iter(self.train_loader))
        image_size = sample["image"].shape[-2:]
        self.prepare_pretrained_model()
        self.prepare_imagenette_data(image_size)
        if not self.model.is_set(self.model.mean_std):
            channel_mean_std = self.teacher_channel_mean_std(self.train_loader)
            self.model.mean_std.update(channel_mean_std)

        super().on_train_start()

    def training_step(self, batch):
        try:
            # infinite dataloader; [0] getting the image not the label
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)
        except StopIteration:
            self.imagenet_iterator = iter(self.imagenet_loader)
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)

        images = batch["image"].to(self.device)
        loss_st, loss_ae, loss_stae = self.model(batch=images, batch_imagenet=batch_imagenet)
        loss = loss_st + loss_ae + loss_stae
        return {"loss": loss}

    def on_validation_epoch_start(self):
        map_norm_quantiles = self.map_norm_quantiles(self.valid_loader)
        self.model.quantiles.update(map_norm_quantiles)

        super().on_validation_epoch_start()

    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        predictions = self.model(images)
        return {**batch, **predictions}

    #################################################################
    # src/anomalib/models/image/efficient_ad/lightning_model.py
    #################################################################

    def prepare_pretrained_model(self) -> None:
        """Prepare the pretrained teacher model."""

        # pretrained_models_dir = Path("./pre_trained/")
        pretrained_models_dir = self.backbone_dir
        if not (pretrained_models_dir / "efficientad_pretrained_weights").is_dir():
            download_and_extract(pretrained_models_dir, WEIGHTS_DOWNLOAD_INFO)
        model_size_str = self.model_size.value if isinstance(self.model_size, EfficientAdModelSize) else self.model_size
        teacher_path = (
            pretrained_models_dir / "efficientad_pretrained_weights" / f"pretrained_teacher_{model_size_str}.pth"
        )
        # logger.info(f"Load pretrained teacher model from {teacher_path}")
        print(f"Load pretrained teacher model from {teacher_path}")
        self.model.teacher.load_state_dict(
            torch.load(teacher_path, map_location=torch.device(self.device), weights_only=True),
        )

    def prepare_imagenette_data(self, image_size: tuple[int, int] | torch.Size) -> None:
        """Prepare ImageNette dataset transformations."""

        self.data_transforms_imagenet = Compose(
            [
                Resize((image_size[0] * 2, image_size[1] * 2)),
                RandomGrayscale(p=0.3),
                CenterCrop((image_size[0], image_size[1])),
                ToTensor(),
            ],
        )

        if not self.imagenet_dir.is_dir():
            download_and_extract(self.imagenet_dir, IMAGENETTE_DOWNLOAD_INFO)
        imagenet_dataset = ImageFolder(self.imagenet_dir, transform=self.data_transforms_imagenet)
        self.imagenet_loader = DataLoader(imagenet_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.imagenet_iterator = iter(self.imagenet_loader)

    @torch.no_grad()
    def teacher_channel_mean_std(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Calculate channel-wise mean and std of teacher model activations."""

        arrays_defined = False
        n: torch.Tensor | None = None
        chanel_sum: torch.Tensor | None = None
        chanel_sum_sqr: torch.Tensor | None = None

        # print("Calculate teacher channel mean & std")
        for batch in tqdm.tqdm(dataloader, desc="Calculate teacher channel mean & std", ascii=True, leave=False):
            y = self.model.teacher(batch["image"].to(self.device))
            if not arrays_defined:
                _, num_channels, _, _ = y.shape
                n = torch.zeros((num_channels,), dtype=torch.int64, device=y.device)
                chanel_sum = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                chanel_sum_sqr = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                arrays_defined = True

            n += y[:, 0].numel()
            chanel_sum += torch.sum(y, dim=[0, 2, 3])
            chanel_sum_sqr += torch.sum(y**2, dim=[0, 2, 3])

        if n is None:
            msg = "The value of 'n' cannot be None."
            raise ValueError(msg)

        channel_mean = chanel_sum / n
        channel_std = (torch.sqrt((chanel_sum_sqr / n) - (channel_mean**2))).float()[None, :, None, None]
        channel_mean = channel_mean.float()[None, :, None, None]
        return {"mean": channel_mean, "std": channel_std}

    @torch.no_grad()
    def map_norm_quantiles(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Calculate quantiles of student and autoencoder feature maps."""

        maps_st = []
        maps_ae = []
        # logger.info("Calculate Validation Dataset Quantiles")
        # print("Calculate Validation Dataset Quantiles")

        for batch in tqdm.tqdm(dataloader, desc="Calculate Validation Dataset Quantiles", ascii=True, leave=False):
            for img, label in zip(batch["image"], batch["label"], strict=True):
                if label == 0:  # only use good images of validation set!
                    map_st, map_ae = self.model.get_maps(img.to(self.device), normalize=False)
                    maps_st.append(map_st)
                    maps_ae.append(map_ae)

        qa_st, qb_st = self._get_quantiles_of_maps(maps_st)
        qa_ae, qb_ae = self._get_quantiles_of_maps(maps_ae)
        return {"qa_st": qa_st, "qa_ae": qa_ae, "qb_st": qb_st, "qb_ae": qb_ae}

    def _get_quantiles_of_maps(self, maps: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate quantiles of anomaly maps."""

        maps_flat = reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=0.9).to(self.device)
        qb = torch.quantile(maps_flat, q=0.995).to(self.device)
        return qa, qb
