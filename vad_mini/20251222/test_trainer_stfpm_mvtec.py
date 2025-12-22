# experiments/test_trainer_stfpm.py
import os, sys
source_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

import os
import numpy as numpy
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as T

from vad_mini.utils import set_seed
from vad_mini.data.datasets import MVTecDataset
from vad_mini.data.dataloaders import get_train_loader, get_test_loader
from vad_mini.data.transforms import get_train_transform, get_test_transform, get_mask_transform

# from vad_mini.models.stfpm.torch_model import STFPMModel
# from vad_mini.models.stfpm.loss import STFPMLoss
# from vad_mini.models.stfpm.anomaly_map import AnomalyMapGenerator


# DATA_DIR = "/mnt/d/deep_learning/datasets/mvtec"
DATA_DIR = "/home/namu/myspace/NAMU/datasets/mvtec"
CATEGORY = "grid"
IMG_SIZE = 256
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


if __name__ == "__main__":

    set_seed(SEED)

    #######################################################
    ## Load Datste and Dataloader
    #######################################################

    train_dataset = MVTecDataset(
        root_dir=DATA_DIR,
        category=CATEGORY,
        split="train",
        transform=get_train_transform(img_size=IMG_SIZE, normalize=True),
        mask_transform=get_mask_transform(img_size=IMG_SIZE),
    )
    test_dataset = MVTecDataset(
        root_dir=DATA_DIR,
        category=CATEGORY,
        split="test",
        transform=get_test_transform(img_size=IMG_SIZE, normalize=True),
        mask_transform=get_mask_transform(img_size=IMG_SIZE),
    )
    train_loader = get_train_loader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
    )
    test_loader = get_test_loader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE // 2,
    )

    #######################################################
    ## Train Model
    #######################################################

    from vad_mini.models.stfpm.trainer import STFPMTrainer

    trainer = STFPMTrainer(backbone="resnet34")
    train_outputs = trainer.fit(train_loader, num_epochs=10, valid_loader=test_loader)

    threshold = trainer.calibrate_threshold(train_loader, method="quantile", quantile=0.99)
    print(f">> quantile threshold (99%): {threshold:.3f}")
    
    threshold = trainer.calibrate_threshold(train_loader, method="quantile", quantile=0.97)
    print(f">> quantile threshold (97%): {threshold:.3f}")

    threshold = trainer.calibrate_threshold(train_loader, method="mean_std", sigma=3)
    print(f">> mean_std threshold (3-sigma): {threshold:.3f}")

    threshold = trainer.calibrate_threshold(train_loader, method="mean_std", sigma=2)
    print(f">> mean_std threshold (2-sigma): {threshold:.3f}")

    

