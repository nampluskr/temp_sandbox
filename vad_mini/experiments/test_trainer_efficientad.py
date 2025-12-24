# experiments/load_stfpm.py
import os, sys
source_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
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

from vad_mini.models.efficientad.torch_model import EfficientAdModel


# DATA_DIR = "/mnt/d/deep_learning/datasets/mvtec"
DATA_DIR = "/home/namu/myspace/NAMU/datasets/mvtec"
CATEGORY = "grid"
IMG_SIZE = 256
BATCH_SIZE = 1
NORMALIZE = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


if __name__ == "__main__":

    set_seed(SEED)

    #######################################################
    ## Load Datset and Dataloader
    #######################################################

    train_dataset = MVTecDataset(
        root_dir=DATA_DIR,
        category=CATEGORY,
        split="train",
        transform=get_train_transform(img_size=IMG_SIZE, normalize=NORMALIZE),
        mask_transform=get_mask_transform(img_size=IMG_SIZE),
    )
    train_loader = get_train_loader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )
    test_dataset = MVTecDataset(
        root_dir=DATA_DIR,
        category=CATEGORY,
        split="test",
        transform=get_test_transform(img_size=IMG_SIZE, normalize=NORMALIZE),
        mask_transform=get_mask_transform(img_size=IMG_SIZE),
    )
    test_loader = get_test_loader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )

    #######################################################
    ## Train Model
    #######################################################

    from vad_mini.models.efficientad.trainer import EfficientAdTrainer

    if 0:
        trainer = EfficientAdTrainer(model_size="small")
        trainer.fit(train_loader, num_epochs=10, num_steps=1000, valid_loader=test_loader)

    if 1:
        trainer = EfficientAdTrainer(model_size="medium")
        trainer.fit(train_loader, num_epochs=10, num_steps=1000, valid_loader=test_loader)
        
