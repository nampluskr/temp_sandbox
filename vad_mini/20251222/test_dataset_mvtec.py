# experiments/load_dataset_mvtec.py
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

from vad_mini.data.datasets import MVTecDataset
from vad_mini.data.dataloaders import get_train_loader, get_test_loader
# from vad_mini.data.dataloaders import collate_fn
from vad_mini.data.transforms import get_train_transform, get_test_transform, get_mask_transform


# DATA_DIR = "/mnt/d/deep_learning/datasets/mvtec"
DATA_DIR = "/home/namu/myspace/NAMU/datasets/mvtec"
CATEGORY = "bottle"
IMG_SIZE = 256
BATCH_SIZE = 64


if __name__ == "__main__":

    #######################################################
    ## Train dataset
    #######################################################

    train_dataset = MVTecDataset(
        root_dir=DATA_DIR,
        category=CATEGORY,
        split="train",
        transform=get_train_transform(img_size=IMG_SIZE, normalize=True),
        mask_transform=get_mask_transform(img_size=IMG_SIZE),
    )

    data = train_dataset[10]
    image = data["image"]
    label = data["label"]
    defect_type = data["defect_type"]
    mask = data["mask"]

    print("\n" + "=" * 60 + "\n" + "*** Train Dataset:" + "\n" + "=" * 60)
    print(f">> total: {len(train_dataset)}")
    print(f">> normal: {train_dataset.count_normal()}")
    print(f">> anomaly: {train_dataset.count_anomaly()}")

    print("\n*** Train Sample:")
    print(f">> image: {image.shape}")
    print(f">> label: {label}")
    print(f">> defect_type: {defect_type}")
    print(f">> mask:  {mask.shape}")

    #######################################################
    ## Train Dataloader
    #######################################################

    train_loader = get_train_loader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
    )
    print("\n" + "=" * 60 + "\n" + "*** Train Dataloader:" + "\n" + "=" * 60)
    print(f">> total: {len(train_dataset)}")

    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]
    print(f">> batch size: {BATCH_SIZE}")
    print(f">> batch images: {images.shape}")
    print(f">> batch images: {labels.shape}")

    #######################################################
    ## Test dataset
    #######################################################

    test_dataset = MVTecDataset(
        root_dir=DATA_DIR,
        category=CATEGORY,
        split="test",
        transform=get_test_transform(img_size=IMG_SIZE, normalize=True),
        mask_transform=get_mask_transform(img_size=IMG_SIZE),
    )
    print("\n" + "=" * 60 + "\n" + "*** Test Dataset:" + "\n" + "=" * 60)
    print(f">> total: {len(test_dataset)}")
    print(f">> normal: {test_dataset.count_normal()}")
    print(f">> anomaly: {test_dataset.count_anomaly()}")

    data = test_dataset[20]
    image = data["image"]
    label = data["label"]
    defect_type = data["defect_type"]
    mask = data["mask"]

    print("\n*** Test Sample:")
    print(f">> image: {image.shape}")
    print(f">> label: {label}")
    print(f">> defect_type: {defect_type}")
    print(f">> mask:  {mask.shape}")

    #######################################################
    ## Test Dataloader
    #######################################################

    test_loader = get_test_loader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
    )
    print("\n" + "=" * 60 + "\n" + "*** Test Dataloader:" + "\n" + "=" * 60)
    print(f">> total: {len(test_dataset)}")

    batch = next(iter(test_loader))
    images = batch["image"]
    labels = batch["label"]
    masks = batch["mask"].numpy()
    print(f">> batch size: {BATCH_SIZE}")
    print(f">> batch images: {images.shape}")
    print(f">> batch labels: {labels.shape}")
    print(f">> batch masks: {masks.shape}")
