# experiments/load_dataset.py
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

from anomaly_detection.datasets import ViSADataset
from anomaly_detection.config import load_config


if __name__ == "__main__":

    config = load_config(os.path.join(source_dir, "..", "configs", "paths.yaml"))

    #######################################################
    ## train dataset
    #######################################################

    dataset = ViSADataset(
        root_dir=config["VISA_DIR"], 
        category="candle", 
        split="train",
        transform=T.Compose([T.Resize((256, 256)), T.ToTensor(),]),
        mask_transform=T.Compose([T.Resize((256, 256)), T.ToTensor(),]),
    )

    data = dataset[10]
    image = data["image"].permute(1, 2, 0).numpy()
    label = data["label"].numpy()
    defect_type = data["defect_type"]
    mask = None if data["mask"] is None else data["mask"].squeeze().numpy()

    print("\n*** Train Dataset:")
    print(f">> total: {len(dataset)}")
    print(f">> normal: {dataset.count_normal()}")
    print(f">> anomaly: {dataset.count_anomaly()}")

    print("\n*** Train Sample:")
    print(f">> image: {image.shape}")
    print(f">> label: {label}")
    print(f">> defect_type: {defect_type}")
    print(f">> mask:  {mask if mask is None else mask.shape}")

    #######################################################
    ## test dataset
    #######################################################

    dataset = ViSADataset(
        root_dir=config["VISA_DIR"], 
        category="candle", 
        split="test",
        transform=T.Compose([T.Resize((256, 256)), T.ToTensor(),]),
        mask_transform=T.Compose([T.Resize((256, 256)), T.ToTensor(),]),
    )
    print("\n*** Test Dataset:")
    print(f">> total: {len(dataset)}")
    print(f">> normal: {dataset.count_normal()}")
    print(f">> anomaly: {dataset.count_anomaly()}")

    data = dataset[-10]
    image = data["image"].permute(1, 2, 0).numpy()
    label = data["label"].numpy()
    defect_type = data["defect_type"]
    mask = None if data["mask"] is None else data["mask"].squeeze().numpy()

    print("\n*** Test Sample:")
    print(f">> image: {image.shape}")
    print(f">> label: {label}")
    print(f">> defect_type: {defect_type}")
    print(f">> mask:  {mask if mask is None else mask.shape}")
