# experiments/load_stfpm.py
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
from vad_mini.data.transforms import get_train_transform, get_test_transform, get_mask_transform

from vad_mini.models.stfpm.torch_model import STFPMModel
from vad_mini.models.stfpm.loss import STFPMLoss
from vad_mini.models.stfpm.anomaly_map import AnomalyMapGenerator


# DATA_DIR = "/mnt/d/deep_learning/datasets/mvtec"
DATA_DIR = "/home/namu/myspace/NAMU/datasets/mvtec"
CATEGORY = "bottle"
IMG_SIZE = 256
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    if 0:
        train_dataset = MVTecDataset(
            root_dir=DATA_DIR,
            category=CATEGORY,
            split="train",
            transform=get_train_transform(img_size=IMG_SIZE, normalize=True),
            mask_transform=get_mask_transform(img_size=IMG_SIZE),
        )
        train_loader = get_train_loader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
        )
        model = STFPMModel(
            backbone='resnet34', 
            layers=['layer1', 'layer2', 'layer3']
        ).to(DEVICE)
        batch = next(iter(train_loader))
        images = batch["image"].to(DEVICE)
        
        model.train()
        outputs = model(images)

        print(f" > teature_features: {outputs[0]['layer1'].shape}")
        print(f" > teature_features: {outputs[0]['layer2'].shape}")
        print(f" > teature_features: {outputs[0]['layer3'].shape}")
        print(f" > student_features: {outputs[1]['layer1'].shape}")
        print(f" > student_features: {outputs[1]['layer2'].shape}")
        print(f" > student_features: {outputs[1]['layer3'].shape}")

    if 1:
        test_dataset = MVTecDataset(
            root_dir=DATA_DIR,
            category=CATEGORY,
            split="test",
            transform=get_test_transform(img_size=IMG_SIZE, normalize=True),
            mask_transform=get_mask_transform(img_size=IMG_SIZE),
        )
        test_loader = get_test_loader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
        )
        model = STFPMModel(
            backbone='resnet34', 
            layers=['layer1', 'layer2', 'layer3']
        ).to(DEVICE)
        batch = next(iter(test_loader))
        images = batch["image"].to(DEVICE)

        model.eval()
        outputs = model(images)
        
        print(f" > anomaly_map: {outputs['anomaly_map'].shape}")    
        print(f" > pred_score: {outputs['pred_score'].shape}")    
