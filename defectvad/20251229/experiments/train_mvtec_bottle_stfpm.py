# experiments/train_mvtec_bottle_stfpm.py

import os
import sys
import argparse
import importlib

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
source_dir = os.path.join(project_dir, "src")
config_dir = os.path.join(project_dir, "configs")

if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

from defectvad.common.seed import set_seed
from defectvad.common.config import load_config, merge_configs
from defectvad.data.transforms import get_train_transform, get_test_transform, get_mask_transform
from defectvad.data.dataloaders import get_train_loader, get_test_loader


def main():
    #################################################################
    # Hyperparameters
    #################################################################

    BACKBONE_DIR = "/home/namu/myspace/NAMU/backbones"
    DATASET_DIR = "/home/namu/myspace/NAMU/datasets/mvtec"
    CATEGORY = "bottle"

    SEED = 42
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True

    TRAIN_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    MAX_EPOCHS = 10

    #################################################################
    # Initialization
    #################################################################

    os.environ["BACKBONE_DIR"] = BACKBONE_DIR
    set_seed(SEED)

    #################################################################
    # Datasets
    #################################################################

    from defectvad.data.datasets import MVTecDataset

    train_dataset = MVTecDataset(
        root_dir=DATASET_DIR,
        category=CATEGORY,
        split="train",
        transform=get_train_transform(
            img_size=IMG_SIZE,
            crop_size=CROP_SIZE,
            normalize=NORMALIZE,
        ),
        mask_transform=get_mask_transform(
            img_size=IMG_SIZE,
            crop_size=CROP_SIZE,
        ),
    )

    test_dataset = MVTecDataset(
        root_dir=DATASET_DIR,
        category=CATEGORY,
        split="test",
        transform=get_train_transform(
            img_size=IMG_SIZE,
            crop_size=CROP_SIZE,
            normalize=NORMALIZE,
        ),
        mask_transform=get_mask_transform(
            img_size=IMG_SIZE,
            crop_size=CROP_SIZE,
        ),
    )

    #################################################################
    # Dataloaders
    #################################################################

    train_loader = get_train_loader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
    )

    test_loader = get_test_loader(
        dataset=test_dataset,
        batch_size=TEST_BATCH_SIZE,
    )

    #################################################################
    # Model Trainer
    #################################################################

    from defectvad.models.stfpm.trainer import STFPMTrainer

    trainer = STFPMTrainer(backbone="resnet50", layers=["layer1", "layer2", "layer3"])

    trainer.fit(train_loader, max_epochs=MAX_EPOCHS, valid_loader=test_loader)

if __name__ == "__main__":

    main()
