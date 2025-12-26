# experiments/test_trainer_stfpm.py
import os, sys
source_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

from vad_mini.utils import set_seed
from vad_mini.data.datasets import MVTecDataset
from vad_mini.data.dataloaders import get_train_loader, get_test_loader
from vad_mini.data.transforms import get_train_transform, get_test_transform, get_mask_transform


# DATA_DIR = "/mnt/d/deep_learning/datasets/mvtec"
DATA_DIR = "/home/namu/myspace/NAMU/datasets/mvtec"
CATEGORY = "bottle"
IMG_SIZE = 224
CROP_SIZE = None
BATCH_SIZE = 1
NORMALIZE = False
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
        transform=get_train_transform(img_size=IMG_SIZE, crop_size=CROP_SIZE, normalize=NORMALIZE),
        mask_transform=get_mask_transform(img_size=IMG_SIZE),
    )
    test_dataset = MVTecDataset(
        root_dir=DATA_DIR,
        category=CATEGORY,
        split="test",
        transform=get_test_transform(img_size=IMG_SIZE, crop_size=CROP_SIZE, normalize=NORMALIZE),
        mask_transform=get_mask_transform(img_size=IMG_SIZE),
    )
    train_loader = get_train_loader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loader = get_test_loader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        persistent_workers=False,
    )

    #######################################################
    ## Train Model
    #######################################################

    from vad_mini.models.anomaly_dino.trainer import AnomalyDINOTrainer

    trainer = AnomalyDINOTrainer(encoder_name="dinov2_vit_small_14", sampling_ratio=0.1)
    train_outputs = trainer.fit(train_loader, max_epochs=1, valid_loader=test_loader)
    thresholds = trainer.calibrate_threshold(train_loader)

    print()
    print(f">> quantile threshold (99%): {thresholds['99%']:.3f}")
    print(f">> quantile threshold (97%): {thresholds['97%']:.3f}")
    print(f">> quantile threshold (95%): {thresholds['95%']:.3f}")
    print(f">> mean_std threshold (3-sigma): {thresholds['3-sigma']:.3f}")
    print(f">> mean_std threshold (2-sigma): {thresholds['2-sigma']:.3f}")
    print(f">> mean_std threshold (1-sigma): {thresholds['1-sigma']:.3f}")

    
