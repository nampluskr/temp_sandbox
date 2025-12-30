# common/factory.py

import importlib

from ..data.transforms import get_train_transform, get_test_transform, get_mask_transform
from ..data.dataloaders import get_dataloader


def create_dataset_from_config(split, dataset_config):
    module = importlib.import_module(dataset_config["module"])
    DatasetClass = getattr(module, dataset_config["class"])

    if split == "train":
        transform_fn = get_train_transform
    elif split == "test":
        transform_fn = get_test_transform

    return DatasetClass(
        root_dir=dataset_config["path"],
        category=dataset_config["category"],
        split=split,
        transform=transform_fn(
            img_size=dataset_config["img_size"],
            crop_size=dataset_config["crop_size"],
            normalize=dataset_config["normalize"],
            mean=dataset_config["mean"],
            std=dataset_config["std"],
        ),
        mask_transform=get_mask_transform(
            img_size=dataset_config["img_size"],
            crop_size=dataset_config["crop_size"],
        ),
    )


def create_dataloader_from_config(dataset, dataloader_config):
    return get_dataloader(
        dataset=dataset,
        batch_size=dataloader_config["batch_size"],
        **dataloader_config["params"],
    )

def create_trainer_from_config(trainer_config):
    module = importlib.import_module(trainer_config["module"])
    TrainerClass = getattr(module, trainer_config["class"])
    return TrainerClass(**trainer_config["params"])


def train_from_config(trainer, trainer_config, train_loader, valid_loader):
    trainer.fit(train_loader,
        max_epochs=trainer_config["max_epochs"],
        valid_loader=valid_loader if trainer_config["validate"] else None,
    )

    if not trainer_config["validate"]:
        results = trainer.validate(valid_loader)
        print(">> " + ", ".join([f"{k}:{v:.3f}" for k, v in results.items()]))
