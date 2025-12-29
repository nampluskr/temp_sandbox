# experiments/train.py

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
from defectvad.data.dataloaders import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def import_class(module_path, class_name):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main(dataset_name, category_name, model_name):
    #################################################################
    # Configuration
    #################################################################

    config = merge_configs(
        load_config(os.path.join(config_dir, "base.yaml")),
        load_config(os.path.join(config_dir, "datasets", f"{dataset_name}.yaml")),
        load_config(os.path.join(config_dir, "models", f"{model_name}.yaml")),
        load_config(os.path.join(config_dir, "override.yaml")),
    )

    path_config = config["path"]
    dataset_config = config["dataset"]
    train_config = config["dataloader"]["train"]
    test_config = config["dataloader"]["test"]
    trainer_config = config["trainer"]

    #################################################################
    # Initialization
    #################################################################

    os.environ["BACKBONE_DIR"] = path_config["backbone"]
    os.environ["DATASET_DIR"] = path_config["dataset"]
    set_seed(config.get("seed", 42))

    #################################################################
    # Datasets
    #################################################################

    DatasetClass = import_class(dataset_config["module"], dataset_config["class"])

    train_dataset = DatasetClass(
        root_dir=path_config[dataset_name],
        category=category_name,
        split="train",
        transform=get_train_transform(
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

    test_dataset = DatasetClass(
        root_dir=path_config[dataset_name],
        category=category_name,
        split="test",
        transform=get_train_transform(
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

    #################################################################
    # Dataloaders
    #################################################################

    train_loader = get_dataloader(
        dataset=train_dataset,
        batch_size=train_config["batch_size"],
        **train_config["params"],
    )

    test_loader = get_dataloader(
        dataset=test_dataset,
        batch_size=test_config["batch_size"],
        **test_config["params"],
    )

    #################################################################
    # Model Trainer
    #################################################################

    TrainerClass = import_class(trainer_config["module"], trainer_config["class"])

    trainer = TrainerClass(**trainer_config["params"])

    trainer.fit(train_loader, 
        max_epochs=trainer_config["max_epochs"],
        valid_loader=test_loader if trainer_config["validate"] else None,
    )

if __name__ == "__main__":

    args = parse_args()
    main(args.dataset, args.category, args.model)

