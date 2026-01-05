# experiments/evaluate.py

import os
import sys
import argparse
import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_CIR = os.path.join(PROJECT_DIR, "src")
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

if SOURCE_CIR not in sys.path:
    sys.path.insert(0, SOURCE_CIR)


from defectvad.common.config import load_config, merge_configs, save_config
from defectvad.common.utils import set_seed
from defectvad.common.factory import create_dataset, create_dataloader
from defectvad.common.factory import create_model, create_trainer
from defectvad.common.evaluator import Evaluator


def set_environment(config):
    os.environ["BACKBONE_DIR"] = config["path"]["backbone"] # IMPORTANT: used in backbone loading
    os.environ["DATASET_DIR"] = config["path"]["dataset"]   # IMPORTTAT: used in dataset loading


def run_evaluation(dataset_name, category_name, model_name, max_epochs):
    experiment_name = f"{dataset_name}_{category_name}_{model_name}_{max_epochs}epoch"
    experiment_dir = os.path.join(OUTPUT_DIR, dataset_name, category_name, model_name)
    weights_name = f"weights_{experiment_name}.pth"
    configs_name = f"configs_{experiment_name}.yaml"

    config = load_config(experiment_dir, configs_name)
    set_environment(config)
    set_seed(config["seed"])

    train_dataset = create_dataset("train", config["dataset"])
    test_dataset = create_dataset("test", config["dataset"])
    train_loader = create_dataloader(train_dataset, config["train_loader"])
    test_loader = create_dataloader(test_dataset, config["test_loader"])

    model = create_model(config["model"])
    model.load(os.path.join(experiment_dir, weights_name))

    evaluator = Evaluator(model)

    print(f"\n*** Evaluation: Test Dataset (normal + anomaly)")
    image_results = evaluator.evaluate_image_level(test_loader)
    print(" > Image: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))
    pixel_results = evaluator.evaluate_pixel_level(test_loader)
    print(" > Pixel: " + ", ".join([f"{k}:{v:.3f}" for k, v in pixel_results.items()]))

    if 0:
        print(f"\n*** Threshold calibration: Train Dataset (normal-only)")

        image_thresholds = evaluator.calibrate_image_thresholds(train_loader)
        image_results = evaluator.evaluate_image_level(test_loader, threshold=image_thresholds["95%"])
        print(" > Image: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))

        pixel_thresholds = evaluator.calibrate_pixel_thresholds(train_loader)
        pixel_results = evaluator.evaluate_pixel_level(test_loader, threshold=pixel_thresholds["99%"])
        print(" > Pixel: " + ", ".join([f"{k}:{v:.3f}" for k, v in pixel_results.items()]))


if __name__ == "__main__":

    dataset, category, model, max_epochs = "mvtec", "bottle", "stfpm", 10
    run_evaluation(dataset, category, model, max_epochs)
