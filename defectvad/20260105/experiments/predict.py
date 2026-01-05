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
from defectvad.common.visualizer import Visualizer


def set_environment(config):
    os.environ["BACKBONE_DIR"] = config["path"]["backbone"] # IMPORTANT: used in backbone loading
    os.environ["DATASET_DIR"] = config["path"]["dataset"]   # IMPORTTAT: used in dataset loading


def run_prediction(dataset_name, category_name, model_name, max_epochs):
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

    if 1:
        print("\n*** Prediction (dataloader):")
        preds = model.predict(test_loader)
        for k, v in preds.items():
            if torch.is_tensor(v):
                print(f" > {k}: {v.shape}")

    if 0:
        print("\n*** Prediction (batch):")
        batch = next(iter(test_loader))
        preds = model.predict(batch)
        for k, v in preds.items():
            if torch.is_tensor(v):
                print(f" > {k}: {v.shape}")

    if 0:
        print("\n*** Prediction (images):")
        batch = next(iter(test_loader))
        batch = {"image": batch["image"], "label": batch["label"]}
        preds = model.predict(batch)
        for k, v in preds.items():
            if torch.is_tensor(v):
                print(f" > {k}: {v.shape}")

    # Visualize (without thresholds)
    visualizer = Visualizer(preds)
    # visualizer.show_anomaly(max_samples=3)
    # visualizer.show_normal(max_samples=3)

    # Visualize (with thresholds)
    evaluator = Evaluator(model)
    image_results = evaluator.evaluate_image_level(test_loader)
    pixel_results = evaluator.evaluate_pixel_level(test_loader)
    visualizer.set_image_threshold(image_results["th"])
    visualizer.set_pixel_threshold(pixel_results["th"])

    if 0:
        visualizer.show_anomaly(max_samples=3, denormalize=config["dataset"]["normalize"])
        visualizer.show_normal(max_samples=3, denormalize=config["dataset"]["normalize"])

    if 1:
        anomaly_dir = os.path.join(experiment_dir, "anomaly")
        visualizer.save_anomaly(save_dir=anomaly_dir, denormalize=config["dataset"]["normalize"])
        normal_dir = os.path.join(experiment_dir, "normal")
        visualizer.save_normal(save_dir=normal_dir, denormalize=config["dataset"]["normalize"])


if __name__ == "__main__":

    dataset, category, model, max_epochs = "mvtec", "bottle", "efficientad", 20
    run_prediction(dataset, category, model, max_epochs)
