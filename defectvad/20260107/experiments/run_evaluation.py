# experiments/run.py

#####################################################################
# Experiment split lists
#####################################################################

DATASET_LIST = ["mvtec"]
CATEGORY_LIST = {
    # "mvtec": ["carpet", "grid", "leather", "tile", "wood"],  # texture
    "mvtec": [["carpet", "grid", "leather", "tile", "wood"]],  # texture
    # "mvtec": [["bottle", "grid"]],  # test category
}
MODEL_LIST = ["stfpm"]
# MODEL_LIST = ["fastflow"]
# MODEL_LIST = ["reversedistill"]
# MODEL_LIST = ["reversedistill", "efficientad", "stfpm"]
# MODEL_LIST = ["cflow", "fastflow", "csflow", "uflow"]
# MODEL_LIST = ["dinomaly"]

MAX_EPOCHS = 10
IMAGE_LEVEL = True
PIXEL_LEVEL = False

#####################################################################
# Script file path (absolute)
#####################################################################

import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_FILE = os.path.join(PROJECT_DIR, "experiments", "evaluate.py")


#####################################################################
# Run function
#####################################################################

def run(script_file, dataset_list, category_list, model_list):
    import subprocess

    if not os.path.isfile(SCRIPT_FILE):
        raise FileNotFoundError(SCRIPT_FILE)

    total = 0
    for dataset in dataset_list:
        if dataset not in category_list:
            raise ValueError(f"category_list not defined for dataset: {dataset}")
        total += len(category_list[dataset]) * len(model_list)

    counter = 0
    for dataset in dataset_list:
        if dataset not in category_list:
            raise ValueError(f"category_list not defined for dataset: {dataset}")

        for model in model_list:
            for category in category_list[dataset]:
                counter += 1
                category = [category] if isinstance(category, str) else category

                print("\n" + "=" * 80)
                print(f"[RUN {counter}/{total}] {dataset} | "
                      f"{', '.join(category)} | {model} ({MAX_EPOCHS} epochs)"
                )
                print("=" * 80)

                cmd = [sys.executable, script_file]
                cmd.extend(["--dataset", dataset])
                cmd.extend(["--category"] + category)
                cmd.extend(["--model", model])
                cmd.extend(["--max_epochs", str(MAX_EPOCHS)])

                if IMAGE_LEVEL:
                    cmd.extend(["--image_level"])

                if PIXEL_LEVEL:
                    cmd.extend(["--pixel_level"])

                result = subprocess.run(cmd, cwd=PROJECT_DIR)

                if result.returncode != 0:
                    print("[ERROR] execution failed")
                    print(f"  dataset : {dataset}")
                    print(f"  category: {category}")
                    print(f"  model   : {model}")
                    return

    print("\n" + "=" * 80)
    print("[FINISHED] All experiments completed!")
    print("=" * 80)


if __name__ == "__main__":

    run(SCRIPT_FILE, DATASET_LIST, CATEGORY_LIST, MODEL_LIST)
