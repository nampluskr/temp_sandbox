# experiments/run.py

#####################################################################
# Experiment split lists
#####################################################################

DATASET_LIST = ["mvtec"]
CATEGORY_LIST = {
    # "mvtec": ["carpet", "grid", "leather", "tile", "wood"],  # texture
    "mvtec": ["bottle"] # test category
}
# MODEL_LIST = ["stfpm"]
# MODEL_LIST = ["efficientad"]
# MODEL_LIST = ["reversedistill"]
# MODEL_LIST = ["efficientad"]
# MODEL_LIST = ["reversedistill", "efficientad", "stfpm"]
# MODEL_LIST = ["cflow", "fastflow", "csflow", "uflow"]
MODEL_LIST = ["dinomaly"]

MAX_EPOCHS = 5
SAVE_MODEL = False
VALIDATE = False

#####################################################################
# Script file path (absolute)
#####################################################################

import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_FILE = os.path.join(PROJECT_DIR, "experiments", "train.py")


#####################################################################
# Main function
#####################################################################

def main(script_file, dataset_list, category_list, model_list):
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

                print("\n" + "=" * 80)
                print(f"[RUN {counter}/{total}] dataset: {dataset} | category: {category} | model: {model} "
                      f" ({MAX_EPOCHS} epochs)"
                )
                print("=" * 80)

                cmd = [sys.executable, script_file]
                cmd.extend(["--dataset", dataset])
                cmd.extend(["--model", model])
                cmd.extend(["--category", category])
                cmd.extend(["--max_epochs", str(MAX_EPOCHS)])

                if SAVE_MODEL:
                    cmd.extend(["--save_model"])

                if VALIDATE:
                    cmd.extend(["--validate"])

                result = subprocess.run(cmd, cwd=PROJECT_DIR)

                if result.returncode != 0:
                    print("[ERROR] execution failed")
                    print(f"  dataset : {dataset}")
                    print(f"  category: {category}")
                    print(f"  model   : {model}")
                    return

    print("\n" + "=" * 70)
    print("[FINISHED] All experiments completed!")
    print("=" * 70)

if __name__ == "__main__":

    main(SCRIPT_FILE, DATASET_LIST, CATEGORY_LIST, MODEL_LIST)
