# experiments/run.py

#####################################################################

DATASET_LIST = ["mvtec"]
CATEGORY_LIST = {
    "mvtec": ["carpet", "grid", "leather", "tile", "wood"]  # texture
}
MODEL_LIST = ["efficientad"]

#####################################################################

import os
import sys
import subprocess

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_FILE = os.path.join(PROJECT_DIR, "experiments", "train.py")

if not os.path.isfile(SCRIPT_FILE):
    raise FileNotFoundError(SCRIPT_FILE)


def main(script_file, dataset_list, category_list, model_list):
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
                print("\n" + "=" * 70)
                print(f"[RUN {counter}/{total}] dataset: {dataset} | category: {category} | model: {model}")
                print("=" * 70)

                cmd = [
                    sys.executable, script_file,
                    "--dataset", dataset,
                    "--model", model,
                    "--category", category,
                ]

                result = subprocess.run(cmd, cwd=PROJECT_DIR)

                if result.returncode != 0:
                    print("[ERROR] execution failed")
                    print(f"  dataset : {dataset}")
                    print(f"  category: {category}")
                    print(f"  model   : {model}")
                    return

                print(f"\n[DONE {counter}/{total}] dataset: {dataset} | category: {category} | model: {model}")

    print("\n[FINISHED] all experiments completed")


if __name__ == "__main__":

    main(SCRIPT_FILE, DATASET_LIST, CATEGORY_LIST, MODEL_LIST)
