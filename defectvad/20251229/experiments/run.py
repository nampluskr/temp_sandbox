# experiments/run.py

import os
import sys
import subprocess

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
script_file = os.path.join(project_dir, "experiments", "train.py")

if not os.path.isfile(script_file):
    raise FileNotFoundError(script_file)

dataset_list = ["mvtec"]
category_list = {"mvtec": ["carpet", "grid", "leather", "tile", "wood"]}    # texture
model_list = ["ganomaly"]


def main(script_file, dataset_list, category_list, model_name):
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

                result = subprocess.run(cmd, cwd=project_dir)

                if result.returncode != 0:
                    print("[ERROR] execution failed")
                    print(f"  dataset : {dataset}")
                    print(f"  category: {category}")
                    print(f"  model   : {model}")
                    return

                print(f"\n[DONE {counter}/{total}] dataset: {dataset} | category: {category} | model: {model}")

    print("\n[FINISHED] all experiments completed")


if __name__ == "__main__":

    main(script_file, dataset_list, category_list, model_list)
