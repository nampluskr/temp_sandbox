import subprocess
import sys

def run(file_list):

    for i, file in enumerate(file_list, 1):
        print(f"\n>> [{i}/{len(file_list)}] Running: {file}\n")
        try:
            result = subprocess.run([sys.executable, file], check=True,  encoding="utf-8")
            print(f">> Completed: {file}")
        except subprocess.CalledProcessError:
            print(f"[Error] Script '{file}' failed during execution")
            break
        except FileNotFoundError:
            print(f"[Error] Script '{file}' not found")
            break

if __name__ == "__main__":

    file_list = [
        "./experiments/load_dataset_mvtec.py",
        "./experiments/load_dataset_visa.py",
        "./experiments/load_dataset_btad.py",
    ]

    run(file_list)
