import subprocess
import shutil
import os

def run_command(command):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command '{command}' failed with return code {result.returncode}")
        raise Exception(f"Command '{command}' failed with return code {result.returncode}")

commands = [
    "pip install -r requirements.txt",
    "python scripts/collect_data.py",
    "python scripts/preprocess_data.py",
    "python scripts/classical_model.py",
    "python scripts/deep_learning_model.py"
]

for command in commands:
    run_command(command)

print("Setup complete.")
