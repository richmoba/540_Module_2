import subprocess # subprocess module
import os     # os module

def run_command(command):    # Function to run a command
    result = subprocess.run(command, shell=True)    # Run the command
    if result.returncode != 0:  # If the return code is not 0
        raise Exception(f"Command '{command}' failed with return code {result.returncode}")   # Raise an exception

commands = [    # List of commands to run
    "python scripts/collect_data.py",   # Collect data
    "python scripts/preprocess_data.py",    # Preprocess data
    "python scripts/classical_model.py",    # Run classical model
    "python scripts/deep_learning_model.py",    # Run deep learning model
    "streamlit run app.py"  # Run Streamlit app
]

for command in commands:    # Loop through the commands
    print(f"Running command: {command}")    # Print the command
    run_command(command)    # Run the command
