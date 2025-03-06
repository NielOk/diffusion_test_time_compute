### This is a script that runs the gpu_low_labels_various_approaches_ema ###

#!/bin/bash

# Prompt user for the API key and instance details
#read -p "Enter the name of your lambda API key (e.g. niel_lambda_api_key): " user_lambda_api_key_name
#USER_LAMBDA_API_KEY=$(eval echo \$$user_lambda_api_key_name)
#read -p "Enter the directory location of your private SSH key: " private_ssh_key
#read -p "Enter the SSH user (e.g. ubuntu): " remote_ssh_user
#read -p "Enter the SSH host/instance address (e.g. 129.146.33.218): " remote_ssh_host

# Copy experiment scripts to remote instance
cd ../
EXPERIMENTS_REQUIREMENTS_PATH="./inference_experiments/inference_experiments_requirements.txt"
EXPERIMENT_PYTHON_SCRIPT_PATH="./inference_experiments/low_labels_various_approaches_ema.py"
