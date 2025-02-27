### This is a script that runs the gpu accelerated model testing ###

#!/bin/bash

# Prompt user for the API key and instance details
read -p "Enter the name of your lambda API key (e.g. niel_lambda_api_key): " user_lambda_api_key_name
USER_LAMBDA_API_KEY=$(eval echo \$$user_lambda_api_key_name)
read -p "Enter the directory location of your private SSH key: " private_ssh_key
read -p "Enter the SSH user (e.g. ubuntu): " remote_ssh_user
read -p "Enter the SSH host/instance address (e.g. 129.146.33.218): " remote_ssh_host

# Copy testing scripts to remote instance
cd ../
LC_GPU_ACCELERATED_TRAINING_DIR="./lc_gpu_accelerated_training/"
NLC_GPU_ACCELERATED_TRAINING_DIR="./nlc_gpu_accelerated_training/"
MODEL_TEST_DIR="./model_tests/"
LC_TRAINED_MODELS_DIR="./lc_trained_ddpm/"
NLC_TRAINED_MODELS_DIR="./nlc_trained_ddpm/"

read -p "Would you like to ssh into the instance to first copy the gpu-accelerated model testing scripts to the cluster? [y/n]: " SSH_CONNECT_1
if [[ $SSH_CONNECT_1 == "y" ]]; then

    # SSH into the instance and copy the training scripts
    echo "Connecting to SSH..."
    scp -i "$private_ssh_key" -r "$LC_GPU_ACCELERATED_TRAINING_DIR" "$remote_ssh_user@$remote_ssh_host:/home/$remote_ssh_user/"
    scp -i "$private_ssh_key" -r "$NLC_GPU_ACCELERATED_TRAINING_DIR" "$remote_ssh_user@$remote_ssh_host:/home/$remote_ssh_user/"
    scp -i "$private_ssh_key" -r "$MODEL_TEST_DIR" "$remote_ssh_user@$remote_ssh_host:/home/$remote_ssh_user/"
    scp -i "$private_ssh_key" -r "$LC_TRAINED_MODELS_DIR" "$remote_ssh_user@$remote_ssh_host:/home/$remote_ssh_user/"
    scp -i "$private_ssh_key" -r "$NLC_TRAINED_MODELS_DIR" "$remote_ssh_user@$remote_ssh_host:/home/$remote_ssh_user/"
else
    echo "Skipping files copy into the cluster"
fi

# Install requirements
read -p "Would you like to ssh into the instance to install the requirements for model testing? [y/n]: " SSH_CONNECT_2
if [[ $SSH_CONNECT_2 == "y" ]]; then

    # Set up virtual environment
    echo "Setting up virtual environment..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "python3 -m venv .venv"
    echo "Virtual environment setup complete"

    echo "Setting up virtual environment and installing requirements..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" << EOF
    source .venv/bin/activate
    pip install -r model_tests/requirements.txt
EOF
    echo "Requirements installation complete"
else
    echo "Skipping requirements installation"
fi

# Run the model testing script
read -p "Would you like to ssh into the instance to run the model testing script? [y/n]: " SSH_CONNECT_2
if [[ $SSH_CONNECT_2 == "y" ]]; then

    # SSH into the instance and run the testing script in the backgrounc
    echo "Running model testing script..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "nohup bash -c 'source .venv/bin/activate && python model_tests/test_models.py' > test.log 2>&1 &" &

    echo "Model testing script is running in the background on the remote server."
else
    echo "Skipping model testing script execution"
fi