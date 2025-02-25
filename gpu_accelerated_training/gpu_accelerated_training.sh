### This is a script that runs the gpu_accelerated_training ###

#!/bin/bash

# Prompt user for the API key and instance details
read -p "Enter the name of your lambda API key (e.g. niel_lambda_api_key): " user_lambda_api_key_name
USER_LAMBDA_API_KEY=$(eval echo \$$user_lambda_api_key_name)
read -p "Enter the directory location of your private SSH key: " private_ssh_key
read -p "Enter the SSH user (e.g. ubuntu): " remote_ssh_user
read -p "Enter the SSH host/instance address (e.g. 129.146.33.218): " remote_ssh_host

# Copy training scripts to remote instance
MODEL_DEFINITION_PATH="./model.py"
UNET_ARCHITECTURE_PATH="./unet.py"
TRAINING_SCRIPT_PATH="./train_mnist.py"
REQUIREMENTS_PATH="./requirements.txt"
UTILS_PATH="./utils.py"

read -p "Would you like to ssh into the instance to first copy the gpu-accelerated training scripts to the cluster? [y/n]: " SSH_CONNECT_1
if [[ $SSH_CONNECT_1 == "y" ]]; then

    # SSH into the instance and copy the training scripts
    echo "Connecting to SSH..."
    scp -i "$private_ssh_key" $MODEL_DEFINITION_PATH "$remote_ssh_user@$remote_ssh_host:/home/$remote_ssh_user/"
    scp -i "$private_ssh_key" $UNET_ARCHITECTURE_PATH "$remote_ssh_user@$remote_ssh_host:/home/$remote_ssh_user/"
    scp -i "$private_ssh_key" $TRAINING_SCRIPT_PATH "$remote_ssh_user@$remote_ssh_host:/home/$remote_ssh_user/"
    scp -i "$private_ssh_key" $REQUIREMENTS_PATH "$remote_ssh_user@$remote_ssh_host:/home/$remote_ssh_user/"
    scp -i "$private_ssh_key" $UTILS_PATH "$remote_ssh_user@$remote_ssh_host:/home/$remote_ssh_user/"
else
    echo "Skipping files copy into the cluster"
fi

# Install requirements
read -p "Would you like to ssh into the instance to install the requirements for training? [y/n]: " SSH_CONNECT_2
if [[ $SSH_CONNECT_2 == "y" ]]; then

    # Set up virtual environment
    echo "Setting up virtual environment..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "python3 -m venv .venv"
    echo "Virtual environment setup complete"

    echo "Setting up virtual environment and installing requirements..."
ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" << EOF
    source .venv/bin/activate
    pip install -r requirements.txt
EOF
    echo "Requirements installation complete"
else
    echo "Skipping requirements installation"
fi

# Run the training script
read -p "Would you like to ssh into the instance to run the training script? [y/n]: " SSH_CONNECT_3
if [[ $SSH_CONNECT_3 == "y" ]]; then

    # SSH into the instance and run the training script in the background
    echo "Running training script..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "nohup bash -c 'source .venv/bin/activate && python train_mnist.py' > train.log 2>&1 &" &

    echo "Training script is running in the background on the remote server."
else
    echo "Skipping training script execution"
fi