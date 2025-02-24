'''
This script launches an instance 
'''

import json
import os
import subprocess
from dotenv import load_dotenv

# API Key
load_dotenv()
user_lambda_api_key_name = input("Enter the name of your lambda API key as it is saved in .env (e.g. niel_lambda_api_key): ")
user_lambda_ssh_key_name = input("Enter the name of your lambda API key as it is saved in Lambda Cloud (e.g. niel_lambda_ssh): ")
USER_LAMBDA_API_KEY = os.getenv(user_lambda_api_key_name)

check_instance = input("Would you like to check the instance types available? (y/n): ")
while check_instance.lower() not in ["y", "n"]:
    check_instance = input("Invalid input. Please enter 'y' or 'n': ")
if check_instance.lower() == "y":
    check_instance_types_command = [
        "curl",
        "-u", f'{USER_LAMBDA_API_KEY}:',
        "https://cloud.lambdalabs.com/api/v1/instance-types"
    ]
    result = subprocess.run(check_instance_types_command, capture_output=True, text=True)
    if result.returncode == 0:
        output = json.loads(result.stdout)
        print(json.dumps(output, indent=2))
    else:
        print(f"Error: {result.stderr}")
else:
    print("Skipping instance type check.")

# Get various information you need about the instance
print("You may need to get some of the following information by checking the lambda console.")
region_name = input("Enter the region name (e.g. us-west-2): ")
instance_type_name = input("Enter the instance type name (e.g. gpu_1x_a100_sxm4): ")
file_system_name = input("Enter the file system name (e.g. heft-fs-1). If none, just press enter: ")
quantity = input("Enter the quantity (e.g. 1). Default is 1 if you just press enter: ")
instance_name = input("Enter the instance name (e.g. my-instance). Default is null if you just press enter: ")

# Make request json

# Mandatory fields
request_dict = {
    "region_name": region_name,
    "instance_type_name": instance_type_name,
    "ssh_key_names": [
        user_lambda_ssh_key_name
    ]
}

if file_system_name != "":
    request_dict["file_system_names"] = [file_system_name]

if quantity != "":
    request_dict["quantity"] = int(quantity)

if instance_name != "":
    request_dict["name"] = instance_name

with open("request.json", "w") as json_file:
    json.dump(request_dict, json_file, indent=4)

# Make request
curl_command = [
    "curl",
    "-u", f'{USER_LAMBDA_API_KEY}:',
    "https://cloud.lambdalabs.com/api/v1/instance-operations/launch",
    "-d", f"@request.json", 
    "-H", "Content-Type: application/json"
]

# Run the command and capture the output
result = subprocess.run(curl_command, capture_output=True, text=True)

# Check if the command was successful
if result.returncode == 0:
    print(result.stdout)
else:
    print(f"Error: {result.stderr}")

# Delete the request file
os.remove("request.json")