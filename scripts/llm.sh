#!/bin/bash

# Read configuration file
config_file="./configs/cfg.json"

# Specify the number of GPUs in the script
num_gpus=1


# Check if the file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file $config_file does not exist"
    exit 1
fi

# Parse JSON file and extract parameters
use_deepspeed=$(python -c "import json; print(json.load(open('$config_file'))['EnvironmentArguments']['use_deepspeed'])")



# Generate GPU list based on num_gpus
gpu_list=$(seq -s, 0 $((num_gpus-1)))

echo "GPU list: $gpu_list"

# Set the running command based on parameters
if [ "$use_deepspeed" = "True" ]; then
    run_command="deepspeed --include localhost:$gpu_list --master_port 15555 src/Main.py $config_file"
    echo "Running with DeepSpeed"
elif [ "$num_gpus" -gt 1 ]; then
    run_command="CUDA_VISIBLE_DEVICES=$gpu_list torchrun --nproc_per_node=$num_gpus --master_port=15200 src/Main.py $config_file"
    echo "Running in multi-GPU distributed mode"
else
    run_command="CUDA_VISIBLE_DEVICES=0 python src/Main.py $config_file"
    echo "Running in single GPU mode"
fi

# Print the command that will be executed
echo "-------------------------------------"    
echo "$run_command"
echo "-------------------------------------"
# Execute the command
eval $run_command