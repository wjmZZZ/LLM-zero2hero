#!/bin/bash

# Read configuration file
config_file="./configs/dpo_cfg.json"

# Default number of GPUs
default_num_gpus=1

# Variables for specifying GPUs and number of GPUs to use
specified_gpus=""
num_gpus=$default_num_gpus

# Check if the file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file $config_file does not exist"
    exit 1
fi

# Parse JSON file and extract parameters
use_deepspeed=$(python -c "import json; print(json.load(open('$config_file'))['EnvironmentArguments']['use_deepspeed'])")

# Check for command line arguments
while [ $# -gt 0 ]; do
    case $1 in
        -g|--gpus)
            specified_gpus="$2"
            shift 2
            ;;
        -n|--num_gpus)
            num_gpus="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Generate GPU list based on specified GPUs or num_gpus
if [ -n "$specified_gpus" ]; then
    gpu_list=$specified_gpus
    num_gpus=$(echo $specified_gpus | tr ',' '\n' | wc -l)
    echo "Using specified GPUs: $gpu_list"
else
    gpu_list=$(seq -s, 0 $((num_gpus-1)))
    echo "No GPUs specified, using first $num_gpus GPUs: $gpu_list"
fi

echo "Number of GPUs: $num_gpus"
echo "GPU list: $gpu_list"

# Set the running command based on parameters
if [ "$use_deepspeed" = "True" ]; then
    run_command="deepspeed --include localhost:$gpu_list --master_port 15555 src/Main.py $config_file"
    echo "Running with DeepSpeed"
elif [ "$num_gpus" -gt 1 ]; then
    run_command="CUDA_VISIBLE_DEVICES=$gpu_list torchrun --nproc_per_node=$num_gpus --master_port=15200 src/Main.py $config_file"
    echo "Running in multi-GPU distributed mode"
else
    run_command="CUDA_VISIBLE_DEVICES=$gpu_list python src/Main.py $config_file"
    echo "Running in single GPU mode"
fi

# Print the command that will be executed
echo "-------------------------------------"    
echo "$run_command"
echo "-------------------------------------"
# Execute the command
eval $run_command