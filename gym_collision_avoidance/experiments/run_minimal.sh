#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/utils.sh

# Train tf 
print_header "Running example python script"

# # Comment for using GPU
 export CUDA_VISIBLE_DEVICES=-1

# Experiment
cd $DIR
python src/minimal.py --experiment_num 1 --dataset_name ETH --timeout 60 --map ../envs/world_maps/001.png
python src/minimal.py --experiment_num 1 --dataset_name ETH --timeout 60 --map ../envs/world_maps/000.png


