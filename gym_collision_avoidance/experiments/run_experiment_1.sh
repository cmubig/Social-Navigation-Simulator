#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/utils.sh

# Train tf 
print_header "Running Experiment 1 (Self-play)"

# # Comment for using GPU
 export CUDA_VISIBLE_DEVICES=0
#-1

# Experiment
cd $DIR

#python3 src/run_experiment_1.py --output_name "exp1_ETH_CADRL" --experiment_num 1 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 10 --dataset_name "ETH"

#python3 src/run_experiment_1.py --output_name "exp1_UNIV_CADRL" --experiment_num 1 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 10 --dataset_name "UNIV"

#ETH
python3 src/run_experiment_1.py --output_name "exp1_ETH_CADRL" --experiment_num 1 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --dataset_name "ETH"
python3 src/run_experiment_1.py --output_name "exp1_ETH_RVO"   --experiment_num 1 --algorithm_name "RVO"   --experiment_iteration 3 --timeout 60 --dataset_name "ETH"
python3 src/run_experiment_1.py --output_name "exp1_ETH_SOCIALFORCE" --experiment_num 1 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --dataset_name "ETH"
python3 src/run_experiment_1.py --output_name "exp1_ETH_LINEAR" --experiment_num 1 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --dataset_name "ETH"

python3 src/run_experiment_1.py --output_name "exp1_ETH_SLSTM" --experiment_num 1 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --dataset_name "ETH"
python3 src/run_experiment_1.py --output_name "exp1_ETH_SPEC"  --experiment_num 1 --algorithm_name "SPEC"  --experiment_iteration 3 --timeout 60 --dataset_name "ETH"
python3 src/run_experiment_1.py --output_name "exp1_ETH_CVM"   --experiment_num 1 --algorithm_name "CVM"   --experiment_iteration 3 --timeout 60 --dataset_name "ETH"
python3 src/run_experiment_1.py --output_name "exp1_ETH_SOCIALGAN"  --experiment_num 1 --algorithm_name "SOCIALGAN"  --experiment_iteration 3 --timeout 60 --dataset_name "ETH"
python3 src/run_experiment_1.py --output_name "exp1_ETH_STGCNN"  --experiment_num 1 --algorithm_name "STGCNN"  --experiment_iteration 3 --timeout 60 --dataset_name "ETH"


#HOTEL
python3 src/run_experiment_1.py --output_name "exp1_HOTEL_CADRL" --experiment_num 1 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --dataset_name "HOTEL"
python3 src/run_experiment_1.py --output_name "exp1_HOTEL_RVO"   --experiment_num 1 --algorithm_name "RVO"   --experiment_iteration 3 --timeout 60 --dataset_name "HOTEL"
python3 src/run_experiment_1.py --output_name "exp1_HOTEL_SOCIALFORCE" --experiment_num 1 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --dataset_name "HOTEL"
python3 src/run_experiment_1.py --output_name "exp1_HOTEL_LINEAR" --experiment_num 1 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --dataset_name "HOTEL"

python3 src/run_experiment_1.py --output_name "exp1_HOTEL_SLSTM" --experiment_num 1 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --dataset_name "HOTEL"
python3 src/run_experiment_1.py --output_name "exp1_HOTEL_SPEC"  --experiment_num 1 --algorithm_name "SPEC"  --experiment_iteration 3 --timeout 60 --dataset_name "HOTEL"
python3 src/run_experiment_1.py --output_name "exp1_HOTEL_CVM"   --experiment_num 1 --algorithm_name "CVM"   --experiment_iteration 3 --timeout 60 --dataset_name "HOTEL"
python3 src/run_experiment_1.py --output_name "exp1_HOTEL_SOCIALGAN"  --experiment_num 1 --algorithm_name "SOCIALGAN"  --experiment_iteration 3 --timeout 60 --dataset_name "HOTEL"
python3 src/run_experiment_1.py --output_name "exp1_HOTEL_STGCNN"  --experiment_num 1 --algorithm_name "STGCNN"  --experiment_iteration 3 --timeout 60 --dataset_name "HOTEL"


#UNIV
python3 src/run_experiment_1.py --output_name "exp1_UNIV_CADRL" --experiment_num 1 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --dataset_name "UNIV"
python3 src/run_experiment_1.py --output_name "exp1_UNIV_RVO"   --experiment_num 1 --algorithm_name "RVO"   --experiment_iteration 3 --timeout 60 --dataset_name "UNIV"
python3 src/run_experiment_1.py --output_name "exp1_UNIV_SOCIALFORCE" --experiment_num 1 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --dataset_name "UNIV"
python3 src/run_experiment_1.py --output_name "exp1_UNIV_LINEAR" --experiment_num 1 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --dataset_name "UNIV"

python3 src/run_experiment_1.py --output_name "exp1_UNIV_SLSTM" --experiment_num 1 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --dataset_name "UNIV"
python3 src/run_experiment_1.py --output_name "exp1_UNIV_SPEC"  --experiment_num 1 --algorithm_name "SPEC"  --experiment_iteration 3 --timeout 60 --dataset_name "UNIV"
python3 src/run_experiment_1.py --output_name "exp1_UNIV_CVM"   --experiment_num 1 --algorithm_name "CVM"   --experiment_iteration 3 --timeout 60 --dataset_name "UNIV"
python3 src/run_experiment_1.py --output_name "exp1_UNIV_SOCIALGAN"  --experiment_num 1 --algorithm_name "SOCIALGAN"  --experiment_iteration 3 --timeout 60 --dataset_name "UNIV"
python3 src/run_experiment_1.py --output_name "exp1_UNIV_STGCNN"  --experiment_num 1 --algorithm_name "STGCNN"  --experiment_iteration 3 --timeout 60 --dataset_name "UNIV"


#ZARA1
python3 src/run_experiment_1.py --output_name "exp1_ZARA1_CADRL" --experiment_num 1 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --dataset_name "ZARA1"
python3 src/run_experiment_1.py --output_name "exp1_ZARA1_RVO"   --experiment_num 1 --algorithm_name "RVO"   --experiment_iteration 3 --timeout 60 --dataset_name "ZARA1"
python3 src/run_experiment_1.py --output_name "exp1_ZARA1_SOCIALFORCE" --experiment_num 1 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --dataset_name "ZARA1"
python3 src/run_experiment_1.py --output_name "exp1_ZARA1_LINEAR" --experiment_num 1 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --dataset_name "ZARA1"

python3 src/run_experiment_1.py --output_name "exp1_ZARA1_SLSTM" --experiment_num 1 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --dataset_name "ZARA1"
python3 src/run_experiment_1.py --output_name "exp1_ZARA1_SPEC"  --experiment_num 1 --algorithm_name "SPEC"  --experiment_iteration 3 --timeout 60 --dataset_name "ZARA1"
python3 src/run_experiment_1.py --output_name "exp1_ZARA1_CVM"   --experiment_num 1 --algorithm_name "CVM"   --experiment_iteration 3 --timeout 60 --dataset_name "ZARA1"
python3 src/run_experiment_1.py --output_name "exp1_ZARA1_SOCIALGAN"  --experiment_num 1 --algorithm_name "SOCIALGAN"  --experiment_iteration 3 --timeout 60 --dataset_name "ZARA1"
python3 src/run_experiment_1.py --output_name "exp1_ZARA1_STGCNN"  --experiment_num 1 --algorithm_name "STGCNN"  --experiment_iteration 3 --timeout 60 --dataset_name "ZARA1"


#ZARA2
python3 src/run_experiment_1.py --output_name "exp1_ZARA2_CADRL" --experiment_num 1 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --dataset_name "ZARA2"
python3 src/run_experiment_1.py --output_name "exp1_ZARA2_RVO"   --experiment_num 1 --algorithm_name "RVO"   --experiment_iteration 3 --timeout 60 --dataset_name "ZARA2"
python3 src/run_experiment_1.py --output_name "exp1_ZARA2_SOCIALFORCE" --experiment_num 1 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --dataset_name "ZARA2"
python3 src/run_experiment_1.py --output_name "exp1_ZARA2_LINEAR" --experiment_num 1 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --dataset_name "ZARA2"

python3 src/run_experiment_1.py --output_name "exp1_ZARA2_SLSTM" --experiment_num 1 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --dataset_name "ZARA2"
python3 src/run_experiment_1.py --output_name "exp1_ZARA2_SPEC"  --experiment_num 1 --algorithm_name "SPEC"  --experiment_iteration 3 --timeout 60 --dataset_name "ZARA2"
python3 src/run_experiment_1.py --output_name "exp1_ZARA2_CVM"   --experiment_num 1 --algorithm_name "CVM"   --experiment_iteration 3 --timeout 60 --dataset_name "ZARA2"
python3 src/run_experiment_1.py --output_name "exp1_ZARA2_SOCIALGAN"  --experiment_num 1 --algorithm_name "SOCIALGAN"  --experiment_iteration 3 --timeout 60 --dataset_name "ZARA2"
python3 src/run_experiment_1.py --output_name "exp1_ZARA2_STGCNN"  --experiment_num 1 --algorithm_name "STGCNN"  --experiment_iteration 3 --timeout 60 --dataset_name "ZARA2"
