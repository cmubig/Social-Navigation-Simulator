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



#########################################################################
python3 src/run_experiment_1.py --output_name "exp2_0.1_CADRL" --experiment_num 2 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_CADRL" --experiment_num 2 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_CADRL" --experiment_num 2 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_CADRL" --experiment_num 2 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_CADRL" --experiment_num 2 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_CADRL" --experiment_num 2 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_CADRL" --experiment_num 2 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_CADRL" --experiment_num 2 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_CADRL" --experiment_num 2 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_CADRL" --experiment_num 2 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 60 --population_density 1.0

########################################################################
#########################################################################
python3 src/run_experiment_1.py --output_name "exp2_0.1_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 3 --timeout 60 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 3 --timeout 60 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 3 --timeout 60 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 3 --timeout 60 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 3 --timeout 60 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 3 --timeout 60 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 3 --timeout 60 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 3 --timeout 60 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 3 --timeout 60 --population_density 1.0

########################################################################
#########################################################################
python3 src/run_experiment_1.py --output_name "exp2_0.1_SOCIALFORCE" --experiment_num 2 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_SOCIALFORCE" --experiment_num 2 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_SOCIALFORCE" --experiment_num 2 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_SOCIALFORCE" --experiment_num 2 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_SOCIALFORCE" --experiment_num 2 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_SOCIALFORCE" --experiment_num 2 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_SOCIALFORCE" --experiment_num 2 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_SOCIALFORCE" --experiment_num 2 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_SOCIALFORCE" --experiment_num 2 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_SOCIALFORCE" --experiment_num 2 --algorithm_name "SOCIALFORCE" --experiment_iteration 3 --timeout 60 --population_density 1.0

########################################################################
#########################################################################
python3 src/run_experiment_1.py --output_name "exp2_0.1_LINEAR" --experiment_num 2 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_LINEAR" --experiment_num 2 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_LINEAR" --experiment_num 2 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_LINEAR" --experiment_num 2 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_LINEAR" --experiment_num 2 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_LINEAR" --experiment_num 2 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_LINEAR" --experiment_num 2 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_LINEAR" --experiment_num 2 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_LINEAR" --experiment_num 2 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_LINEAR" --experiment_num 2 --algorithm_name "LINEAR" --experiment_iteration 3 --timeout 60 --population_density 1.0

########################################################################
#########################################################################
python3 src/run_experiment_1.py --output_name "exp2_0.1_SLSTM" --experiment_num 2 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_SLSTM" --experiment_num 2 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_SLSTM" --experiment_num 2 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_SLSTM" --experiment_num 2 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_SLSTM" --experiment_num 2 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_SLSTM" --experiment_num 2 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_SLSTM" --experiment_num 2 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_SLSTM" --experiment_num 2 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_SLSTM" --experiment_num 2 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_SLSTM" --experiment_num 2 --algorithm_name "SLSTM" --experiment_iteration 3 --timeout 60 --population_density 1.0

########################################################################
#########################################################################
python3 src/run_experiment_1.py --output_name "exp2_0.1_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 3 --timeout 60 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 3 --timeout 60 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 3 --timeout 60 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 3 --timeout 60 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 3 --timeout 60 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 3 --timeout 60 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 3 --timeout 60 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 3 --timeout 60 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 3 --timeout 60 --population_density 1.0

########################################################################
#########################################################################
python3 src/run_experiment_1.py --output_name "exp2_0.1_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 3 --timeout 60 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 3 --timeout 60 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 3 --timeout 60 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 3 --timeout 60 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 3 --timeout 60 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 3 --timeout 60 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 3 --timeout 60 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 3 --timeout 60 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 3 --timeout 60 --population_density 1.0

########################################################################
#########################################################################
python3 src/run_experiment_1.py --output_name "exp2_0.1_SPEC" --experiment_num 2 --algorithm_name "SPEC" --experiment_iteration 3 --timeout 60 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_SPEC" --experiment_num 2 --algorithm_name "SPEC" --experiment_iteration 3 --timeout 60 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_SPEC" --experiment_num 2 --algorithm_name "SPEC" --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_SPEC" --experiment_num 2 --algorithm_name "SPEC" --experiment_iteration 3 --timeout 60 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_SPEC" --experiment_num 2 --algorithm_name "SPEC" --experiment_iteration 3 --timeout 60 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_SPEC" --experiment_num 2 --algorithm_name "SPEC" --experiment_iteration 3 --timeout 60 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_SPEC" --experiment_num 2 --algorithm_name "SPEC" --experiment_iteration 3 --timeout 60 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_SPEC" --experiment_num 2 --algorithm_name "SPEC" --experiment_iteration 3 --timeout 60 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_SPEC" --experiment_num 2 --algorithm_name "SPEC" --experiment_iteration 3 --timeout 60 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_SPEC" --experiment_num 2 --algorithm_name "SPEC" --experiment_iteration 3 --timeout 60 --population_density 1.0

########################################################################
#########################################################################
python3 src/run_experiment_1.py --output_name "exp2_0.1_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 3 --timeout 60 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 3 --timeout 60 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 3 --timeout 60 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 3 --timeout 60 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 3 --timeout 60 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 3 --timeout 60 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 3 --timeout 60 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 3 --timeout 60 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 3 --timeout 60 --population_density 1.0

########################################################################

