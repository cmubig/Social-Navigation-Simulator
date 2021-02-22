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

#################################Navigation
#########################################################################CADRL
python3 src/run_experiment_1.py --output_name "exp3_0.3_CADRL_RVO" --experiment_num 3 --algorithm_name ["CADRL","RVO"] --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_CADRL_LINEAR" --experiment_num 3 --algorithm_name ["CADRL","LINEAR"] --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_CADRL_SOCIALFORCE" --experiment_num 3 --algorithm_name ["CADRL","SOCIALFORCE"] --experiment_iteration 3 --timeout 60 --population_density 0.3

########################################################################SOCIALFORCE
python3 src/run_experiment_1.py --output_name "exp3_0.3_SOCIALFORCE_CADRL" --experiment_num 3 --algorithm_name ["SOCIALFORCE","CADRL"] --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_SOCIALFORCE_RVO" --experiment_num 3 --algorithm_name ["SOCIALFORCE","RVO"] --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_SOCIALFORCE_LINEAR" --experiment_num 3 --algorithm_name ["SOCIALFORCE","LINEAR"] --experiment_iteration 3 --timeout 60 --population_density 0.3

########################################################################RVO
python3 src/run_experiment_1.py --output_name "exp3_0.3_RVO_CADRL" --experiment_num 3 --algorithm_name ["RVO","CADRL"] --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_RVO_LINEAR" --experiment_num 3 --algorithm_name ["RVO","LINEAR"] --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_RVO_SOCIALFORCE" --experiment_num 3 --algorithm_name ["RVO","SOCIALFORCE"] --experiment_iteration 3 --timeout 60 --population_density 0.3

########################################################################LINEAR
python3 src/run_experiment_1.py --output_name "exp3_0.3_LINEAR_CADRL" --experiment_num 3 --algorithm_name ["LINEAR","CADRL"] --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_LINEAR_RVO" --experiment_num 3 --algorithm_name ["LINEAR","RVO"] --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_LINEAR_SOCIALFORCE" --experiment_num 3 --algorithm_name ["LINEAR","SOCIALFORCE"] --experiment_iteration 3 --timeout 60 --population_density 0.3


##########################################

python3 src/run_experiment_1.py --output_name "exp4_0.3_CADRL_RVO" --experiment_num 4 --algorithm_name ["CADRL","RVO"] --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp4_0.3_CADRL_LINEAR" --experiment_num 4 --algorithm_name ["CADRL","LINEAR"] --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp4_0.3_CADRL_SOCIALFORCE" --experiment_num 4 --algorithm_name ["CADRL","SOCIALFORCE"] --experiment_iteration 3 --timeout 60 --population_density 0.3

########################################################################SOCIALFORCE

python3 src/run_experiment_1.py --output_name "exp4_0.3_SOCIALFORCE_RVO" --experiment_num 4 --algorithm_name ["SOCIALFORCE","RVO"] --experiment_iteration 3 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp4_0.3_SOCIALFORCE_LINEAR" --experiment_num 4 --algorithm_name ["SOCIALFORCE","LINEAR"] --experiment_iteration 3 --timeout 60 --population_density 0.3

########################################################################RVO

python3 src/run_experiment_1.py --output_name "exp4_0.3_RVO_LINEAR" --experiment_num 4 --algorithm_name ["RVO","LINEAR"] --experiment_iteration 3 --timeout 60 --population_density 0.3


################################Motion

###############################
############################################################################## EXP 3
########################################################################CVM
python3 src/run_experiment_1.py --output_name "exp3_0.3_CVM_SLSTM" --experiment_num 3 --algorithm_name ["CVM","SLSTM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_CVM_SPEC" --experiment_num 3 --algorithm_name ["CVM","SPEC"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_CVM_SOCIALGAN" --experiment_num 3 --algorithm_name ["CVM","SOCIALGAN"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_CVM_STGCNN" --experiment_num 3 --algorithm_name ["CVM","STGCNN"] --experiment_iteration 3 --timeout 15 --population_density 0.3


########################################################################SLSTM
python3 src/run_experiment_1.py --output_name "exp3_0.3_SLSTM_CVM" --experiment_num 3 --algorithm_name ["SLSTM","CVM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_SLSTM_SPEC" --experiment_num 3 --algorithm_name ["SLSTM","SPEC"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_SLSTM_SOCIALGAN" --experiment_num 3 --algorithm_name ["SLSTM","SOCIALGAN"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_SLSTM_STGCNN" --experiment_num 3 --algorithm_name ["SLSTM","STGCNN"] --experiment_iteration 3 --timeout 15 --population_density 0.3



########################################################################SPEC
python3 src/run_experiment_1.py --output_name "exp3_0.3_SPEC_CVM" --experiment_num 3 --algorithm_name ["SPEC","CVM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_SPEC_SLSTM" --experiment_num 3 --algorithm_name ["SPEC","SLSTM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_SPEC_SOCIALGAN" --experiment_num 3 --algorithm_name ["SPEC","SOCIALGAN"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_SPEC_STGCNN" --experiment_num 3 --algorithm_name ["SPEC","STGCNN"] --experiment_iteration 3 --timeout 15 --population_density 0.3

########################################################################SOCIALGAN
python3 src/run_experiment_1.py --output_name "exp3_0.3_SOCIALGAN_CVM" --experiment_num 3 --algorithm_name ["SOCIALGAN","CVM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_SOCIALGAN_SLSTM" --experiment_num 3 --algorithm_name ["SOCIALGAN","SLSTM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_SOCIALGAN_SPEC" --experiment_num 3 --algorithm_name ["SOCIALGAN","SPEC"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_SOCIALGAN_STGCNN" --experiment_num 3 --algorithm_name ["SOCIALGAN","STGCNN"] --experiment_iteration 3 --timeout 15 --population_density 0.3

########################################################################STGCNN
python3 src/run_experiment_1.py --output_name "exp3_0.3_STGCNN_CVM" --experiment_num 3 --algorithm_name ["STGCNN","CVM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_STGCNN_SLSTM" --experiment_num 3 --algorithm_name ["STGCNN","SLSTM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_STGCNN_SPEC" --experiment_num 3 --algorithm_name ["STGCNN","SPEC"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp3_0.3_STGCNN_SOCIALGAN" --experiment_num 3 --algorithm_name ["STGCNN","SOCIALGAN"] --experiment_iteration 3 --timeout 15 --population_density 0.3



##############################################################################
python3 src/run_experiment_1.py --output_name "exp4_0.3_CVM_SLSTM" --experiment_num 4 --algorithm_name ["CVM","SLSTM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp4_0.3_CVM_SPEC" --experiment_num 4 --algorithm_name ["CVM","SPEC"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp4_0.3_SLSTM_SPEC" --experiment_num 4 --algorithm_name ["SLSTM","SPEC"] --experiment_iteration 3 --timeout 15 --population_density 0.3

##########################################################################
python3 src/run_experiment_1.py --output_name "exp4_0.3_SOCIALGAN_CVM" --experiment_num 4 --algorithm_name ["SOCIALGAN","CVM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp4_0.3_SOCIALGAN_SLSTM" --experiment_num 4 --algorithm_name ["SOCIALGAN","SLSTM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp4_0.3_SOCIALGAN_SPEC" --experiment_num 4 --algorithm_name ["SOCIALGAN","SPEC"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp4_0.3_SOCIALGAN_STGCNN" --experiment_num 4 --algorithm_name ["SOCIALGAN","STGCNN"] --experiment_iteration 3 --timeout 15 --population_density 0.3

#########
python3 src/run_experiment_1.py --output_name "exp4_0.3_STGCNN_CVM" --experiment_num 4 --algorithm_name ["STGCNN","CVM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp4_0.3_STGCNN_SLSTM" --experiment_num 4 --algorithm_name ["STGCNN","SLSTM"] --experiment_iteration 3 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp4_0.3_STGCNN_SPEC" --experiment_num 4 --algorithm_name ["STGCNN","SPEC"] --experiment_iteration 3 --timeout 15 --population_density 0.3


