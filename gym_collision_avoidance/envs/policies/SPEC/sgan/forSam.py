import torch
import scnn.model as model
import scnn.utils as utils

import argparse
import logging
import sys
import numpy as np

#model = np.load("model.npy",allow_pickle=True)[0]

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


''' Parameter '''
# Dataset options
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--min_ped', default=2, type=int)
parser.add_argument('--hist_len', default=8, type=int)
parser.add_argument('--fut_len', default=12, type=int)
parser.add_argument('--loadNpy', default=1, type=int)
parser.add_argument('--untracked_ratio', default=1.0, type=float)
# Network design
parser.add_argument('--l2d', default=1, type=int)
parser.add_argument('--tanh', default=0, type=int)
parser.add_argument('--n_ch', default=2, type=int)
parser.add_argument('--use_max', default=1, type=int)
parser.add_argument('--targ_ker_num', default=[], type=list) # [7,28]
parser.add_argument('--targ_ker_size', default=[], type=list)
parser.add_argument('--targ_pool_size', default=[2,2], type=list)
parser.add_argument('--cont_ker_num', default=[-1], type=list) # [17,72]
parser.add_argument('--cont_ker_size', default=[2,2], type=list)
parser.add_argument('--cont_pool_size', default=[2,1], type=list)
parser.add_argument('--n_fc', default=-1, type=int)
parser.add_argument('--fc_width', default=[20], type=list) # 280,200,120,80
parser.add_argument('--output_size', default=5, type=int)
parser.add_argument('--neighbor', default=1, type=int)
parser.add_argument('--drop_rate', default=0.0, type=float)
parser.add_argument('--lock_l2d', default=0, type=int)
# Training
parser.add_argument('--loadModel', default='', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--n_epoch', default=1000, type=int)
parser.add_argument('--n_iteration', default=300, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--start', default=0, type=int)
# Validation and Output
parser.add_argument('--batch_size_val', default=2, type=int)
parser.add_argument('--batch_size_tst', default=2, type=int)
parser.add_argument('--n_batch_val', default=6, type=int)
parser.add_argument('--n_batch_tst', default=4, type=int)
parser.add_argument('--val_freq', default=1, type=int)
parser.add_argument('--n_guess', default=2, type=int)
parser.add_argument('--n_sample', default=20, type=int)
parser.add_argument('--coef', default=1.000000001, type=float)

args = parser.parse_args()


N = 23
hist = torch.arange(N*2*8).view(N,2,8)*1.0 #randomly generate some data in correct shape
fut = model.LocPredictor(args).predictTraj(hist)


##N = 3
##
##data = np.array( [ [[1,1],[2,2],[3,3]] , [[2,2],[2,2],[3,3]], [[3,3],[2,2],[3,3]], [[4,4],[2,2],[3,3]], [[5,5],[2,2],[3,3]], [[6,6],[2,2],[3,3]], [[7,7],[2,2],[3,3]], [[8,8],[2,2],[3,3]] ] )
##data = torch.from_numpy(np.transpose(data, (1, 2, 0))).double()
###hist = torch.arange(N*2*8).view(N,2,8)*1.0 #randomly generate some data in correct shape
##fut = model.LocPredictor(args).predictTraj(data)

#print(fut)



