# run
# CUDA_VISIBLE_DEVICES="1" python -i scnn/pg.py

import argparse
import gc
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import data_loader

import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep

import scnn.model as model
import scnn.utils as utils

import datetime
_TIME = str(datetime.datetime.now())[5:19]

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

gpu, cpu = "cuda:0", "cpu"


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
parser.add_argument('--cont_ker_num', default=[-1,26], type=list) # [17,72]
parser.add_argument('--cont_ker_size', default=[2,2], type=list)
parser.add_argument('--cont_pool_size', default=[2,1], type=list)
parser.add_argument('--n_fc', default=-1, type=int)
parser.add_argument('--fc_width', default=[25,12], type=list) # 280,200,120,80
parser.add_argument('--output_size', default=5, type=int)
parser.add_argument('--neighbor', default=1, type=int)
parser.add_argument('--drop_rate', default=0.0, type=float)
parser.add_argument('--lock_l2d', default=0, type=int)
# Training
parser.add_argument('--loadModel', default='', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--n_epoch', default=1000, type=int)
parser.add_argument('--n_iteration', default=30, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--start', default=0, type=int)
# Validation and Output
parser.add_argument('--batch_size_val', default=1, type=int)
parser.add_argument('--batch_size_tst', default=1, type=int)
parser.add_argument('--n_batch_trn', default=3, type=int)
parser.add_argument('--n_batch_val', default=3, type=int)
parser.add_argument('--n_batch_tst', default=4, type=int)
parser.add_argument('--val_freq', default=10, type=int)
parser.add_argument('--n_guess', default=2, type=int)
parser.add_argument('--n_sample', default=20, type=int)
parser.add_argument('--coef', default=1.000000001, type=float)

args = parser.parse_args()

''' Load data '''
npy_path = '/home/dzhao/proj/scnn/datasets/'+args.dataset_name+'_loaders.npy'
if args.loadNpy:
    logger.info("loading data from npy ...")
    loader_trn, loader_val, loader_tst = np.load(npy_path,allow_pickle=True)
else:
    data_path = '/home/dzhao/Dropbox/sonav/sgan/data/'+args.dataset_name
    logger.info("reading data from txt ...")
    loader_trn, loader_val, loader_tst = data_loader(args, data_path, batch_size=(args.batch_size,args.batch_size_val,args.batch_size_tst), shuffle=False)
    np.save(npy_path,[loader_trn, loader_val, loader_tst])
    print('npy saved!')

predictor = np.load(args.loadModel+'.npy',allow_pickle=True)[0] if args.loadModel else model.LocPredictor(args).to(gpu)


w = np.zeros(predictor.c_conv[0].weight.shape[0])
N = len(loader_trn)

# for i, b in enumerate(loader_trn):
#     print(i,'/',N,end='\r')
#     th, ch, ei, tfl = b
#     # wi = predictor.c_conv[0](ch.to(gpu)).detach().to(cpu).sum(dim=2).sum(dim=0)
#     wi = (predictor.c_conv[0](ch.to(gpu)).detach().to(cpu)[:,:,-1]).sum(dim=0)
#     w += wi.numpy()
#
# w = utils.remap(w)
#
# a,b=np.load("traj_sample.npy",allow_pickle=True)
# # utils.plotWeights(predictor.c_conv[0].weight.detach().to(cpu),targ_traj=a)
# utils.plotWeights(predictor.c_conv[0].weight.detach().to(cpu),targ_traj=a,fn=args.dataset_name,W=w)



# for i, b in enumerate(loader_trn):
#     print(i,'/',N,end='\n')
#     th, ch, ei, tfl = b
#     for j in range(64):
#         print(j,end='\r')
#         sta, end = ei[j], ei[j+1]
#         if end-sta>=1:
#             # wi = predictor.c_conv[0](ch[sta:end].to(gpu)).detach().to(cpu).sum(dim=2).sum(dim=0)
#             wi = (predictor.c_conv[0](ch[sta:end].to(gpu)).detach().to(cpu)[:,:,-1]).max(dim=0).values
#             wi = utils.remap(wi)
#             utils.plotWeights(predictor.c_conv[0].weight.detach().to(cpu),targ_traj=th[j],Traj=ch[sta:end],fn=args.dataset_name+'/step'+str(j),W=wi)
#     break




for i, b in enumerate(loader_trn):
    print(i,'/',N,end='\n')
    th, ch, ei, tfl = b
    for t in range(8):
        j = 34
        print(j,end='\r')
        sta, end = ei[j], ei[j+1]
        if end-sta>=1:
            # wi = predictor.c_conv[0](ch[sta:end].to(gpu)).detach().to(cpu).sum(dim=2).sum(dim=0)
            wi = (predictor.c_conv[0](ch[sta:end].to(gpu)).detach().to(cpu)[:,:,t]).max(dim=0).values
            wi = utils.remap(wi)
            utils.plotWeights(predictor.c_conv[0].weight.detach().to(cpu),targ_traj=th[j],Traj=ch[sta:end],fn=args.dataset_name+'/step'+str(t),W=wi)
    break






#
