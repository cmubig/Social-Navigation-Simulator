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
import data_loader0

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
# if args.dataset_name=="univ": args.n_batch_tst=1
# if args.dataset_name=="zara2": args.n_batch_tst=2
gpu, cpu = "cuda:0", "cpu"




class CVM:
    def rel_to_abs(self,rel_traj, start_pos):
        rel_traj = rel_traj.permute(1, 0, 2)
        displacement = torch.cumsum(rel_traj, dim=1)
        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos
        return abs_traj.permute(1, 0, 2)

    def constant_velocity_model(self,observed, sample=False):
        """
        CVM can be run with or without sampling. A call to this function always
        generates one sample if sample option is true.
        """
        obs_rel = observed[1:] - observed[:-1]
        deltas = obs_rel[-1].unsqueeze(0)
        if sample:
            sampled_angle = np.random.normal(0, 25, 1)[0]
            theta = (sampled_angle * np.pi)/ 180.
            c, s = np.cos(theta), np.sin(theta)
            rotation_mat = torch.tensor([[c, s],[-s, c]]).to(gpu)
            deltas = torch.t(rotation_mat.matmul(torch.t(deltas.squeeze(dim=0)))).unsqueeze(0)
        y_pred_rel = deltas.repeat(12, 1, 1)
        return y_pred_rel

    def predictTraj(self,hist,ei):
        observed = hist.permute(2,0,1)
        y_pred_rel = self.constant_velocity_model(observed)
        y_pred_abs = self.rel_to_abs(y_pred_rel, observed[-1])
        return y_pred_abs.permute(1,2,0)

    def predictTrajSample(self,hist,ei):
        print('sample',end=',')
        observed = hist.permute(2,0,1)
        y_samples = []
        for i in range(20):
            print(i,end='\r')
            y_pred_rel = self.constant_velocity_model(observed,sample=True)
            y_pred_abs = self.rel_to_abs(y_pred_rel, observed[-1])
            y_samples.append(y_pred_abs.permute(1,2,0))
        return torch.stack(y_samples)


''' Load data '''
npy_path = '/home/dzhao/proj/scnn/datasets/'+args.dataset_name+'_loaders.npy'
if args.loadNpy:
    logger.info("loading data from npy ...")
    loader_trn, loader_val, loader_tst = np.load(npy_path,allow_pickle=True)
else:
    # data_path = '/home/dzhao/Dropbox/sonav/sgan/data/'+args.dataset_name
    # logger.info("reading data from txt ...")
    # loader_trn, loader_val, loader_tst = data_loader(args, data_path, batch_size=(args.batch_size,args.batch_size_val,args.batch_size_tst))
    # np.save(npy_path,[loader_trn, loader_val, loader_tst])
    trn_path = utils.get_dset_path(args.dataset_name, 'train')
    val_path = utils.get_dset_path(args.dataset_name, 'val')
    tst_path = utils.get_dset_path(args.dataset_name, 'test')
    loader_trn = data_loader0.data_loader(args, trn_path, fut_loc=True, batch_size=args.batch_size)
    loader_val = data_loader0.data_loader(args, val_path, fut_loc=False, batch_size=args.batch_size_val)
    loader_tst = data_loader0.data_loader(args, tst_path, fut_loc=False, batch_size=args.batch_size_tst)
    # np.save(npy_path,[loader_trn, loader_val, loader_tst])

num_sample = len(loader_trn.dataset)
iterations_per_epoch = num_sample/args.batch_size
args.n_iteration = min(args.n_iteration,iterations_per_epoch)
logger.info('{} samples in an epoch, {} iterations per epoch'.format(num_sample,iterations_per_epoch))
logger.info('{} epochs;  batch_size: {} '.format(args.n_epoch,args.batch_size))
print("max()"if args.use_max else "sum()")


cvm = CVM()
# utils.eval_model(cvm,loader_tst,determ=1,n_batch=args.n_batch_tst,repeat=True)
utils.eval_model(cvm,loader_tst,determ=0,n_batch=args.n_batch_tst,repeat=True)
while True: pass
