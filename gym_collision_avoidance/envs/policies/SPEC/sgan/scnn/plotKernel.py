
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
from time import time

import scnn.model as model
import scnn.utils as utils


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)
import datetime
_TIME = str(datetime.datetime.now())[5:19] # str(np.datetime64('now'))[5:]

parser.add_argument('--model', default='', type=str)
args = parser.parse_args()

gpu, cpu = "cuda:0", "cpu"
predictor = np.load(args.model,allow_pickle=True)[0]
predictor.ag.targ_pool_size, predictor.ag.cont_pool_size = [1,1], [1,1]




n, boundary = 1, 8
lr = 0.01



tn, cn = predictor.ag.targ_ker_num[-1], predictor.ag.cont_ker_num[-1]
th_x, ch_x = torch.Tensor(tn*n,2,2).uniform_(-boundary,boundary).to(gpu), torch.Tensor(cn*n,2,2).uniform_(-boundary,boundary).to(gpu)
th_x.requires_grad, ch_x.requires_grad = True, True
_n = torch.ones(1,n).long()
th_y, ch_y = (torch.arange(tn).view(-1,1)@_n).flatten().to(gpu), (torch.arange(cn).view(-1,1)@_n).flatten().to(gpu)

criterion = nn.CrossEntropyLoss()
op_t, op_c = optim.Adam([th_x], lr=lr), optim.Adam([ch_x], lr=lr)


t0 = time()
N = 1000
for epoch in range(10*N):
    ty, cy = predictor.encodeTraj(th_x,ch_x)

    op_t.zero_grad()
    loss_t = criterion(ty, th_y)
    loss_t.backward()
    op_t.step()

    op_c.zero_grad()
    loss_c = criterion(cy, ch_y)
    loss_c.backward()
    op_c.step()

    print('\r[%d] loss_t: %f loss_c: %f' % (epoch, loss_t, loss_c), end='')


for
utils.plotTraj2(th_x.detach().to(cpu),fn="z-ker_t")
utils.plotTraj2(ch_x.detach().to(cpu),fn="z-ker_c")









#
