import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utilsv2 import * 
from metrics import * 
from model import social_stgcnn
import copy

dset_test = Sam_TrajectoryDataset(
        skip=1,norm_lap_matr=True)

loader_test = dset_test.__getitem__(0)


batch_size=128
clip_grad=None
dataset='eth'
input_size=2
kernel_size=3
lr=0.01
lr_sh_rate=150
n_stgcnn=1
n_txpcnn=5
num_epochs=250
obs_seq_len=8
output_size=5
pred_seq_len=12
tag='social-stgcnn-eth'
use_lrschd=True

#Defining the model 
model = social_stgcnn(n_stgcnn =n_stgcnn,n_txpcnn=n_txpcnn,
output_feat=output_size,seq_len=obs_seq_len,
kernel_size=kernel_size,pred_seq_len=pred_seq_len).cuda()
model.load_state_dict(torch.load(os.path.dirname(__file__)+"/checkpoint/social-stgcnn-eth/val_best.pth"))


model.eval()

ade_bigls = []
fde_bigls = []
raw_data_dict = {}
step=0

batch = loader_test

step+=1
#Get data
batch = [tensor.cuda() for tensor in batch]
obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
 loss_mask,V_obs,A_obs,V_tr,A_tr = batch

num_of_objs = obs_traj_rel.shape[0]

V_obs = V_obs.unsqueeze(0) #add the batch dimension back
A_obs = A_obs.unsqueeze(0) #add the batch dimension back

##################
#V_obs      len of sequnce = 8     node/ person = 37,    feature/ position tuples = 2
#torch.Size([8, 37, 2])
###################

#Forward
#V_obs = batch,seq,node,feat
#V_obs_tmp = batch,feat,seq,node

V_obs_tmp =V_obs.permute(0,3,1,2) #permute is used to swap/rearrange channels in array


V_pred,_ = model(V_obs_tmp,A_obs.squeeze())




# print(V_pred.shape)
# torch.Size([1, 5, 12, 2])
# torch.Size([12, 2, 5])
V_pred = V_pred.permute(0,2,3,1)
# torch.Size([1, 12, 2, 5])>>seq,node,feat
# V_pred= torch.rand_like(V_tr).cuda()

V_tr = V_tr.squeeze()
A_tr = A_tr.squeeze()
V_pred = V_pred.squeeze()
num_of_objs = obs_traj_rel.shape[0]
V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]
#print(V_pred.shape)

#For now I have my bi-variate parameters 
#normx =  V_pred[:,:,0:1]
#normy =  V_pred[:,:,1:2]
sx = torch.exp(V_pred[:,:,2]) #sx
sy = torch.exp(V_pred[:,:,3]) #sy
corr = torch.tanh(V_pred[:,:,4]) #corr

cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda()
cov[:,:,0,0]= sx*sx
cov[:,:,0,1]= corr*sx*sy
cov[:,:,1,0]= corr*sx*sy
cov[:,:,1,1]= sy*sy
mean = V_pred[:,:,0:2]

mvnormal = torchdist.MultivariateNormal(mean,cov)


### Rel to abs 
##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 

#Now sample 20 samples
ade_ls = {}
fde_ls = {}
V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                         V_x[0,:,:].copy())

V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                         V_x[-1,:,:].copy())

raw_data_dict[step] = {}
raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
raw_data_dict[step]['pred'] = []

print("observation?")
print(V_x_rel_to_abs.shape)
print(V_x_rel_to_abs)




global V_pred_rel_to_abs
V_pred = mvnormal.sample()
V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                             V_x[-1,:,:].copy())

print("prediction?")
print(V_pred_rel_to_abs.shape)
print(V_pred_rel_to_abs)

        



'''
observation?
(8, 5, 2)
[[[6.47      6.68     ]
  [8.59      6.85     ]
  [1.76      4.89     ]
  [1.6       4.13     ]
  [9.09      6.02     ]]

 [[5.8599997 6.8199997]
  [7.78      6.8399997]
  [2.81      4.83     ]
  [2.67      4.07     ]
  [8.2300005 6.04     ]]

 [[5.24      6.98     ]
  [6.96      6.8399997]
  [3.8       4.77     ]
  [3.6999998 4.02     ]
  [7.4       6.15     ]]

 [[4.87      7.16     ]
  [6.29      7.       ]
  [4.9700003 4.69     ]
  [4.75      4.04     ]
  [6.52      6.15     ]]

 [[4.5099998 7.58     ]
  [5.62      7.1      ]
  [5.9700003 4.7      ]
  [5.7599998 4.06     ]
  [5.7       6.21     ]]

 [[4.2       7.2999997]
  [5.0600004 7.04     ]
  [6.9700003 4.67     ]
  [6.7599998 4.04     ]
  [4.96      6.1      ]]

 [[3.9499998 7.71     ]
  [4.69      7.       ]
  [8.12      4.95     ]
  [7.8199997 4.17     ]
  [4.08      6.2      ]]

 [[3.4699998 7.8599997]
  [4.35      7.0099998]
  [9.110001  5.0099998]
  [8.85      4.21     ]
  [3.31      6.2      ]]]
prediction?
(12, 5, 2)
[[[ 3.16522     7.9197836 ]
  [ 3.903026    7.0568776 ]
  [ 9.876129    5.43436   ]
  [ 9.587457    4.553406  ]
  [ 2.3988733   5.6783066 ]]

 [[ 2.9049807   7.9428453 ]
  [ 3.4102368   7.074091  ]
  [10.735053    5.5532103 ]
  [10.574282    4.663058  ]
  [ 1.513771    5.4130764 ]]

 [[ 2.5319738   8.111646  ]
  [ 2.7537546   7.0080614 ]
  [11.3346405   5.65913   ]
  [11.648674    4.6484976 ]
  [ 1.0116153   5.2932067 ]]

 [[ 2.2866802   8.290825  ]
  [ 2.2289457   7.118914  ]
  [12.234463    5.671326  ]
  [12.470726    4.667672  ]
  [ 0.2607894   5.42359   ]]

 [[ 2.11819     8.508596  ]
  [ 1.7825551   7.2772517 ]
  [13.123756    5.8263135 ]
  [13.31522     4.8961062 ]
  [-0.23150587  5.7129154 ]]

 [[ 2.1208882   8.70346   ]
  [ 1.3395376   7.3461967 ]
  [13.932061    5.9268756 ]
  [14.060167    5.0561295 ]
  [-0.9449277   5.6790338 ]]

 [[ 1.7190648   8.810028  ]
  [ 0.6664095   7.3673334 ]
  [14.796032    5.799988  ]
  [15.106211    5.1928425 ]
  [-2.0644217   5.807244  ]]

 [[ 1.3389492   8.896507  ]
  [-0.07990599  7.3709736 ]
  [15.434995    5.7159634 ]
  [15.9754305   5.1814823 ]
  [-2.6520867   6.0698547 ]]

 [[ 1.0153625   9.134145  ]
  [-0.46419334  7.552571  ]
  [15.811386    5.952366  ]
  [16.84682     5.2686567 ]
  [-2.9933553   5.8865895 ]]

 [[ 0.6129689   9.013208  ]
  [-0.9778433   7.4187684 ]
  [16.495682    6.225641  ]
  [17.819202    5.533411  ]
  [-3.5986729   5.8739552 ]]

 [[ 0.08531117  9.476206  ]
  [-1.2717972   7.336082  ]
  [17.193462    6.089966  ]
  [18.430614    5.259799  ]
  [-4.379125    5.502166  ]]

 [[-0.22672844  9.594957  ]
  [-1.7534418   7.512884  ]
  [18.067535    6.345502  ]
  [19.42656     5.4838014 ]
  [-5.1957793   5.786113  ]]]
'''
