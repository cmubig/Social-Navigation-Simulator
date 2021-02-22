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

''' Load data '''
npy_path = '/home/dzhao/proj/scnn/datasets/'+args.dataset_name+'_loaders.npy'
if args.loadNpy:
    logger.info("loading data from npy ...")
    loader_trn, loader_val, loader_tst = np.load(npy_path,allow_pickle=True)
else:
    data_path = '/home/dzhao/Dropbox/sonav/sgan/data/'+args.dataset_name
    logger.info("reading data from txt ...")
    loader_trn, loader_val, loader_tst = data_loader(args, data_path, batch_size=(args.batch_size,args.batch_size_val,args.batch_size_tst))
    np.save(npy_path,[loader_trn, loader_val, loader_tst])
    # trn_path = utils.get_dset_path(args.dataset_name, 'train')
    # val_path = utils.get_dset_path(args.dataset_name, 'val')
    # tst_path = utils.get_dset_path(args.dataset_name, 'test')
    # trn_loader_loc = data_loader(args, trn_path, fut_loc=True, batch_size=args.batch_size)
    # trn_loader_trj = data_loader(args, trn_path, fut_loc=False, batch_size=args.batch_size)
    # val_loader_trj = data_loader(args, val_path, fut_loc=False, batch_size=args.batch_size_val)
    # tst_loader_trj = data_loader(args, tst_path, fut_loc=False, batch_size=args.batch_size_tst)
    # np.save(npy_path,[trn_loader_loc, trn_loader_trj, val_loader_trj, tst_loader_trj])

num_sample = len(loader_trn.dataset)
iterations_per_epoch = num_sample/args.batch_size
args.n_iteration = min(args.n_iteration,iterations_per_epoch)
logger.info('{} samples in an epoch, {} iterations per epoch'.format(num_sample,iterations_per_epoch))
logger.info('{} epochs;  batch_size: {} '.format(args.n_epoch,args.batch_size))
print("max()"if args.use_max else "sum()")



''' Construct Model '''
gpu, cpu = "cuda:0", "cpu"
if args.loadModel and args.loadModel[-4:]!=".npy": args.loadModel+=".npy"
predictor = np.load(args.dataset_name+"_"+args.loadModel,allow_pickle=True)[0] if args.loadModel else model.LocPredictor(args).to(gpu)
optimizer = optim.Adam(predictor.parameters(), lr=args.lr)

t0 = time()
err = []
nan_cnt = 0
best_i, min_err = -1, 9999
model_i = 0
# plt.figure(); plt.ion(); plt.show()
for epoch in range(args.n_epoch):
    # predictor.c_conv[0].lock_weights(epoch<8)

    sum_trn_loss = 0.0
    for i, b in enumerate(loader_trn):
        if i>=args.n_iteration: break
        targ_hist, cont_hist, end_idx, targ_nextLoc = b
        loc_pred = predictor.forward(targ_hist.to(gpu), cont_hist.to(gpu), end_idx)
        loss = utils.getLoss(targ_nextLoc, loc_pred.to(cpu))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_trn_loss += loss

        print('\r[%d-%d/%d-%d] %.3f' % (model_i,epoch,args.n_epoch,i,sum_trn_loss/(i+1)),end=' ')
        if torch.isnan(sum_trn_loss): print("NaN!!!");raise Exception('Nan error.')

    # optimizer.defaults['lr'] = utils.update_lr(epoch,optimizer.defaults['lr'])
    t1 = time()-t0
    print("%.2fs/%.2fm %f"%(t1,t1/60, optimizer.defaults['lr']), end=' ')

    if (epoch+1)%args.val_freq==0:
        utils.plotWeights(predictor.c_conv[0].weight.detach().to(cpu),fn="weights/2layer/"+args.dataset_name+"/7aug"+str(epoch) )
        ade_vd,fde_vd= utils.eval_model(predictor,loader_val,determ=1,n_batch=args.n_batch_val)
        ade_v, fde_v = 0,0 # utils.eval_model(predictor,loader_val,determ=0,n_batch=args.n_batch_val)
        print('v%.3f/%.3f' % (ade_vd,fde_vd), end=' ')
        err.append([ade_vd,fde_vd,ade_v,fde_v])
        err_val = (ade_vd+fde_vd)
        if epoch>0.7*args.n_epoch:
            if err_val<min_err or best_i<0:
                best_i, min_err = len(err)*args.val_freq, err_val
                utils.save_model(predictor,fn=args.dataset_name+str(model_i))
    print('!')


# best_predictor = utils.load_model(fn=args.dataset_name+str(model_i))
ade_t, fde_t = utils.eval_model(predictor,loader_tst,determ=0,n_batch=args.n_batch_tst,repeat=1)
# Err.append([ade_t,fde_t])
# print('tp:%.3f/%.3f' % (ade_t,fde_t))
print('Finished Training',time()-t0)
# utils.write_csv([model_i,*Params[model_i],best_i,ade_t,fde_t],fn=args.dataset_name)
utils.write_csv([model_i,best_i,ade_t,fde_t],fn=args.dataset_name)
#     utils.plot_err(err,ade_t,fde_t,fn=args.dataset_name+str(model_i))



























while True:
    pass
p = list(predictor.parameters())
utils.plotTraj2(p[0].detach().to(cpu),fn="weights_designed")


# hist_t, fut_t, ei_t, pred_t, ade_t, fde_t = eval_tst(predictor,tst_path)
# predictor.ag.coef=1.001; pred = predictor.predictTrajSample(hist_t.to(gpu),ei_t).cpu(); ade_s = utils.ade(fut_t,pred); fde_s = utils.fde(fut_t,pred); ade_s, fde_s


b = list( data_loader(args, utils.get_dset_path(dsn, 'test'), False, batch_size=-1, shuffle=True) )[0]
hist_t, fut_t, ei_t = b[0][:,:,:args.hist_len], b[0][:,:,args.hist_len:], b[1]
utils.plot_valBatch( ei_t, hist_t.numpy(), fut_t.numpy(), bestPred.detach().numpy(),fn=args.dataset_name+"/", num=100 )
utils.plot_valBatch( ei_t, hist_t.numpy(), fut_t.numpy(), pred_t.detach().numpy(),fn=args.dataset_name+"/", num=100 )


np.save(mn,[predictor])

p = list(predictor.parameters())
utils.plotTraj2(p[0].detach().to(cpu),fn="weights/t_cnn")
utils.plotTraj2(p[4].detach().to(cpu),fn="weights/c_cnn")

hist_t, fut_t, ei_t, pred_t, ade_t, fde_t = eval_tst(predictor,tst_path,fn='c_')

''' plot weights '''
# utils.plotTraj2(p[0].detach().to(cpu),fn="weights/t_l2d_"+str(epoch))
# utils.plotTraj2(p[8].detach().to(cpu),fn="weights/c_l2d_"+str(epoch))


p = list(predictor.parameters())
utils.plotTraj2(p[0].detach().to(cpu),fn="weights/t_l2d")
utils.plotTraj2(p[8].detach().to(cpu),fn="weights/c_l2d")

hist_t, fut_t, ei_t, pred_t, ade_t, fde_t = eval_tst(predictor,tst_path,fn='l_')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



dset = TrajDataset_val(data_dir=path,hist_len=args.hist_len,fut_len=args.fut_len,min_ped=args.min_ped,delim=args.delim,full_scene_only=0)
