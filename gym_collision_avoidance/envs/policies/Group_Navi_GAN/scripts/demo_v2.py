import argparse
import os
import sys
import torch
import imageio
import numpy as np
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from attrdict import AttrDict
torch.backends.cudnn.benchmark = True


# ! pip install libsvm
from libsvm.svmutil import *

from sgan.various_length_models import TrajectoryDiscriminator,LateAttentionFullGenerator

from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

parser.add_argument('--model_path', type=str, default='../models/trackedgroup_zara1_batch64_epoch500_poolnet_with_model.pt')

parser.add_argument('--plot_dir', default='./plots/')

def get_discriminator(checkpoint, best=False):
    args = AttrDict(checkpoint['args'])
    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type)
    if best:
        discriminator.load_state_dict(checkpoint['d_best_state'])
    else:
        discriminator.load_state_dict(checkpoint['d_state'])
    discriminator.cuda()
    discriminator.train()
    return discriminator

# For late attent model by full state
def get_attention_generator(checkpoint, best=False):
    args = AttrDict(checkpoint['args'])
    try:
        args.delta
    except AttributeError:
         args.delta = False
    try:
        args.group_pooling
    except AttributeError:
        args.group_pooling = False
    generator = LateAttentionFullGenerator(
        goal_dim=(2,),
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        group_pooling=args.group_pooling,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
        spatial_dim=2)
    if best:
        generator.load_state_dict(checkpoint['g_best_state'])
    else:
        generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

def row_repeat( tensor, num_reps):
    """
    Inputs:
    -tensor: 2D tensor of any shape
    -num_reps: Number of times to repeat each row
    Outpus:
    -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
    """
    col_len = tensor.size(1)
    tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
    tensor = tensor.view(-1, col_len)
    return tensor

def evaluate(
        args, obs_traj, obs_traj_rel, seq_start_end, goals, goals_rel,  obs_delta, attention_generator, discriminator, plot_dir=None
    ):
    ade_outer, fde_outer = [], []
    total_traj = 0
    count = 1
    guid = 0
    attention_generator.eval()
    with torch.no_grad():
        D_real, D_fake = [], []
        #ade, fde = [], []
    

        if args.delta is True:
            pred_traj_fake_rel, _ = attention_generator(
                obs_traj, obs_traj_rel, seq_start_end, _obs_delta_in = obs_delta, seq_len=attention_generator.pred_len, goal_input=goals_rel
            )
        else:
            pred_traj_fake_rel, _ = attention_generator(
                obs_traj, obs_traj_rel, seq_start_end, _obs_delta_in = None, seq_len=attention_generator.pred_len, goal_input=goals_rel
            )
        ######################################################
        ###prediction from gan, [lens,agents,(x,y)coorindates]]############
        ######################################################
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0])
        #ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
        #fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
        # Record the trajectories
        # For each batch, draw only the first sample

        goal_point = goals[0,0, :]
        ######################################################
        ###prediction, [prediction lens+observation lens,agents,(x,y)coorindates]]############
        #####################################################
        
        # time, agent_index, (x,y)
        print("Observation")
        print(obs_traj.cpu().numpy())
        print("Prediction")
        print(pred_traj_fake.cpu().numpy())

        print("="*50)
        print("Observation 0")
        print(obs_traj.cpu().numpy()[:,0,:])
        print("Prediction 0")
        print(pred_traj_fake.cpu().numpy()[:,0,:])

        #####################
        #robot prediction###
        ####################
        ###############################
        #ground truth for all agents###
        ###############################
         ###############################
         #ground truth for observed agents###
        ###############################



def main(args):
    if os.path.isdir(args.model_path):
        
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
  
    else: 
        paths = [args.model_path]

    for path in paths:

        checkpoint = torch.load(path, map_location='cuda:0')
        attention_generator = get_attention_generator(checkpoint)
        discriminator = get_discriminator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        try:
            _args.delta
            
        except AttributeError:
             _args.delta = False
        try:
            _args.group_pooling
        except AttributeError:
            _args.group_pooling = False
        num_ped = 10
        len_pred = 8

        # time, agent_index, (x,y)
        ############################
        ####### observation ########
        ############################
        obs_traj = torch.ones([len_pred, num_ped, 2], dtype=torch.float32).cuda()

        obs_traj_rel = torch.ones([len_pred, num_ped, 2], dtype=torch.float32).cuda()

        ############################
        # goal for predictive agent#
        ############################
        goals = torch.ones([1, num_ped, 2], dtype=torch.float32).cuda()
        goals_rel = torch.ones([1, num_ped, 2], dtype=torch.float32).cuda()
        obs_delta = torch.zeros([4, num_ped, num_ped], dtype=torch.float32).cuda()
        
        for t in range(len_pred):
            obs_traj[t,0,:] = torch.Tensor([5 - 0.5*t, 0])
            
            obs_traj[t,1,:] = torch.Tensor([-11 + 0.6*t, -0.1])
            obs_traj[t,2,:] = torch.Tensor([-11 + 0.6*t, 0.4])
            obs_traj[t,3,:] = torch.Tensor([-11+ 0.6*t, 1])
            obs_traj[t,4,:] = torch.Tensor([4.8 - 0.6*t, -0.5])
            obs_traj[t,5,:] = torch.Tensor([4.8 - 0.6*t, -1])
            obs_traj[t,6,:] = torch.Tensor([4.8 - 0.6*t, -1.5])

            obs_traj[t,7,:] = torch.Tensor([8 - 0.7*t, -0.5])
            obs_traj[t,8,:] = torch.Tensor([8 - 0.7*t, -1])
            obs_traj[t,9,:] = torch.Tensor([8 - 0.7*t, -1.5])
            
        obs_traj_rel = obs_traj - obs_traj[0,:,:]
        goals[0,0,:] =  torch.Tensor([5-0.5*16,0])

    
        goals_rel = goals - obs_traj[0,:,:]
        seq_start_end = torch.Tensor([[0,num_ped]]).to(torch.int64).cuda()
        if _args.delta is True:
            model=svm_load_model('../spencer/group/social_relationships/groups_probabilistic_small.model')
    
            end_pos = obs_traj_rel[-1, :, :]

            # r1,r1,r1, r2,r2,r2, r3,r3,r3 - r1,r2,r3, r1,r2,r3, r1,r2,r3
            end_pos_difference = row_repeat(end_pos, num_ped) - end_pos.repeat(num_ped, 1)
            end_displacement = obs_traj_rel[-1,:,:] - obs_traj_rel[-2,:,:] / 0.4
            end_speed = torch.sqrt(torch.sum(end_displacement**2, dim=1)).view(-1,1)

            end_speed_difference =  row_repeat(end_speed, num_ped) - end_speed.repeat(num_ped, 1)
            end_heading = torch.atan2(end_displacement[:,0], end_displacement[:,1]).view(-1,1)
            end_heading_difference =  row_repeat(end_heading, num_ped) - end_heading.repeat(num_ped, 1)
            # num_ped**2
            delta_distance = torch.sqrt(torch.sum(end_pos_difference**2, dim=1)).view(-1,1)
            # num_ped
            delta_speed = torch.abs(end_speed_difference)
            delta_heading = torch.abs(torch.atan2(torch.sin(end_heading_difference), torch.cos(end_heading_difference)))

            _x = torch.cat((delta_distance, delta_speed, delta_heading),1)
            _, _, prob = svm_predict([], _x.tolist(), model,'-b 1 -q')
            prob = torch.FloatTensor(prob)[:,0]
            #positive prob >0.5 consider group relationship 
            obs_delta[3, :, :num_ped] = (prob>0.5).long().view(num_ped, num_ped)
            obs_delta[0, :, :num_ped] = delta_distance.view(num_ped, num_ped)
            obs_delta[1, :, :num_ped] = delta_speed.view(num_ped, num_ped)
            obs_delta[2, :, :num_ped] = delta_heading.view(num_ped, num_ped)
            #print(obs_delta[3, :, :num_ped])

        
        evaluate(    
            _args, obs_traj, obs_traj_rel,  seq_start_end, goals, goals_rel, obs_delta, attention_generator, discriminator,
           plot_dir=args.plot_dir)
        
        #print(' Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(_args.pred_len, ade, fde))
    

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main(args)
