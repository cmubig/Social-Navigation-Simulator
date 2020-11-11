from socialgan.data.loader import data_loader, custom_data_loader
from socialgan.models import TrajectoryGenerator
from socialgan.losses import displacement_error, final_displacement_error
from socialgan.utils import relative_to_abs, get_dset_path

import argparse
import os
import torch

from attrdict import AttrDict

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='models/sgan-models')
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
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
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator):
    ade_outer, fde_outer = [], []
    total_traj = 0
    print("pre no grad")
    with torch.no_grad():
        print("in no grad")
        print("loader")
        print(loader)
        for batch in loader:
            print("in evaluate")
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)


            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1]
            )
            ade.append(displacement_error(
                pred_traj_fake, pred_traj_gt, mode='raw'
            ))
            fde.append(final_displacement_error(
                pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
            ))

            print("Observation")
            print(obs_traj)
            print(obs_traj.cpu().numpy().shape)
            print("Prediction")
            print(pred_traj_fake)
            print(pred_traj_fake.cpu().numpy().shape)

##observation_input = []
##    for time_ind in range(self.obs_seq_len):
##        temp = []
##        for agent_ind in range(self.n_agents):
##            temp.append([ agent_ind, observation_x_input[agent_ind][time_ind], observation_y_input[agent_ind][time_ind] ])
##
##        observation_input.append( temp  )
##
##    for time_ind in range(self.obs_seq_len,20):
##        temp = []
##        for agent_ind in range(self.n_agents):
##            temp.append([ agent_ind, 0, 0 ])
##
##        observation_input.append( temp  )
num_of_agents = 10
observation_input = []
for time_ind in range(8):
    for agent_ind in range(num_of_agents):
        observation_input.append([ time_ind*10, agent_ind, time_ind, time_ind ])

for time_ind in range(8,20):
    for agent_ind in range(num_of_agents):
        observation_input.append([ time_ind*10, agent_ind, 0, 0 ])

print("GOD")
print(observation_input)
observation_input = np.array( observation_input )

print("load 1")
checkpoint = torch.load("models/sgan-models/univ_12_model.pt")
print("load 2")        
generator = get_generator(checkpoint)
print("load 3")
_args = AttrDict(checkpoint['args'])
data_input=None
_, loader = custom_data_loader(_args, observation_input)
print("load 4")
evaluate(_args, loader, generator)
print("="*50)
