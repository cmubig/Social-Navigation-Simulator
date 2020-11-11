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

from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import *

# ! pip install libsvm
from libsvm.svmutil import *

from gym_collision_avoidance.envs.policies.Group_Navi_GAN.scripts.sgan.various_length_models import TrajectoryDiscriminator,LateAttentionFullGenerator

from gym_collision_avoidance.envs.policies.Group_Navi_GAN.scripts.sgan.losses import displacement_error, final_displacement_error
from gym_collision_avoidance.envs.policies.Group_Navi_GAN.scripts.sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)




class GROUPNAVIGANPolicy(InternalPolicy):
    def __init__(self):
        InternalPolicy.__init__(self, str="GROUPNAVIGAN")
        self.dt = Config.DT
        self.obs_seq_len=8
        self.pred_seq_len=8 #for others it is 12, but for group navigan it is 8
        
        self.is_init = False

        self.checkpoint = torch.load('../envs/policies/Group_Navi_GAN/models/trackedgroup_zara1_batch64_epoch500_poolnet_with_model.pt', map_location='cuda:0')
        self.model=svm_load_model('../envs/policies/Group_Navi_GAN/spencer/group/social_relationships/groups_probabilistic_small.model')
        self.attention_generator = self.get_attention_generator(self.checkpoint)
        self.discriminator = self.get_discriminator(self.checkpoint)
        self._args = AttrDict(self.checkpoint['args'])
        try:
            self._args.delta
            
        except AttributeError:
            self._args.delta = False
        try:
            self._args.group_pooling
        except AttributeError:
            self._args.group_pooling = False

    def init(self,agents):
 
        self.total_agents_num = [None]*self.n_agents

        self.timestamp   = []
        self.agent_id    = []
        self.agent_pos_x = [None]*self.n_agents
        self.agent_pos_y = [None]*self.n_agents
 
        self.is_init = True
        
    def get_discriminator(self, checkpoint, best=False):
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
    def get_attention_generator(self,checkpoint, best=False):
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

    def row_repeat( self, tensor, num_reps):
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
            self, args, obs_traj, obs_traj_rel, seq_start_end, goals, goals_rel,  obs_delta, attention_generator, discriminator, agent_index, plot_dir=None
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

            ######################################################
            ###prediction, [prediction lens+observation lens,agents,(x,y)coorindates]]############
            #####################################################
            
            # time, agent_index, (x,y)
            print("Observation")
            print(obs_traj.cpu().numpy())
            print("Prediction")
            print(pred_traj_fake.cpu().numpy())

            print("="*50)
            print("Observation Agent "+str(agent_index))
            print(obs_traj.cpu().numpy()[:,agent_index,:])
            print("Prediction Agent "+str(agent_index))
            print(pred_traj_fake.cpu().numpy()[:,agent_index,:])

            return pred_traj_fake.cpu().numpy()

            #####################
            #robot prediction###
            ####################
            ###############################
            #ground truth for all agents###
            ###############################
             ###############################
             #ground truth for observed agents###
            ###############################

    def find_next_action(self, obs, agents, target_agent_index ):
        if not self.is_init:   #Execute one time per init (complete simulation iteration)
            self.n_agents = len(agents)
            self.init(agents)
        #current_time = self.world_info().sim_time

        agent_index = target_agent_index
        num_ped  = self.n_agents
        len_obs  = self.obs_seq_len
        len_pred = self.pred_seq_len #seems to be 8 in group navigan
        
        ############################
        # goal for predictive agent#
        ############################
        goals = torch.ones([1, num_ped, 2], dtype=torch.float32).cuda()
        goals_rel = torch.ones([1, num_ped, 2], dtype=torch.float32).cuda()
        obs_delta = torch.zeros([4, num_ped, num_ped], dtype=torch.float32).cuda()

        #################Calculate intermediate waypoints if the goal is too far#################
        goal = agents[agent_index].goal_global_frame#[20, 0]

        #create intermediate waypoint for the agent, if the goal is too far from the agent.
        displacement = goal - agents[agent_index].pos_global_frame
        dist_to_goal =  np.linalg.norm( displacement )

        goal_step_threshold = 1
        if dist_to_goal > goal_step_threshold:
            #normalized to 1 in magntitude length
            dist_next_waypoint = displacement /np.linalg.norm( displacement ,ord=1)
            #walk 1M towards the goal direction
            robot_goal = agents[agent_index].pos_global_frame + dist_next_waypoint * ( agents[agent_index].pref_speed * 0.4 ) #0.5M
        else:
            #it is within 1m, use goal direction, no need for intermediate waypoint
            robot_goal = agents[agent_index].pos_global_frame
            
        goal = torch.from_numpy( np.array( robot_goal )  )
        goals[0,agent_index,:] =  goal  #torch.Tensor([20,0])


        for i in range(self.n_agents):
            # Copy current agent positions, goal and preferred speeds into np array

            if self.agent_pos_x[i] is None:
                self.agent_pos_x[i] =   [ agents[i].pos_global_frame[0] ]
            else:
                self.agent_pos_x[i] +=  [ agents[i].pos_global_frame[0] ]
            
            if self.agent_pos_y[i] is None:
                self.agent_pos_y[i] =   [ agents[i].pos_global_frame[1] ]
            else:
                self.agent_pos_y[i] +=  [ agents[i].pos_global_frame[1] ]

        #Only take the latest 8 observation #test[:,-8:]

        observation_x_input     = np.array( self.agent_pos_x ) [ :  , -self.obs_seq_len: ] 
        observation_y_input     = np.array( self.agent_pos_y ) [ :  , -self.obs_seq_len: ]

        #np.column_stack(())        

        observation_len = len(observation_x_input[0]) #the time size of first agent

        prev_history_x = []
        prev_history_y = []
        if observation_len < (self.obs_seq_len):
            #return [0,0]

            for agent_ind in range(self.n_agents):
                #Set up and generate previous history for each agent
                #prev_start:  start position for prev history calculation
                #prev_goal :   goal positoin for prev history calculation
                
                prev_start = np.array( [  observation_x_input[agent_ind][0] , observation_y_input[agent_ind][0]  ] )
                prev_goal  = np.array( agents[agent_index].goal_global_frame )
                prev_history_len = self.obs_seq_len - observation_len
                #generate prev waypoints using prefered speed
                pos_difference = prev_goal - prev_start
                dist_next_waypoint = ( pos_difference /np.linalg.norm( pos_difference ,ord=1)  ) * ( agents[agent_index].pref_speed * 0.4 )

                prev_history_agent_x = []
                prev_history_agent_y = []
                #Generate prev waypoints
                for prev_history_ind in range(prev_history_len):
                    prev_waypoint = prev_start - dist_next_waypoint * (prev_history_len - prev_history_ind + 1)  # start position -
                    
                    prev_history_agent_x.append( prev_waypoint[0] )
                    prev_history_agent_y.append( prev_waypoint[1] )

                prev_history_x.append( prev_history_agent_x )
                prev_history_y.append( prev_history_agent_y )
                
            print("original observation=")
            print(observation_x_input)
                
            observation_x_input = np.concatenate(( np.array(prev_history_x) ,observation_x_input),axis=1)
            observation_y_input = np.concatenate(( np.array(prev_history_y) ,observation_y_input),axis=1)

            
            print("Since the observation length is only "+str(observation_len))
            print("observation_x_input is now")
            print(observation_x_input)

        # time, agent_index, (x,y)
        ############################
        ####### observation ########
        ############################


        obs_traj_rel = torch.ones([len_obs, num_ped, 2], dtype=torch.float32).cuda()

        #================================================================#
        #agents_history = np.empty((num_ped, 999999, 2 ))
        observation_input = []
        for time_ind in range(len_obs):
            temp = []
            for agent_ind in range(num_ped):
                temp.append([  observation_x_input[agent_ind][time_ind], observation_y_input[agent_ind][time_ind] ])
                
            observation_input.append( temp  )


        #============================================================#
            
        obs_traj = torch.from_numpy( np.array( observation_input ).astype(np.float32) ).cuda()
        #obs_traj = torch.ones([len_obs, num_ped, 2], dtype=torch.float32).cuda()

        obs_traj_rel = obs_traj - obs_traj[0,:,:]
        #goals[0,0,:] =  torch.Tensor([5-0.5*16,0])



        goals_rel = goals - obs_traj[0,:,:]
        seq_start_end = torch.Tensor([[0,num_ped]]).to(torch.int64).cuda()
        if self._args.delta is True:
            

            end_pos = obs_traj_rel[-1, :, :]

            # r1,r1,r1, r2,r2,r2, r3,r3,r3 - r1,r2,r3, r1,r2,r3, r1,r2,r3
            end_pos_difference = self.row_repeat(end_pos, num_ped) - end_pos.repeat(num_ped, 1)
            end_displacement = obs_traj_rel[-1,:,:] - obs_traj_rel[-2,:,:] / 0.4
            end_speed = torch.sqrt(torch.sum(end_displacement**2, dim=1)).view(-1,1)

            end_speed_difference =  self.row_repeat(end_speed, num_ped) - end_speed.repeat(num_ped, 1)
            end_heading = torch.atan2(end_displacement[:,0], end_displacement[:,1]).view(-1,1)
            end_heading_difference =  self.row_repeat(end_heading, num_ped) - end_heading.repeat(num_ped, 1)
            # num_ped**2
            delta_distance = torch.sqrt(torch.sum(end_pos_difference**2, dim=1)).view(-1,1)
            # num_ped
            delta_speed = torch.abs(end_speed_difference)
            delta_heading = torch.abs(torch.atan2(torch.sin(end_heading_difference), torch.cos(end_heading_difference)))

            _x = torch.cat((delta_distance, delta_speed, delta_heading),1)
            _, _, prob = svm_predict([], _x.tolist(), self.model,'-b 1 -q')
            prob = torch.FloatTensor(prob)[:,0]
            #positive prob >0.5 consider group relationship 
            obs_delta[3, :, :num_ped] = (prob>0.5).long().view(num_ped, num_ped)
            obs_delta[0, :, :num_ped] = delta_distance.view(num_ped, num_ped)
            obs_delta[1, :, :num_ped] = delta_speed.view(num_ped, num_ped)
            obs_delta[2, :, :num_ped] = delta_heading.view(num_ped, num_ped)
            #print(obs_delta[3, :, :num_ped])


        prediction = self.evaluate( self._args, obs_traj, obs_traj_rel, seq_start_end, goals, goals_rel, obs_delta, self.attention_generator, self.discriminator, agent_index )

        prediction_index = 2
        self.next_waypoint = prediction[prediction_index][agent_index]
        #print(next_waypoint)

        #Directly update agent's position
        #position_x = np.clip( ( (self.next_waypoint[0] - agents[agent_index].pos_global_frame[0])/5) , -0.1, 0.1) + agents[agent_index].pos_global_frame[0]
        #position_y = np.clip( ( (self.next_waypoint[1] - agents[agent_index].pos_global_frame[1])/5) , -0.1, 0.1) + agents[agent_index].pos_global_frame[1]
        pos_difference = self.next_waypoint -  agents[agent_index].pos_global_frame    
        dist_next_waypoint = ( pos_difference /np.linalg.norm( pos_difference ,ord=1)  ) * ( agents[agent_index].pref_speed * 0.4 )

        position_x = agents[agent_index].pos_global_frame[0] + dist_next_waypoint[0]
        position_y = agents[agent_index].pos_global_frame[1] + dist_next_waypoint[1]
        agents[agent_index].set_state( position_x , position_y )

        resultant_speed_global_frame         = agents[agent_index].speed_global_frame
        resultant_delta_heading_global_frame = agents[agent_index].delta_heading_global_frame

        #Although documentation and code comment mentioned that action is consisted with  [heading delta, speed]
        #But in reality, the format of action is [speed, heading_delta]
        action = [ resultant_speed_global_frame , resultant_delta_heading_global_frame ]

        return action



