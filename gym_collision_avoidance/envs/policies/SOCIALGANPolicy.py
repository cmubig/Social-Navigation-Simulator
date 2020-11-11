import argparse
import os
import torch

from attrdict import AttrDict

import numpy as np

from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import *

from gym_collision_avoidance.envs.policies.SOCIALGAN.socialgan.data.loader import data_loader, custom_data_loader
from gym_collision_avoidance.envs.policies.SOCIALGAN.socialgan.models import TrajectoryGenerator
from gym_collision_avoidance.envs.policies.SOCIALGAN.socialgan.losses import displacement_error, final_displacement_error
from gym_collision_avoidance.envs.policies.SOCIALGAN.socialgan.utils import relative_to_abs, get_dset_path



parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='models/sgan-models')
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

class SOCIALGANPolicy(InternalPolicy):
    def __init__(self):
        InternalPolicy.__init__(self, str="SOCIALGAN")
        self.dt = Config.DT
        self.obs_seq_len=8
        self.pred_seq_len=12
        
        self.is_init = False


        print("load 1")
        self.checkpoint = torch.load("../envs/policies/SOCIALGAN/models/sgan-models/univ_12_model.pt")
        print("load 2")        
        self.generator = self.get_generator(self.checkpoint)
        print("load 3")
        self._args = AttrDict(self.checkpoint['args'])

    def init(self,agents):
 
        self.total_agents_num = [None]*self.n_agents

        self.total_agents_num = [None]*self.n_agents
        self.agent_pos_x = [None]*self.n_agents
        self.agent_pos_y = [None]*self.n_agents
        
        self.near_goal_threshold = 0.5
        
        self.is_init = True




        
    def get_generator(self,checkpoint):
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

    def evaluate(self, args, loader, generator):
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

                return pred_traj_fake.cpu().numpy()


    def find_next_action(self, obs, agents, i ):

        if not self.is_init:   #Execute one time per init (complete simulation iteration)
            self.n_agents = len(agents)
            self.init(agents)
        #current_time = self.world_info().sim_time

        agent_index = i
        #CHECK IF AT GOAL (0.5 radius for motion prediction algoritm)
        #print("TEST")
        """ Set :code:`self.is_at_goal` if norm(pos_global_frame - goal_global_frame) <= near_goal_threshold """

        is_near_goal = (agents[agent_index].pos_global_frame[0] - agents[agent_index].goal_global_frame[0])**2 + (agents[agent_index].pos_global_frame[1] - agents[agent_index].goal_global_frame[1])**2 <= self.near_goal_threshold**2
        if is_near_goal:
            agents[agent_index].is_at_goal = is_near_goal
            return np.array([0,0])

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

                #if agents[0].step_num < 5 != 0: return [ agents[agent_index].speed_global_frame , 0 ]

        if agents[0].step_num <= 3:


            goal_direction = agents[agent_index].goal_global_frame - agents[agent_index].pos_global_frame
            self.dist_to_goal = math.sqrt(goal_direction[0]**2 + goal_direction[1]**2)
            if self.dist_to_goal > 1e-8:
                ref_prll = goal_direction / agents[agent_index].dist_to_goal
            else:
                ref_prll = goal_direction
            ref_orth = np.array([-ref_prll[1], ref_prll[0]])  # rotate by 90 deg

            ref_prll_angle_global_frame = np.arctan2(ref_prll[1],
                                                     ref_prll[0])
            heading_ego_frame = wrap( agents[agent_index].heading_global_frame -
                                          ref_prll_angle_global_frame)

        

            vel_global_frame = ( agents[agent_index].goal_global_frame - agents[agent_index].pos_global_frame) / agents[agent_index].dt_nominal

            speed_global_frame = np.linalg.norm(vel_global_frame)
            if speed_global_frame > agents[agent_index].pref_speed: speed_global_frame = agents[agent_index].pref_speed

            #But in reality, the format of action is [speed, heading_delta]

            action = np.array([agents[agent_index].pref_speed, -heading_ego_frame])
            print("action")
            print(action)
            return action


        #if agents[0].step_num % 2!=0 : return [ agents[agent_index].speed_global_frame ,0 ] #comment or %2 or %4
        
        #Only take the latest 8 observation #test[:,-8:]
        #select every 4 element from the end,    reverse it, and select the latest 8 entry
        observation_x_input = np.array( self.agent_pos_x )[:,::-4][:,::-1][:,-8:]
        observation_y_input = np.array( self.agent_pos_y )[:,::-4][:,::-1][:,-8:]


        #np.column_stack(())        

        observation_len = len(observation_x_input[0])
        
        prev_history_x = []
        prev_history_y = []
        if observation_len < (self.obs_seq_len):
            #return [0,0]

            for agent_ind in range(self.n_agents):
                #Set up and generate previous history for each agent
                #prev_start:  start position for prev history calculation
                #prev_goal :   goal positoin for prev history calculation
                
                prev_start = np.array( [  observation_x_input[agent_ind][0] , observation_y_input[agent_ind][0]  ] )
                prev_goal  = np.array( agents[agent_ind].goal_global_frame )
                prev_history_len = self.obs_seq_len - observation_len
                #generate prev waypoints using prefered speed
                pos_difference = prev_goal - prev_start
                dist_next_waypoint = ( pos_difference / (np.linalg.norm( pos_difference ,ord=1)+0.000001)  ) * ( agents[agent_ind].pref_speed * agents[agent_ind].dt_nominal )

                prev_history_agent_x = []
                prev_history_agent_y = []
                #Generate prev waypoints
                for prev_history_ind in range(prev_history_len):
                    prev_waypoint = prev_start - dist_next_waypoint * (prev_history_len - prev_history_ind )  *  4   #every 4 steps
                    
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

        #observation_x_input = observation_x_input - observation_x_input[agent_index,0]#observation_x_input[:,0][:,None]
        #observation_y_input = observation_y_input - observation_y_input[agent_index,0]#observation_y_input[:,0][:,None]

        #print("relative")
        #print(observation_x_input)

        ####FOR Observation input, its shape is [20, num_agents, 3(agent_id,x,y) ]
        #####HOWEVER, the target agent have to be the last within the timestamp array, e.g:  for 1,2,3 agents, if agent 2 is target agent, then for timestamp x, the array is [ [1,x,y],[3,x,y],[2,x,y] ]

        ####Basically, remove the target agent from the array and append to the end of the array

        observation_input = []
        for time_ind in range(self.obs_seq_len):
            
            for agent_ind in range(self.n_agents):
                observation_input.append([ time_ind , agent_ind , observation_x_input[agent_ind][time_ind], observation_y_input[agent_ind][time_ind] ])


        #add 12 empty record entry after the observation, as it is required for input
        for time_ind in range(self.obs_seq_len, ( self.obs_seq_len + self.pred_seq_len ) ):
            for agent_ind in range(self.n_agents):
                observation_input.append([ time_ind, agent_ind, 0, 0 ])

##        print("debug")
##        for a in observation_input:
##            print(a)
        observation_input = np.array( observation_input )        

        _, loader = custom_data_loader(self._args, observation_input)
        print("load 4")
        prediction = self.evaluate(self._args, loader, self.generator)

        prediction_index = 2
        self.next_waypoint = prediction[prediction_index][agent_index] 
        #print(next_waypoint)

        goal_direction = self.next_waypoint - agents[agent_index].pos_global_frame
        self.dist_to_goal = math.sqrt(goal_direction[0]**2 + goal_direction[1]**2)
        if self.dist_to_goal > 1e-8:
            ref_prll = goal_direction / agents[agent_index].dist_to_goal
        else:
            ref_prll = goal_direction
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])  # rotate by 90 deg

        ref_prll_angle_global_frame = np.arctan2(ref_prll[1],
                                                 ref_prll[0])
        heading_ego_frame = wrap( agents[agent_index].heading_global_frame -
                                      ref_prll_angle_global_frame)

    

        vel_global_frame = ( self.next_waypoint - agents[agent_index].pos_global_frame) / agents[agent_index].dt_nominal

        speed_global_frame = np.linalg.norm(vel_global_frame)
        if speed_global_frame > agents[agent_index].pref_speed: speed_global_frame = agents[agent_index].pref_speed

        #But in reality, the format of action is [speed, heading_delta]

        action = np.array([speed_global_frame, -heading_ego_frame])
        print("action")
        print(action)
       
        return action


