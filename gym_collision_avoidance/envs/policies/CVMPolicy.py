import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import glob
import torch.distributions.multivariate_normal as torchdist

from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import *


import copy


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

    def predictTraj(self,hist,ei=None):
        if ei is None: ei = [0,len(hist)]
        observed = hist.permute(2,0,1)
        y_pred_rel = self.constant_velocity_model(observed)
        y_pred_abs = self.rel_to_abs(y_pred_rel, observed[-1])
        return y_pred_abs.permute(1,2,0)

    def predictTrajSample(self,hist,ei=None):
        if ei is None: ei = [0,len(hist)]
        print('sample',end=',')
        observed = hist.permute(2,0,1)
        y_samples = []
        for i in range(20):
            print(i,end='\r')
            y_pred_rel = self.constant_velocity_model(observed,sample=True)
            y_pred_abs = self.rel_to_abs(y_pred_rel, observed[-1])
            y_samples.append(y_pred_abs.permute(1,2,0))
        return torch.stack(y_samples)






class CVMPolicy(InternalPolicy):
    def __init__(self):
        InternalPolicy.__init__(self, str="CVM")
        self.dt = Config.DT
        self.obs_seq_len  = 8
        self.obs_pred_len = 12

        self.model = CVM()

        self.is_init = False
    

    def init(self,agents):
 
        self.total_agents_num = [None]*self.n_agents
        self.agent_pos_x = [None]*self.n_agents
        self.agent_pos_y = [None]*self.n_agents

        self.near_goal_threshold = 0.5

        self.is_init = True

    def find_next_action(self, obs, agents, target_agent_index ):
        if not self.is_init:   #Execute one time per init (complete simulation iteration)
            self.n_agents = len(agents)
            self.init(agents)
        #current_time = self.world_info().sim_time

        agent_index = target_agent_index

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
        
        #take the latest 8 observation  every 4 timestamp
        #select every 4 element from the end,    reverse it, and select the latest 8 entry
        observation_x_input = np.array( self.agent_pos_x )[:,::-4][:,::-1][:,-8:]
        observation_y_input = np.array( self.agent_pos_y )[:,::-4][:,::-1][:,-8:]
            
        #take the latest 8 observation  every 1 timestamp
        #observation_x_input     = np.array( self.agent_pos_x ) [ :  , -self.obs_seq_len: ] 
        #observation_y_input     = np.array( self.agent_pos_y ) [ :  , -self.obs_seq_len: ]

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
                print("POS_DIFFERENCE")
                print(pos_difference)
                dist_next_waypoint = ( pos_difference / (np.linalg.norm( pos_difference ,ord=1)+0.000001)  ) * ( agents[agent_ind].pref_speed * agents[agent_ind].dt_nominal )

                prev_history_agent_x = []
                prev_history_agent_y = []
                #Generate prev waypoints
                for prev_history_ind in range(prev_history_len):
                    prev_waypoint = prev_start - dist_next_waypoint * (prev_history_len - prev_history_ind )  *  1   #every 1 steps
                    
                    prev_history_agent_x.append( prev_waypoint[0] )
                    prev_history_agent_y.append( prev_waypoint[1] )

                prev_history_x.append( prev_history_agent_x )
                prev_history_y.append( prev_history_agent_y )
                
            #print("original observation=")
            #print(observation_x_input)
                
            observation_x_input = np.concatenate(( np.array(prev_history_x) ,observation_x_input),axis=1)
            observation_y_input = np.concatenate(( np.array(prev_history_y) ,observation_y_input),axis=1)

            #print("Since the observation length is only "+str(observation_len))
            #print("observation_x_input is now")
            #print(observation_x_input)

        #observation_x_input = observation_x_input - observation_x_input[agent_index,0]#observation_x_input[:,0][:,None]
        #observation_y_input = observation_y_input - observation_y_input[agent_index,0]#observation_y_input[:,0][:,None]

        #print("relative")
        #print(observation_x_input)

        ####FOR Observation input, its shape is [20, num_agents, 3(agent_id,x,y) ]
        #####HOWEVER, the target agent have to be the last within the timestamp array, e.g:  for 1,2,3 agents, if agent 2 is target agent, then for timestamp x, the array is [ [1,x,y],[3,x,y],[2,x,y] ]

        ####Basically, remove the target agent from the array and append to the end of the array

        observation_input = []
        
        for time_ind in range(self.obs_seq_len):
            temp = []
            for agent_ind in range(self.n_agents):
                temp.append([ observation_x_input[agent_ind][time_ind], observation_y_input[agent_ind][time_ind] ])
            observation_input.append( temp  )
            
        
        data = torch.from_numpy(np.transpose(np.array( observation_input ), (1, 2, 0)).astype(np.float32))
        fut = self.model.predictTraj(data)
        prediction = fut.detach().numpy()

        prediction = np.transpose(prediction, (0,2,1) )

##        print("prediction shape")
##        print(prediction.shape)
##
##        print(prediction)
        

        prediction_index = 0 #3 better in 10x10 #2 original test
        self.next_waypoint = prediction[agent_index][prediction_index]
        
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
