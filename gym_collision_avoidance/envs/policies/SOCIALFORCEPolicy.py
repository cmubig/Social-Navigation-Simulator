import os
import math
import sys
import torch
import numpy as np


from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import *

from gym_collision_avoidance.envs.policies import socialforce

import copy
import argparse

# Filter list by Boolean list 
# Using itertools.compress 
from itertools import compress

class SOCIALFORCEPolicy(InternalPolicy):
    def __init__(self):
        InternalPolicy.__init__(self, str="SOCIALFORCE")
        self.dt = Config.DT
        self.obs_seq_len = 8

        self.is_init = False


            

    def init(self,agents):
 
        self.total_agents_num = [None]*self.n_agents

        self.is_init = True

    def find_next_action(self, obs, agents, i , full_agent_list = None, active_agent_mask = None):

        agent_index = i

        #check if elements before index contains non active agents, if yes, remove them, thus calculate the index shift
        before_index = np.array(active_agent_mask)[:agent_index]

        #see how many non active agents are before index,  minus them calculate index shift
        agent_index = agent_index - len( before_index[ before_index==False ] )

        agents = list(compress(full_agent_list, active_agent_mask))

        
        observation_array = [] #observation array for social force, consist of N row of agents, each row = vector (x, y, v_x, v_y, d_x, d_y, [tau])
        
        if not self.is_init:   #Execute one time per init (complete simulation iteration)
            self.n_agents = len(agents)
            self.init(agents)


            #initialize the observation vector because when starts, social force seems to require a starting vel for agents to move
            for a in range(self.n_agents):
                pos_difference = agents[a].goal_global_frame -  agents[a].pos_global_frame    
                dist_next_waypoint = ( pos_difference / (np.linalg.norm( pos_difference ,ord=1)+0.0000001)  ) * ( agents[a].pref_speed )

                vel_next_waypoint  = dist_next_waypoint 
                
                observation_array.append( [  agents[a].pos_global_frame[0], agents[a].pos_global_frame[1], vel_next_waypoint[0], vel_next_waypoint[1], agents[a].goal_global_frame[0], agents[a].goal_global_frame[1]   ]  )


        else:
            ##added for dynamic num of agents compatibility
            self.n_agents = len(agents)
            self.init(agents)

            for a in range(self.n_agents):
                if  agents[a].speed_global_frame<= agents[a].pref_speed/3:
                    pos_difference = agents[a].goal_global_frame -  agents[a].pos_global_frame    
                    dist_next_waypoint = ( pos_difference / (np.linalg.norm( pos_difference ,ord=1)+0.0000001)  ) * ( agents[a].pref_speed )

                    vel_next_waypoint  = dist_next_waypoint 
                    
                    observation_array.append( [  agents[a].pos_global_frame[0], agents[a].pos_global_frame[1], vel_next_waypoint[0], vel_next_waypoint[1], agents[a].goal_global_frame[0], agents[a].goal_global_frame[1]   ]  )

                else:
                    
                    observation_array.append( [  agents[a].pos_global_frame[0], agents[a].pos_global_frame[1], agents[a].vel_global_frame[0], agents[a].vel_global_frame[1], agents[a].goal_global_frame[0], agents[a].goal_global_frame[1]   ]  )

        #print("goal")
        #print(agents[agent_index].goal_global_frame)
        
        initial_state = np.array( observation_array )
        s=None
        #s = socialforce.Simulator(initial_state, delta_t=0.1)
        s = socialforce.Simulator(initial_state, delta_t=0.1)
        states = np.stack([s.step().state.copy() for _ in range(1)]) #step one time only

        #print("states")
        #print(states)

        next_waypoint_x = states[:, agent_index, 0][0]
        next_waypoint_y = states[:, agent_index, 1][0]
        
        next_waypoint_vel_x = states[:, agent_index, 2][0]
        next_waypoint_vel_y = states[:, agent_index, 3][0]

        self.next_waypoint = np.array( [ next_waypoint_x , next_waypoint_y ] )
        
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

    

        vel_global_frame = np.array( [ next_waypoint_vel_x , next_waypoint_vel_y ] )#( self.next_waypoint - agents[agent_index].pos_global_frame) / agents[agent_index].dt_nominal

        speed_global_frame = np.linalg.norm(vel_global_frame)
        if speed_global_frame > agents[agent_index].pref_speed: speed_global_frame = agents[agent_index].pref_speed

        #But in reality, the format of action is [speed, heading_delta]

        action = np.array([speed_global_frame, -heading_ego_frame])
        #print("action")
        #print(action)
       
        return action

        #agents[agent_index].set_state( next_waypoint_x  , next_waypoint_y, next_waypoint_vel_x, next_waypoint_vel_y )
        
        #resultant_speed_global_frame         = agents[agent_index].speed_global_frame
        #resultant_delta_heading_global_frame = agents[agent_index].delta_heading_global_frame
        
        ###########################################################POSITION APPROACH##########################################################################
##        print("position")
##        print(agents[agent_index].pos_global_frame)
##        next_waypoint_x = states[:, agent_index, 0][0]
##        next_waypoint_y = states[:, agent_index, 1][0]
##
##        next_waypoint = np.array( [ next_waypoint_x, next_waypoint_y  ] )
##        print("next_waypoint")
##        print(next_waypoint)
##
##
##        
##        pos_difference = next_waypoint -  agents[agent_index].pos_global_frame    
##        dist_next_waypoint = ( pos_difference / (np.linalg.norm( pos_difference ,ord=1)+0.0000001)  ) * ( agents[agent_index].pref_speed * 0.1)
##
##        position_x = agents[agent_index].pos_global_frame[0] + dist_next_waypoint[0]
##        position_y = agents[agent_index].pos_global_frame[1] + dist_next_waypoint[1]
##        agents[agent_index].set_state( position_x , position_y )
##        
##        resultant_speed_global_frame         = agents[agent_index].speed_global_frame
##        resultant_delta_heading_global_frame = agents[agent_index].delta_heading_global_frame

        #Although documentation and code comment mentioned that action is consisted with  [heading delta, speed]
        #But in reality, the format of action is [speed, heading_delta]
        ###########################################################################################################################################

