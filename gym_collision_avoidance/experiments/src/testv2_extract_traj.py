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



class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()   #extract the timestep column and remove any duplication, basically the unique timespan of the dataset
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                   # _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        self.seq_list = seq_list


def real_dataset_traj(dataset_name="hotel"):
    data_dir = os.path.dirname(__file__)+"/datasets/"+dataset_name+"/test/"
    delim='\t'
    
    all_files = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, _path) for _path in all_files]

    print(all_files)

    for path in all_files:
        data = read_file(path, delim)
##        frames = np.unique(data[:, 0]).tolist()   #extract the timestep column and remove any duplication, basically the unique timespan of the dataset
##        frame_data = []
##        for frame in frames:
##            frame_data.append(data[frame == data[:, 0], :])
##        num_sequences = int(
##            math.ceil(len(frames) + 1) )

        agent_list = np.unique(data[:, 1]).tolist()
        print(agent_list)

        frame_data = []
        for frame in agent_list:
            frame_data.append(data[frame == data[:, 1], :])

        #start filtering data
        filtered_frame_data = []
        #iterate through agents
        for i,record in enumerate(frame_data):
            diff = np.array(record[-1][2:]) - np.array(record[0][2:])

            #remove record if start and goal are too close     or if the agent record have less than 8 timestamps
            if (np.linalg.norm(diff) < 2) or (len(record)<8): continue
            filtered_frame_data.append( record[:,2:]  )
                
                
            
        


        return np.array(filtered_frame_data)

output=real_dataset_traj()




#Data prep     
##obs_seq_len = 40#8
##pred_seq_len = 12
##dataset = "hotel"
##data_set = os.path.dirname(__file__)+"/datasets/"+dataset+"/test/"
##
##dset_test = TrajectoryDataset(
##        data_set,
##        obs_len=obs_seq_len,
##        pred_len=pred_seq_len,
##        skip=1,norm_lap_matr=True)

##
##
##class real_dataset_traj(object):
##
##    def __init__( self, dataset_name="hotel", population_density=None ):
##
##        self.spawn_distance_threshold = 0.7
##        
##        data_dir = os.path.dirname(__file__)+"/datasets/"+dataset_name+"/test/"
##        delim='\t'
##        
##        all_files = os.listdir(data_dir)
##        all_files = [os.path.join(data_dir, _path) for _path in all_files]
##
##        print(all_files)
##
##
##        filtered_frame_data = []
##        for path in all_files:
##            data = read_file(path, delim)
##
##            agent_list = np.unique(data[:, 1]).tolist()
##            print(agent_list)
##
##            frame_data = []
##            for frame in agent_list:
##                frame_data.append(data[frame == data[:, 1], :])
##
##            #start filtering data
##            
##            #iterate through agents
##            for i,record in enumerate(frame_data):
##                diff = np.array(record[-1][2:]) - np.array(record[0][2:])
##
##                #remove record if start and goal are too close     or if the agent record have less than 8 timestamps
##                if (np.linalg.norm(diff) < 2) or (len(record)<8): continue
##                filtered_frame_data.append( record[:,2:]  )
##
##
##        self.dataset_traj = np.array(filtered_frame_data)
##        
##    def seed(self):
##        self.random_seed+=1
##        np.random.seed(self.random_seed)
##        
##    def pick_one(self, agents):
##        #get target agent's start, goal, current position, so that when picking new traj, prevent choosing one that is close to these points
##        num_traj = len(self.dataset_traj)
##
##        collide = True
##        while collide:
##            self.seed()
##            index = np.random.randint(0,num_traj)
##
##            picked_agent_traj = self.dataset_traj[index]
##
##            start_x,start_y = picked_agent_traj[7] #[0]  not 0, because we want history before the starting point, so we use [7], the number 8 element as starting point
##            goal_x,goal_y   = picked_agent_traj[-1]
##
##            past_traj = picked_agent_traj[:8]
##
##            #past_goal_points = np.array(scenario)[:,4:6].astype(np.float)
##            past_start_points          =   np.array( [ agent.start_global_frame for agent in agents ] ) .astype(np.float)
##            past_goal_points           =   np.array( [ agent.goal_global_frame for agent in agents ] ) .astype(np.float)
##            past_current_points        =   np.array( [ agent.pos_global_frame for agent in agents ] ) .astype(np.float)
##
##            past_start_points = np.concatenate((past_start_points,past_goal_points,past_current_points))
##
##            if len(past_goal_points)==0: break
##
##            closet_distance_to_other_start_point = 999
##            for past_start_point in past_start_points:
##                distance = np.linalg.norm(past_start_point - np.array([start_x,start_y]) )
##                if distance < closet_distance_to_other_start_point: closet_distance_to_other_start_point = distance
##
##            if closet_distance_to_other_start_point >= self.spawn_distance_threshold: #0.7 #default=1, reduce if it is a crowded scene
##                collide=False
##                break
##
##        return [ start_x, start_y, goal_x, goal_y, past_traj ]
##
##   def pick_start(self, population_density, policy_list, x_min, x_max, y_min, y_max, pref_speed, agent_radius, start_timestamp, agents, random_seed=0, num_agents_override=None):
##
##        self.random_seed      = random_seed*1000
##        np.random.seed(self.random_seed)
##
##        if num_agents_override is None:
##            self.num_agents       = int(round(population_density * ( ( x_max - x_min )  *  ( y_max - y_min )  )))
##            print(str(self.num_agents)+" agents created under population density of "+str(population_density))
##            if self.num_agents <=1: self.num_agents=2
##            #override
##            if self.num_agents <=2: self.num_agents=3
##        else:
##            self.num_agents = num_agents_override
##            
##        self.policy_list      = policy_list
##        self.x_min            = x_min
##        self.x_max            = x_max
##        self.y_min            = y_min
##        self.y_max            = y_max
##        self.pref_speed       = pref_speed
##        self.agent_radius     = agent_radius
##        self.start_timestamp  = start_timestamp
##
##        scenario = []
##
##        ###while loop till we find the start position for all agents so it doesnot collide when spawn
##    
##        for i in range(self.num_agents):
##
##            #modify the random seed for each agent
##            
##            if type(self.policy_list)==list:
##                policy = self.policy_list[i]
##            else:
##                policy = self.policy_list
##
##            if type(self.start_timestamp)==list:
##                start_timestamp = self.start_timestamp[i]
##            else:
##                start_timestamp = self.start_timestamp
##
##            if type(self.pref_speed)==list:
##                pref_speed = self.pref_speed[i] #* speed_rescaler
##            else:
##                pref_speed = self.pref_speed #* speed_rescaler
##
##            #make sure speed is not too slow
##            speed_threshold = 0.2
##            if (pref_speed < speed_threshold ): pref_speed = speed_threshold
##
##            #To make sure it is not colliding with other start point
##            collide = True
##            while collide:
##                self.seed()
##                num_traj = len(self.dataset_traj)
##                index = np.random.randint(0,num_traj)
##
##                picked_agent_traj = self.dataset_traj[index]
##
##                start_x,start_y = picked_agent_traj[7] #[0]  not 0, because we want history before the starting point, so we use [7], the number 8 element as starting point
##                goal_x,goal_y   = picked_agent_traj[-1]
##
##                past_traj = picked_agent_traj[:8]
##
##                #First agent traj added, no collision yet, so add directly
##                if len(scenario)==0:
##                    collide=False
##                    break
##
##                past_start_points = np.array(scenario)[:,2:4].astype(np.float)
##                past_goal_points  = np.array(scenario)[:,4:6].astype(np.float)
##
##                past_start_points = np.concatenate((past_start_points,past_goal_points))
##
##                if len(past_start_points)==0: break
##                
##                closet_distance_to_other_start_point = 999
##                for past_start_point in past_start_points:
##                    distance = np.linalg.norm(past_start_point - np.array([start_x,start_y]) )
##                    if distance < closet_distance_to_other_start_point: closet_distance_to_other_start_point = distance
##
##                if closet_distance_to_other_start_point >= self.spawn_distance_threshold: #0.7 #default=1, reduce if it is a crowded scene
##                    collide=False
##                    break
##
##
##            
##            #print([ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ])
##            scenario.append( [ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, past_traj ] )
##
##        return [ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, past_traj ]
##        
##
##
##
##
##
##        
##        
##
##output=real_dataset_traj()
