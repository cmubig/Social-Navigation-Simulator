import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time


def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)
                
def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr: 
            G = nx.from_numpy_matrix(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

######################################################   NEW
class Sam_TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir= os.path.dirname(__file__)+"/datasets/eth/test/biwi_eth_short.txt", obs_len=8, pred_len=8, skip=1, threshold=0.002,
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
        super(Sam_TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len #+ self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        #all_files = os.listdir(self.data_dir)
        #all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        all_files = [self.data_dir]  #changed to only read biwi_eth.txt for easy comparison
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        path = all_files[0] #only one

        data = read_file(path, delim)
        print(data)
        #print("original shape")
        print(data.shape)

        timestamp    = data[:,0]
        agent_id     = data[:,1]
        agent_pos_x  = data[:,2]
        agent_pos_y  = data[:,3]

        old_data = data
        data=None
        data = np.column_stack((timestamp,agent_id,agent_pos_x,agent_pos_y))#[:self.seq_len]  #[:16] since obs =8  pred = 8
        #print("assembled shape")
        #print(data.shape)
        
        frames = np.unique(data[:, 0]).tolist()   #extract the timestep column and remove any duplication, basically the unique timespan of the dataset
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
        num_sequences = 1
        #print("num_sequences")
        #print(num_sequences)

        #print("frame_data")
        #print(frame_data)

        idx = 0
        curr_seq_data = np.concatenate( frame_data, axis=0) #np.concatenate( frame_data[idx:idx + self.seq_len], axis=0)
        global old_curr_seq_data, curr_ped_seq
        old_curr_seq_data  = curr_seq_data

        #print("curr_seq_data")
        #print(curr_seq_data)
        
        peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

        #print("peds_in_curr_seq")
        #print(peds_in_curr_seq)
        
        self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
        curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                 self.seq_len))
        curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
        curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                   self.seq_len))
        num_peds_considered = 0
        _non_linear_ped = []

        print("peds_in_curr_seq")
        print(peds_in_curr_seq)
        for _, ped_id in enumerate(peds_in_curr_seq):
            #print("num_peds_considered ",num_peds_considered)
            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
            curr_ped_seq = np.around(curr_ped_seq, decimals=4)

            #print("curr_ped_seq now")
            #print(curr_ped_seq)
        
            pad_front = frames.index(curr_ped_seq[0, 0]) - idx
            pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

##            print("pad_front ",pad_front)
##            print("pad_end ",pad_end)
            if pad_end - pad_front != self.seq_len:
                continue

            
            
            curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])    #very possibly it wants to extract the position tuple, [x,y]*N from [time, id, x ,y] x N
            curr_ped_seq = curr_ped_seq
            # Make coordinates relative
            
            rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
            rel_curr_ped_seq[:, 1:] = \
                curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
            _idx = num_peds_considered
            
##            print("curr_seq")
##            print(curr_seq.shape)
##            print(curr_seq)
##            print("curr_ped_seq")
##            print(curr_ped_seq.shape)
##            print(curr_ped_seq)
##
##            print("_idx")
##            print(_idx)

            curr_seq[_idx,     :  , pad_front-pad_front:pad_end-pad_front] = curr_ped_seq
            curr_seq_rel[_idx, :  , pad_front-pad_front:pad_end-pad_front] = rel_curr_ped_seq
            # Linear vs Non-Linear Trajectory
            _non_linear_ped.append(
                poly_fit(curr_ped_seq, pred_len, threshold))
            curr_loss_mask[_idx, pad_front:pad_end] = 1
            num_peds_considered += 1

            

        if num_peds_considered > min_ped:
            non_linear_ped += _non_linear_ped
            num_peds_in_seq.append(num_peds_considered)
            loss_mask_list.append(curr_loss_mask[:num_peds_considered])
            seq_list.append(curr_seq[:num_peds_considered])
            seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
##        print(seq_list)
##        print(len(seq_list))
      
        
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.v_pred = [] 
        self.A_pred = [] 
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        #return 10 elements, exactly the number of elements in test.py's batch
        return out

################################################################# ^^NEW


######################################################   NEW
class Simulator_TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data, obs_len=8, pred_len=8, skip=1, threshold=0.002,
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
        super(Simulator_TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len #+ self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []


        timestamp    = data[:,0]
        agent_id     = data[:,1]
        agent_pos_x  = data[:,2]
        agent_pos_y  = data[:,3]

        old_data = data
        data=None
        data = np.column_stack((timestamp,agent_id,agent_pos_x,agent_pos_y))#[:self.seq_len]  #[:16] since obs =8  pred = 8
        #print("assembled shape")
        #print(data.shape)
        
        frames = np.unique(data[:, 0]).tolist()   #extract the timestep column and remove any duplication, basically the unique timespan of the dataset
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
        num_sequences = 1
        #print("num_sequences")
        #print(num_sequences)

        #print("frame_data")
        #print(frame_data)

        idx = 0
        curr_seq_data = np.concatenate( frame_data, axis=0) #np.concatenate( frame_data[idx:idx + self.seq_len], axis=0)
        global old_curr_seq_data, curr_ped_seq
        old_curr_seq_data  = curr_seq_data

        #print("curr_seq_data")
        #print(curr_seq_data)
        
        peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

        #print("peds_in_curr_seq")
        #print(peds_in_curr_seq)
        
        self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
        curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                 self.seq_len))
        curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
        curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                   self.seq_len))
        num_peds_considered = 0
        _non_linear_ped = []

        for _, ped_id in enumerate(peds_in_curr_seq):
            #print("num_peds_considered ",num_peds_considered)
            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
            curr_ped_seq = np.around(curr_ped_seq, decimals=4)

            #print("curr_ped_seq now")
            #print(curr_ped_seq)
        
            pad_front = frames.index(curr_ped_seq[0, 0]) - idx
            pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

            if pad_end - pad_front != self.seq_len:
                continue

            
            
            curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])    #very possibly it wants to extract the position tuple, [x,y]*N from [time, id, x ,y] x N
            curr_ped_seq = curr_ped_seq
            # Make coordinates relative
            
            rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
            rel_curr_ped_seq[:, 1:] = \
                curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
            _idx = num_peds_considered
            
##            print("curr_seq")
##            print(curr_seq.shape)
##            print(curr_seq)
##            print("curr_ped_seq")
##            print(curr_ped_seq.shape)
##            print(curr_ped_seq)
##
##            print("_idx")
##            print(_idx)

            curr_seq[_idx,     :  , pad_front-pad_front:pad_end-pad_front] = curr_ped_seq
            curr_seq_rel[_idx, :  , pad_front-pad_front:pad_end-pad_front] = rel_curr_ped_seq
            # Linear vs Non-Linear Trajectory
            _non_linear_ped.append(
                poly_fit(curr_ped_seq, pred_len, threshold))
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

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.v_pred = [] 
        self.A_pred = [] 
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        #return 10 elements, exactly the number of elements in test.py's batch
        return out

################################################################# ^^NEW


#test = TrajectoryDataset()
#test_sam = Sam_TrajectoryDataset()
