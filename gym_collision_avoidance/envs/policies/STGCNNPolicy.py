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

from gym_collision_avoidance.envs.policies.Social_STGCNN.utilsv2 import * 
from gym_collision_avoidance.envs.policies.Social_STGCNN.metrics import * 
from gym_collision_avoidance.envs.policies.Social_STGCNN.model import social_stgcnn
import copy

# Filter list by Boolean list 
# Using itertools.compress 
from itertools import compress

class STGCNNPolicy(InternalPolicy):
    def __init__(self):
        InternalPolicy.__init__(self, str="STGCNN")
        self.dt = Config.DT
        
        self.near_goal_threshold = 0.5
        self.is_init = False

        self.batch_size=128
        self.clip_grad=None
        self.dataset='eth'
        self.input_size=2
        self.kernel_size=3
        self.lr=0.01
        self.lr_sh_rate=150
        self.n_stgcnn=1
        self.n_txpcnn=5
        self.num_epochs=250
        self.obs_seq_len=8
        self.output_size=5
        self.pred_seq_len=12
        self.tag='social-stgcnn-eth'
        self.use_lrschd=True

        #Defining the model 
        self.model = social_stgcnn(n_stgcnn =self.n_stgcnn,n_txpcnn=self.n_txpcnn,
        output_feat=self.output_size,seq_len=self.obs_seq_len,
        kernel_size=self.kernel_size,pred_seq_len=self.pred_seq_len).cuda()
        self.model.load_state_dict(torch.load(os.path.dirname(__file__)+"/Social_STGCNN/checkpoint/social-stgcnn-eth/val_best.pth"))

        self.model.eval()

    def init(self,agents):
 
        self.total_agents_num = [None]*self.n_agents

        self.total_agents_num = [None]*self.n_agents
        self.agent_pos_x = [None]*self.n_agents
        self.agent_pos_y = [None]*self.n_agents
        
        self.near_goal_threshold = 0.5
        
        self.is_init = True

 
    def find_next_action(self, obs, agents, target_agent_index , full_agent_list, active_agent_mask):

        agents = full_agent_list
        
        if not self.is_init:   #Execute one time per init (complete simulation iteration)
            self.original_n_agents = self.n_agents = len(agents)

            self.init(agents)
        #current_time = self.world_info().sim_time

        full_list_agent_index = agent_index = target_agent_index

        #override due to dynamic number of agents
        self.n_agents = len(agents)


        #CHECK IF AT GOAL (0.5 radius for motion prediction algoritm)
        """ Set :code:`self.is_at_goal` if norm(pos_global_frame - goal_global_frame) <= near_goal_threshold """

        is_near_goal = (agents[agent_index].pos_global_frame[0] - agents[agent_index].goal_global_frame[0])**2 + (agents[agent_index].pos_global_frame[1] - agents[agent_index].goal_global_frame[1])**2 <= self.near_goal_threshold**2
        if is_near_goal:
            agents[agent_index].is_at_goal = is_near_goal
            return np.array([0,0])

        #if agents[0].step_num % 4 != 0: return [ agents[agent_index].speed_global_frame , 0 ] #agents[agent_index].delta_heading_global_frame ]

        #print("self.agent_pos_x")
        #print(self.agent_pos_x)

        #handle new agents due to dynamic number of agents, append length to make room for new agent, (use None for placeholder for newly added)
        if len(self.agent_pos_x)< self.n_agents:
            length_diff  = self.n_agents - len(self.agent_pos_x)
            #print("proceed to add "+str(length_diff))
            for add in range(length_diff):
                #print("add "+str(1+add))
                self.agent_pos_x.append( None )
                self.agent_pos_y.append( None )

        #print("AFTER self.agent_pos_x")
        #print(self.agent_pos_x)
       

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


        if ( agents[agent_index].step_num - agents[agent_index].start_step_num ) <= 3:


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
            #print("action")
            #print(action)
            return action
  
        #New agent history appended, but since the dimension might be less than already existed agent, add nan to make dimension regular.
        self.fill_agent_pos_x = self.agent_pos_x
        self.fill_agent_pos_y = self.agent_pos_y

        #print("self.fill_agent_pos_x")
        #print(self.fill_agent_pos_x)

        #print("number ",self.n_agents)
        for i in range(self.n_agents):
           
            #print("FIRST is ",len(self.fill_agent_pos_x[0]))
            #print(self.fill_agent_pos_x[0])
            #print("THIS  is ",len(self.fill_agent_pos_x[i]))
            #print(self.fill_agent_pos_x[i])

            
            #here we assume the existed (oldest, first) agent history array is the longest  #this assumption will lead to problem
            #if len(self.fill_agent_pos_x[0]) > len(self.fill_agent_pos_x[i]):
            if len(max(self.fill_agent_pos_x, key=len)) > len(self.fill_agent_pos_x[i]):
                
                #length_diff  = len(self.fill_agent_pos_x[0]) - len(self.fill_agent_pos_x[i])
                length_diff  = len(max(self.fill_agent_pos_x, key=len)) - len(self.fill_agent_pos_x[i])
                for add in range(length_diff):
                    #print("INSERT "+str(add+1))
                    self.fill_agent_pos_x[i].insert( 0 , np.nan )
                    self.fill_agent_pos_y[i].insert( 0 , np.nan ) 

        
            
        #Only take the latest 8 observation #test[:,-8:]
        #select every 4 element from the end,    reverse it, and select the latest 8 entry
        #if agents[0].step_num % 2 != 0: return [ agents[agent_index].speed_global_frame , 0 ]


        
        
        #observation_x_input = np.array( self.agent_pos_x )[:,::-4][:,::-1][:,-8:]
        #observation_y_input = np.array( self.agent_pos_y )[:,::-4][:,::-1][:,-8:]

        #print("active agent mask")
        #print(active_agent_mask)

        #print("before mask")
        #print(np.array( self.fill_agent_pos_x )[:,::-4][:,::-1][:,-8:])

        observation_x_input = np.array( self.fill_agent_pos_x )[ active_agent_mask ][:,::-4][:,::-1][:,-8:]
        observation_y_input = np.array( self.fill_agent_pos_y )[ active_agent_mask ][:,::-4][:,::-1][:,-8:]

        #print("after mask")
        #print(observation_x_input)


        #check if elements before index contains non active agents, if yes, remove them, thus calculate the index shift
        before_index = np.array(active_agent_mask)[:agent_index]

        #see how many non active agents are before index,  minus them calculate index shift
        agent_index = agent_index - len( before_index[ before_index==False ] )

        #assign new number of agents because of active_agent_mask
        self.n_agents = len(observation_x_input)

        agents = list(compress(agents, active_agent_mask))


        ########################################################################################################################

        #filter out the nans after extracting the latest 8 steps with 0.4s interval
        #observation_x_input = np.where( ~np.isnan(observation_x_input) , observation_x_input , 0)
        #observation_y_input = np.where( ~np.isnan(observation_y_input) , observation_y_input , 0)


        combined_history_x = []
        combined_history_y = []
        
        
##        if agent_index==0:
##            print("before observation_x_input")
##            print(observation_x_input)    
        
        for agent_ind in range(self.n_agents):


            #remove nan 
            #observation_x_input[agent_ind] = observation_x_input[agent_ind][ ~np.isnan(i) for i in observation_x_input[agent_ind] ]
            #observation_y_input[agent_ind] = observation_y_input[agent_ind][ ~np.isnan(i) for i in observation_y_input[agent_ind] ]

            #temporary version of observation_x_input[agent_ind]
            #temp_x = np.where( ~np.isnan(observation_x_input[agent_ind]) , observation_x_input[agent_ind] , 0)
            #temp_y = np.where( ~np.isnan(observation_y_input[agent_ind]) , observation_y_input[agent_ind] , 0)

            temp_x = np.array(  [x for x in observation_x_input[agent_ind] if str(x) != 'nan']   )
            temp_y = np.array(  [x for x in observation_y_input[agent_ind] if str(x) != 'nan']   )            

            #print("temp_x ")
            #print(len(temp_x) )
            #print(temp_x )
                
            observation_len = len(temp_x)

            #If previous traj provided, use previous traj for motion prediction's observation traj
            if agents[agent_ind].past_traj is not None:
                if observation_len < (self.obs_seq_len):

                    prev_history_len = self.obs_seq_len - observation_len           
                    #Generate prev waypoints

                    prev_history_x = np.array(agents[agent_ind].past_traj)[-prev_history_len:][:,0]
                    prev_history_y = np.array(agents[agent_ind].past_traj)[-prev_history_len:][:,1]

                    combined_history_x.append( np.concatenate(( np.array(prev_history_x) ,temp_x ))  )
                    combined_history_y.append( np.concatenate(( np.array(prev_history_y) ,temp_y ))  )

                else:

                    combined_history_x.append( temp_x  )
                    combined_history_y.append( temp_y  )

            #If there is not previous traj provided, then extrapolate points using start and goal to create past traj
            else:
                
                if observation_len < (self.obs_seq_len):
                    #Set up and generate previous history for each agent
                    #prev_start:  start position for prev history calculation
                    #prev_goal :   goal positoin for prev history calculation
                    
                    prev_start = np.array( [  temp_x[0] , temp_y[0]  ] )
                    prev_goal  = np.array( agents[agent_ind].goal_global_frame )
                    prev_history_len = self.obs_seq_len - observation_len
                    #generate prev waypoints using prefered speed
                    pos_difference = prev_goal - prev_start
                    dist_next_waypoint = ( pos_difference / (np.linalg.norm( pos_difference ,ord=1)+0.000001)  ) * ( agents[agent_ind].pref_speed * agents[agent_ind].dt_nominal )

                    prev_history_agent_x = []
                    prev_history_agent_y = []
                    #Generate prev waypoints
                    for prev_history_ind in range(prev_history_len):
                        prev_waypoint = prev_start - dist_next_waypoint * (prev_history_len - prev_history_ind ) *  4   #every 4 steps
                        
                        prev_history_agent_x.append( prev_waypoint[0] )
                        prev_history_agent_y.append( prev_waypoint[1] )

                    prev_history_x = np.array( prev_history_agent_x )
                    prev_history_y = np.array( prev_history_agent_y )

                    combined_history_x.append( np.concatenate(( np.array(prev_history_x) ,temp_x ))  )
                    combined_history_y.append( np.concatenate(( np.array(prev_history_y) ,temp_y ))  )

                else:

                    combined_history_x.append( temp_x  )
                    combined_history_y.append( temp_y  )


        combined_history_x = np.array( combined_history_x )
        combined_history_y = np.array( combined_history_y )

##        if agent_index==0:
##            print("after observation_x_input")
##            print(combined_history_x)        


        observation_x_input = combined_history_x
        observation_y_input = combined_history_y

            
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



        observation_len = len(observation_input)


        #if only target agent present, no other agent exist in observation
        if observation_len== self.obs_seq_len: return np.array([0,0])
        
        data = np.array(observation_input)#np.column_stack((observation_timestamp,observation_agent_id,observation_agent_pos_x,observation_agent_pos_y))

        dset_test = Simulator_TrajectoryDataset(data,
                skip=1,norm_lap_matr=True)

        loader_test = dset_test.__getitem__(0)
        batch = loader_test

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


        V_pred,_ = self.model(V_obs_tmp,A_obs.squeeze())

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

        try:
            mvnormal = torchdist.MultivariateNormal(mean,cov)
        except RuntimeError:
            print("Encountered cholesky_cuda: For batch 1: U(2,2) is zero, singular U.")
            cov=cov+0.000001
            mvnormal = torchdist.MultivariateNormal(mean,cov)
            #mvnormal = torch.distributions.LowRankMultivariateNormal(mean,cov)

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


        global V_pred_rel_to_abs
        V_pred = mvnormal.sample()
        V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
##        print("observation?")
##        print(V_x_rel_to_abs.shape)
##        print(V_x_rel_to_abs)
##
##        print("prediction?")
##        print(V_pred_rel_to_abs.shape)
##        print(V_pred_rel_to_abs)

        prediction_index = 2 #0
        self.next_waypoint =  V_pred_rel_to_abs[prediction_index][agent_index] #agents[agent_index].pos_global_frame +
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

    

        vel_global_frame = ((( self.next_waypoint - agents[agent_index].pos_global_frame)/(prediction_index+1))/4) / agents[agent_index].dt_nominal

        speed_global_frame = np.linalg.norm(vel_global_frame) 
        #if speed_global_frame > agents[agent_index].pref_speed: speed_global_frame = agents[agent_index].pref_speed
        
        if speed_global_frame > 1.5: speed_global_frame = 1.5
        if speed_global_frame < 0.5: speed_global_frame = 0.5
        #But in reality, the format of action is [speed, heading_delta]

        action = np.array([speed_global_frame, -heading_ego_frame])
        print("action")
        print(action)
       
        return action

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
