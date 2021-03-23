import os
import pickle
import os
import pickle
import argparse
import time
import subprocess


import torch
from torch.autograd import Variable

import numpy as np

from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import *

from gym_collision_avoidance.envs.policies.social_lstm.utilsv2 import DataLoader
from gym_collision_avoidance.envs.policies.social_lstm.helper import getCoef, sample_gaussian_2d, get_mean_error, get_final_error
from gym_collision_avoidance.envs.policies.social_lstm.helper import *
from gym_collision_avoidance.envs.policies.social_lstm.grid import getSequenceGridMask, getGridMask


# Filter list by Boolean list 
# Using itertools.compress 
from itertools import compress


class SLSTMPolicy(InternalPolicy):
    def __init__(self):
        InternalPolicy.__init__(self, str="SLSTM")

        self.obs_seq_len = 8
        self.is_init = False
        '''
        parser = argparse.ArgumentParser()
        #Important parameter
        # Observed length of the trajectory parameter
        parser.add_argument('--obs_length', type=int, default=8,
                            help='Observed length of the trajectory')
        # Predicted length of the trajectory parameter
        parser.add_argument('--pred_length', type=int, default=12,
                            help='Predicted length of the trajectory')
        
        
        # Model to be loaded
    ##    parser.add_argument('--epoch', type=int, default=14,
    ##                        help='Epoch of model to be loaded')
        parser.add_argument('--epoch', type=int, default=29,
                            help='Epoch of model to be loaded')
        # cuda support
        parser.add_argument('--use_cuda', action="store_true", default=False,
                            help='Use GPU or not')
        # drive support
        parser.add_argument('--drive', action="store_true", default=False,
                            help='Use Google drive or not')
        # number of iteration -> we are trying many times to get lowest test error derived from observed part and prediction of observed
        # part.Currently it is useless because we are using direct copy of observed part and no use of prediction.Test error will be 0.
        parser.add_argument('--iteration', type=int, default=1,
                            help='Number of iteration to create test file (smallest test errror will be selected)')
        # gru model
        parser.add_argument('--gru', action="store_true", default=False,
                            help='True : GRU cell, False: LSTM cell')
        # method selection
        parser.add_argument('--method', type=int, default=1,
                            help='Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)')
        # Parse the parameters
        self.sample_args = parser.parse_args()
        '''
        self.obs_length = 8    #Observed length of the trajectory
        self.pred_length = 12  #Predicted length of the trajectory
        self.epoch = 29      #Epoch of model to be loaded
        self.use_cuda = False #False
        self.drive = False   #Use Google drive or not
        self.iteration = 1   #Number of iteration to create test file (smallest test errror will be selected)
        self.gru = False     #True : GRU cell, False: LSTM cell
        self.method = 1      #Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)

        
        
        
        #for drive run
        prefix = '../envs/policies/social_lstm'
        f_prefix = '../envs/policies/social_lstm'
        if self.drive is True:
          prefix='drive/semester_project/social_lstm_final/'
          f_prefix = 'drive/semester_project/social_lstm_final'

    ##    #run sh file for folder creation
    ##    if not os.path.isdir("log/"):
    ##      print("Directory creation script is running...")
    ##      subprocess.call([f_prefix+'/make_directories.sh'])

        method_name = get_method_name(self.method)
        model_name = "LSTM"
        save_tar_name = method_name+"_lstm_model_"
        if self.gru:
            model_name = "GRU"
            save_tar_name = method_name+"_gru_model_"

        print("Selected method name: ", method_name, " model name: ", model_name)

        # Save directory
        save_directory = os.path.join(f_prefix, 'model/', method_name, model_name)
        #plot directory for plotting in the future
        plot_directory = os.path.join(f_prefix, 'plot/', method_name, model_name)

        result_directory = os.path.join(f_prefix, 'result/', method_name)
        plot_test_file_directory = 'test'
        
        # Define the path for the config file for saved args
        with open(os.path.join(save_directory,'config.pkl'), 'rb') as f:
            self.saved_args = pickle.load(f)
            #print("Force set CUDA True")
            #self.saved_args.use_cuda = self.use_cuda

        seq_lenght = self.pred_length + self.obs_length

        # Create the self.dataloader object
        self.dataloader = DataLoader(f_prefix, 1, seq_lenght, forcePreProcess = True, infer=True)
        create_directories(os.path.join(result_directory, model_name), self.dataloader.get_all_directory_namelist())
        create_directories(plot_directory, [plot_test_file_directory])
        self.dataloader.reset_batch_pointer()



        
        dataset_pointer_ins = self.dataloader.dataset_pointer


    # Initialize net
        self.net = get_model(self.method, self.saved_args, True)

        if self.use_cuda:        
            self.net = self.net.cuda()

        # Get the checkpoint path
        checkpoint_path = os.path.join(save_directory, save_tar_name+str(self.epoch)+'.tar')
        if os.path.isfile(checkpoint_path):
            print('Loading checkpoint')
            checkpoint = torch.load(checkpoint_path)
            model_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['state_dict'])
            print('Loaded checkpoint at epoch', model_epoch)


    def init(self,agents):
        net = None
        self.net = get_model(self.method, self.saved_args, True)

        if self.use_cuda:        
            self.net = self.net.cuda()

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

        observation_x_input = combined_history_x - combined_history_x[agent_index,0]#observation_x_input[:,0][:,None]
        observation_y_input = combined_history_y - combined_history_y[agent_index,0]#observation_y_input[:,0][:,None]


        #observation_x_input = combined_history_x  #observation_x_input[:,0][:,None]
        #observation_y_input = combined_history_y  #observation_y_input[:,0][:,None]
        
        if observation_x_input.shape[0]==1: return np.array([0,0])


        ####FOR Observation input, its shape is [20, num_agents, 3(agent_id,x,y) ]
        #####HOWEVER, the target agent have to be the last within the timestamp array, e.g:  for 1,2,3 agents, if agent 2 is target agent, then for timestamp x, the array is [ [1,x,y],[3,x,y],[2,x,y] ]

        ####Basically, remove the target agent from the array and append to the end of the array

        observation_input = []
        for time_ind in range(self.obs_seq_len):
            temp = []
            for agent_ind in range(self.n_agents):
                temp.append([ agent_ind, observation_x_input[agent_ind][time_ind], observation_y_input[agent_ind][time_ind] ])

            observation_input.append( temp  )

        for time_ind in range(self.obs_seq_len,20):
            temp = []
            for agent_ind in range(self.n_agents):
                temp.append([ agent_ind, 0, 0 ])

            observation_input.append( temp  )
                
        x_seq = np.array(observation_input).astype(np.float32)

        target_id = agent_index
        #print("agent_index")
        #print(agent_index)
        numPedsList_seq=  np.array([self.n_agents] * 20).astype(np.float32) #how many agents in the 20 timestamp [num agents] * 20
        pedlist = np.arange(self.n_agents).astype(np.float32)
        
        PedsList_seq = np.tile(pedlist,(20,1))
#[[ 0.  1.  2.  3.  4.  5.  6. 14.  8.  9. 10. 11. 12. 13.  7.

##        print("X"*50)
##        print("x_seq")
##        print(x_seq)
##        print(np.array(x_seq).shape)
##        print("numPedsList_seq")
##        print(numPedsList_seq)
##        print("PedsList_seq")
##        print(PedsList_seq)
##        print("target_id")    #target is the agent that is running the algorithm
##        print(target_id)
        
        self.dataloader.clean_test_data(x_seq, target_id, self.obs_length, self.pred_length)
        self.dataloader.clean_ped_list(x_seq, PedsList_seq, target_id, self.obs_length, self.pred_length)

        
        #get processing file name and then get dimensions of file
        #folder_name = self.dataloader.get_directory_name_with_pointer(d_seq)
        #dataset_data = self.dataloader.get_dataset_dimension(folder_name)
        dataset_data = [720, 576]

        #STARTS HERE
        #dense vector creation
        x_seq, lookup_seq = self.dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
##        print("======"*20)
##        print("x_seq2")
##        print(x_seq.shape)
##        print(x_seq)
##
##        print("lookup_seq")
##        print(len(lookup_seq))
##        print(lookup_seq)
        
        #will be used for error calculation
        orig_x_seq = x_seq.clone() 

        
        #grid mask calculation
        if self.method == 2: #obstacle lstm
            grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, self.saved_args.neighborhood_size, self.saved_args.grid_size, self.saved_args.use_cuda, True)
        elif  self.method == 1: #social lstm   
            grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, self.saved_args.neighborhood_size, self.saved_args.grid_size, self.saved_args.use_cuda)

        #vectorize datapoints
        x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)

        if self.use_cuda:
            x_seq = x_seq.cuda()

        # The sample function
        if self.method == 3: #vanilla lstm
            # Extract the observed part of the trajectories
            obs_traj, obs_PedsList_seq = x_seq[:self.obs_length], PedsList_seq[:self.obs_length]
            ret_x_seq = self.sample(obs_traj, obs_PedsList_seq,  self.net, x_seq, PedsList_seq, self.saved_args, dataset_data, self.dataloader, lookup_seq, numPedsList_seq, self.gru)

        else:
            # Extract the observed part of the trajectories
            obs_traj, obs_PedsList_seq, obs_grid = x_seq[:self.obs_length], PedsList_seq[:self.obs_length], grid_seq[:self.obs_length]
            ret_x_seq = self.sample(obs_traj, obs_PedsList_seq,  self.net, x_seq, PedsList_seq, self.saved_args, dataset_data, self.dataloader, lookup_seq, numPedsList_seq, self.gru, obs_grid)
        
        #revert the points back to original space
        ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, first_values_dict)

##        print("x_seq.data.cpu().numpy()")
##        print(x_seq.data.cpu().numpy())
##        print("ret_x_seq.data.cpu().numpy()")
##        print(ret_x_seq.data.cpu().numpy())


        #print("INPUT")
        #print(ret_x_seq.data.cpu().numpy()[:8][:,agent_index])
        #print("PREDICTION")
        #print(ret_x_seq.data.cpu().numpy()[-12:][:,agent_index])

        prediction_index = 0
        self.next_waypoint = np.array( agents[agent_index].goal_global_frame ) + ret_x_seq.data.cpu().numpy()[-12:][:,agent_index][prediction_index]
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
        heading_ego_frame = wrap( agents[agent_index].heading_global_frame - ref_prll_angle_global_frame)

    
        vel_global_frame = (( goal_direction)/4) / agents[agent_index].dt_nominal

        speed_global_frame = np.linalg.norm(vel_global_frame) 
        print("calc speed")
        print(speed_global_frame)
        #if speed_global_frame > agents[agent_index].pref_speed: speed_global_frame = agents[agent_index].pref_speed

        if speed_global_frame > 1.5: speed_global_frame = 1.5
        if speed_global_frame < 0.5: speed_global_frame = 0.5

        #But in reality, the format of action is [speed, heading_delta]

        action = np.array([speed_global_frame, -heading_ego_frame])
        print("action")
        print(action)
       
        return action

    


    def sample(self, x_seq, Pedlist, net, true_x_seq, true_Pedlist, saved_args, dimensions, dataloader, look_up, num_pedlist, is_gru, grid = None):
        #x_seq, d_seq ,numPedsList_seq, PedsList_seq, target_id = x[0], d[0], numPedsList[0], PedsList[0], target_ids[0]

        #x_seq: Input positions
        #d_seq:
        '''
        The sample function
        params:
        x_seq: Input positions
        Pedlist: Peds present in each frame
        args: arguments
        net: The model
        true_x_seq: True positions
        true_Pedlist: The true peds present in each frame 
        saved_args: Training arguments
        dimensions: The dimensions of the dataset
        target_id: ped_id number that try to predict in this sequence
        '''

        '''

        look_up: Number of peds in the sequence

        '''
      
        # Number of peds in the sequence
        numx_seq = len(look_up)

        with torch.no_grad():
            # Construct variables for hidden and cell states
            hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            if self.use_cuda:
                hidden_states = hidden_states.cuda()
            if not is_gru:
                cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
                if self.use_cuda:
                    cell_states = cell_states.cuda()
            else:
                cell_states = None


            ret_x_seq = Variable(torch.zeros(self.obs_length+self.pred_length, numx_seq, 2))

            # Initialize the return data structure
            if self.use_cuda:
                ret_x_seq = ret_x_seq.cuda()


            # For the observed part of the trajectory
            for tstep in range(self.obs_length-1):
                if grid is None: #vanilla lstm
                   # Do a forward prop
                    out_obs, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 2), hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
                else:
                    # Do a forward prop
                    out_obs, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 2), [grid[tstep]], hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
                # loss_obs = Gaussian2DLikelihood(out_obs, x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]])

                # Extract the mean, std and corr of the bivariate Gaussian
                mux, muy, sx, sy, corr = getCoef(out_obs)
                # Sample from the bivariate Gaussian
                next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, true_Pedlist[tstep], look_up)
                ret_x_seq[tstep + 1, :, 0] = next_x
                ret_x_seq[tstep + 1, :, 1] = next_y


            ret_x_seq[:self.obs_length, :, :] = x_seq.clone()

            # Last seen grid
            if grid is not None: #no vanilla lstm
                prev_grid = grid[-1].clone()

            #assign last position of observed data to temp
            #temp_last_observed = ret_x_seq[args.obs_length-1].clone()
            #ret_x_seq[args.obs_length-1] = x_seq[args.obs_length-1]

            # For the predicted part of the trajectory
            for tstep in range(self.obs_length-1, self.pred_length + self.obs_length-1):
                # Do a forward prop
                if grid is None: #vanilla lstm
                    outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, 2), hidden_states, cell_states, [true_Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
                else:
                    outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, 2), [prev_grid], hidden_states, cell_states, [true_Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)

                # Extract the mean, std and corr of the bivariate Gaussian
                mux, muy, sx, sy, corr = getCoef(outputs)
                # Sample from the bivariate Gaussian
                next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, true_Pedlist[tstep], look_up)

                # Store the predicted position
                ret_x_seq[tstep + 1, :, 0] = next_x
                ret_x_seq[tstep + 1, :, 1] = next_y

                # List of x_seq at the last time-step (assuming they exist until the end)
                true_Pedlist[tstep+1] = [int(_x_seq) for _x_seq in true_Pedlist[tstep+1]]
                next_ped_list = true_Pedlist[tstep+1].copy()
                converted_pedlist = [look_up[_x_seq] for _x_seq in next_ped_list]
                list_of_x_seq = Variable(torch.LongTensor(converted_pedlist))

                if self.use_cuda:
                    list_of_x_seq = list_of_x_seq.cuda()
               
                #Get their predicted positions
                current_x_seq = torch.index_select(ret_x_seq[tstep+1], 0, list_of_x_seq)

                if grid is not None: #no vanilla lstm
                    # Compute the new grid masks with the predicted positions
                    if self.method == 2: #obstacle lstm
                        prev_grid = getGridMask(current_x_seq.data.cpu(), dimensions, len(true_Pedlist[tstep+1]),saved_args.neighborhood_size, saved_args.grid_size, True)
                    elif  self.method == 1: #social lstm   
                        prev_grid = getGridMask(current_x_seq.data.cpu(), dimensions, len(true_Pedlist[tstep+1]),saved_args.neighborhood_size, saved_args.grid_size)

                    prev_grid = Variable(torch.from_numpy(prev_grid).float())
                    if self.use_cuda:
                        prev_grid = prev_grid.cuda()

            #ret_x_seq[args.obs_length-1] = temp_last_observed

            return ret_x_seq

