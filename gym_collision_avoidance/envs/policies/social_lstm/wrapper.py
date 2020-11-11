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
from utils import DataLoader
from helper import getCoef, sample_gaussian_2d, get_mean_error, get_final_error
from helper import *
from grid import getSequenceGridMask, getGridMask


def main():
    
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
    sample_args = parser.parse_args()
    
    #for drive run
    prefix = ''
    f_prefix = '.'
    if sample_args.drive is True:
      prefix='drive/semester_project/social_lstm_final/'
      f_prefix = 'drive/semester_project/social_lstm_final'

    #run sh file for folder creation
    if not os.path.isdir("log/"):
      print("Directory creation script is running...")
      subprocess.call([f_prefix+'/make_directories.sh'])

    method_name = get_method_name(sample_args.method)
    model_name = "LSTM"
    save_tar_name = method_name+"_lstm_model_"
    if sample_args.gru:
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
        saved_args = pickle.load(f)

    seq_lenght = sample_args.pred_length + sample_args.obs_length

    # Create the DataLoader object
    dataloader = DataLoader(f_prefix, 1, seq_lenght, forcePreProcess = True, infer=True)
    create_directories(os.path.join(result_directory, model_name), dataloader.get_all_directory_namelist())
    create_directories(plot_directory, [plot_test_file_directory])
    dataloader.reset_batch_pointer()



    
    dataset_pointer_ins = dataloader.dataset_pointer

    
    smallest_err = 100000
    smallest_err_iter_num = -1
    origin = (0,0)
    reference_point = (0,1)

    submission_store = [] # store submission data points (txt)
    result_store = [] # store points for plotting

    for iteration in range(sample_args.iteration):
        # Initialize net
        net = get_model(sample_args.method, saved_args, True)

        if sample_args.use_cuda:        
            net = net.cuda()

        # Get the checkpoint path
        checkpoint_path = os.path.join(save_directory, save_tar_name+str(sample_args.epoch)+'.tar')
        if os.path.isfile(checkpoint_path):
            print('Loading checkpoint')
            checkpoint = torch.load(checkpoint_path)
            model_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print('Loaded checkpoint at epoch', model_epoch)
        
        # For each batch
        iteration_submission = []
        iteration_result = []
        results = []
        submission = []

       
        # Variable to maintain total error
        total_error = 0
        final_error = 0


      
            
        x_seq = np.array([[[51.  ,  8.44,  7.05],
   [52.  ,  8.84,  8.09],
   [56.  ,  3.35, 12.32]], [[51.  ,  8.44,  7.05],
   [52.  ,  8.84,  8.09],
   [56.  ,  3.22, 12.42]], [[51.  ,  8.44,  7.05],
   [52.  ,  8.84,  8.09],
   [56.  ,  3.21, 12.49]], [[51.  ,  8.44,  7.05],
   [52.  ,  8.84,  8.09],
   [56.  ,  3.36, 12.65]], [[51.  ,  8.44,  7.05],
   [52.  ,  8.84,  8.09],
   [56.  ,  3.45, 12.79]], [[51.  ,  8.44,  7.05],
   [52.  ,  8.84,  8.09],
   [56.  ,  3.47, 12.79]], [[51.  ,  8.44,  7.05],
   [52.  ,  8.84,  8.09],
   [56.  ,  3.38, 12.79]], [[51.  ,  8.44,  7.05],
   [52.  ,  8.84,  8.09],
   [56.  ,  3.16, 12.78]], [[51., 0,0],
   [52., 0,0],
   [56., 0,0]], [[51., 0,0],
   [52., 0,0],
   [56., 0,0]], [[51., 0,0],
   [52., 0,0],
   [56., 0,0]], [[51., 0,0],
   [52., 0,0],
   [56., 0,0]], [[51., 0,0],
   [52., 0,0],
   [56., 0,0]], [[51., 0,0],
   [52., 0,0],
   [56., 0,0]], [[51., 0,0],
   [52., 0,0],
   [56., 0,0]], [[51., 0,0],
   [52., 0,0],
   [56., 0,0]], [[51., 0,0],
   [52., 0,0],
   [56., 0,0]], [[51., 0,0],
   [52., 0,0],
   [56., 0,0]], [[51., 0,0],
   [52., 0,0],
   [56., 0,0]], [[51., 0,0],
   [52., 0,0],
   [56., 0,0]]])


        target_id = 56
        numPedsList_seq=  np.array([3] * 20) #how many agents in the 20 timestamp [num agents] * 20
        PedsList_seq = np.array([[51.0, 52.0, 56.0]] * 20)



        
        
        print("X"*50)
        print("x_seq")
        print(x_seq)
        print(np.array(x_seq).shape)
        print("numPedsList_seq")
        print(numPedsList_seq)
        print("PedsList_seq")
        print(PedsList_seq)
        print("target_id")    #target is the agent that is running the algorithm
        print(target_id)
        
        dataloader.clean_test_data(x_seq, target_id, sample_args.obs_length, sample_args.pred_length)
        dataloader.clean_ped_list(x_seq, PedsList_seq, target_id, sample_args.obs_length, sample_args.pred_length)

        
        #get processing file name and then get dimensions of file
        #folder_name = dataloader.get_directory_name_with_pointer(d_seq)
        #dataset_data = dataloader.get_dataset_dimension(folder_name)
        dataset_data = [720, 576]

        #STARTS HERE
        #dense vector creation
        x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)

        print("======"*20)
        print("x_seq")
        print(x_seq.shape)
        print(x_seq)

        print("lookup_seq")
        print(len(lookup_seq))
        print(lookup_seq)
        
        #will be used for error calculation
        orig_x_seq = x_seq.clone() 
        
        target_id_values = orig_x_seq[0][lookup_seq[target_id], 0:2]


        
        #grid mask calculation
        if sample_args.method == 2: #obstacle lstm
            grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda, True)
        elif  sample_args.method == 1: #social lstm   
            grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda)

        #vectorize datapoints
        x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)

        # <------------- Experimental block ---------------->
        # x_seq = translate(x_seq, PedsList_seq, lookup_seq ,target_id_values)
        # angle = angle_between(reference_point, (x_seq[1][lookup_seq[target_id], 0].data.numpy(), x_seq[1][lookup_seq[target_id], 1].data.numpy()))
        # x_seq = rotate_traj_with_target_ped(x_seq, angle, PedsList_seq, lookup_seq)
        # grid_seq = getSequenceGridMask(x_seq[:sample_args.obs_length], dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, sample_args.use_cuda)
        # x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)


        if sample_args.use_cuda:
            x_seq = x_seq.cuda()

        # The sample function
        if sample_args.method == 3: #vanilla lstm
            # Extract the observed part of the trajectories
            obs_traj, obs_PedsList_seq = x_seq[:sample_args.obs_length], PedsList_seq[:sample_args.obs_length]
            ret_x_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq, PedsList_seq, saved_args, dataset_data, dataloader, lookup_seq, numPedsList_seq, sample_args.gru)

        else:
            # Extract the observed part of the trajectories
            obs_traj, obs_PedsList_seq, obs_grid = x_seq[:sample_args.obs_length], PedsList_seq[:sample_args.obs_length], grid_seq[:sample_args.obs_length]
            ret_x_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq, PedsList_seq, saved_args, dataset_data, dataloader, lookup_seq, numPedsList_seq, sample_args.gru, obs_grid)
        
        #revert the points back to original space
        ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, first_values_dict)
        
        # <--------------------- Experimental inverse block ---------------------->
        # ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, target_id_values, first_values_dict)
        # ret_x_seq = rotate_traj_with_target_ped(ret_x_seq, -angle, PedsList_seq, lookup_seq)
        # ret_x_seq = translate(ret_x_seq, PedsList_seq, lookup_seq ,-target_id_values)
        
        # Record the mean and final displacement error
        total_error += get_mean_error(ret_x_seq[1:sample_args.obs_length].data, orig_x_seq[1:sample_args.obs_length].data, PedsList_seq[1:sample_args.obs_length], PedsList_seq[1:sample_args.obs_length], sample_args.use_cuda, lookup_seq)
        final_error += get_final_error(ret_x_seq[1:sample_args.obs_length].data, orig_x_seq[1:sample_args.obs_length].data, PedsList_seq[1:sample_args.obs_length], PedsList_seq[1:sample_args.obs_length], lookup_seq)

        
        end = time.time()

        print("x_seq.data.cpu().numpy()")
        print(x_seq.data.cpu().numpy())
        print("ret_x_seq.data.cpu().numpy()")
        print(ret_x_seq.data.cpu().numpy())


        print("INPUT")
        print(ret_x_seq.data.cpu().numpy()[:8][:,2])
        print("PREDICTION")
        print(ret_x_seq.data.cpu().numpy()[-12:][:,2])

    


def sample(x_seq, Pedlist, args, net, true_x_seq, true_Pedlist, saved_args, dimensions, dataloader, look_up, num_pedlist, is_gru, grid = None):
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
    print("x_seq: "        , x_seq  )
    print("Pedlist: "      , Pedlist  )
    print("true_x_seq: "   , true_x_seq )
    print("true_Pedlist: " , true_Pedlist )
    print("dimensions: "   , dimensions )
    try:
        print("target_id: "    , target_id )
    except:
        pass

    print("==========================")
  
    # Number of peds in the sequence
    numx_seq = len(look_up)

    with torch.no_grad():
        # Construct variables for hidden and cell states
        hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
        if args.use_cuda:
            hidden_states = hidden_states.cuda()
        if not is_gru:
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            if args.use_cuda:
                cell_states = cell_states.cuda()
        else:
            cell_states = None


        ret_x_seq = Variable(torch.zeros(args.obs_length+args.pred_length, numx_seq, 2))

        # Initialize the return data structure
        if args.use_cuda:
            ret_x_seq = ret_x_seq.cuda()


        # For the observed part of the trajectory
        for tstep in range(args.obs_length-1):
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


        ret_x_seq[:args.obs_length, :, :] = x_seq.clone()

        # Last seen grid
        if grid is not None: #no vanilla lstm
            prev_grid = grid[-1].clone()

        #assign last position of observed data to temp
        #temp_last_observed = ret_x_seq[args.obs_length-1].clone()
        #ret_x_seq[args.obs_length-1] = x_seq[args.obs_length-1]

        # For the predicted part of the trajectory
        for tstep in range(args.obs_length-1, args.pred_length + args.obs_length-1):
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

            if args.use_cuda:
                list_of_x_seq = list_of_x_seq.cuda()
           
            #Get their predicted positions
            current_x_seq = torch.index_select(ret_x_seq[tstep+1], 0, list_of_x_seq)

            if grid is not None: #no vanilla lstm
                # Compute the new grid masks with the predicted positions
                if args.method == 2: #obstacle lstm
                    prev_grid = getGridMask(current_x_seq.data.cpu(), dimensions, len(true_Pedlist[tstep+1]),saved_args.neighborhood_size, saved_args.grid_size, True)
                elif  args.method == 1: #social lstm   
                    prev_grid = getGridMask(current_x_seq.data.cpu(), dimensions, len(true_Pedlist[tstep+1]),saved_args.neighborhood_size, saved_args.grid_size)

                prev_grid = Variable(torch.from_numpy(prev_grid).float())
                if args.use_cuda:
                    prev_grid = prev_grid.cuda()

        #ret_x_seq[args.obs_length-1] = temp_last_observed

        return ret_x_seq



if __name__ == '__main__':
    main()
