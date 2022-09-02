import os
import numpy as np


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_name"          , type=str, required=True, help="the output folder name for this run of experiment... e.g. exp1_ETH_CADRL or exp2_0.3_CADRL")
parser.add_argument("--experiment_num"       , type=int, required=True, help="which experiment are we running (1,2,???)")
parser.add_argument("--algorithm_name"       , type=str, required=True, help="which algorithm are we using? CADRL,RVO,SOCIALFORCE,  SPEC, STGCNN, SLSTM, SOCIALGAN")

parser.add_argument("--experiment_iteration" , type=int, required=True, help="for each experiment, how many iteration (how many scenario, from start to goal) should it generate?")
parser.add_argument("--timeout"              , type=int, required=True, help="how many seconds for the experiment to terminate, and declare on-going agents timeout?")

parser.add_argument("--population_density"   , type=float, required=False, default="-1.0" ,help="under exp2, what population density should be used?")
parser.add_argument("--dataset_name"         , type=str  , required=False, default="None" ,help="under exp1, for the exp settings of algortihms, which dataset should they mimick?")
args = parser.parse_args()
print(args)

experiment_number = args.experiment_num
algorithm_name    = args.algorithm_name

experiment_iteration_num = args.experiment_iteration
timeout = args.timeout

dataset_name = args.dataset_name
population_density = args.population_density

os.environ["global_timeout"]             = str(timeout)
os.environ["global_experiment_number"]   = str(experiment_number)
os.environ["global_dataset_name"]        = str(dataset_name)
os.environ["global_population_density"]  = str(population_density)

##print("AFter set")
##print(os.environ["global_timeout"] )
##print(os.environ["global_experiment_number"])
##print(os.environ["global_dataset_name"])


import pickle
from tqdm import tqdm

os.environ['GYM_CONFIG_CLASS'] = 'Custom' 


from gym_collision_avoidance.experiments.src.master_config_deploy import Master_Config
from gym_collision_avoidance.experiments.src.master_config_deploy import Scenario_Config

##master_config   = Master_Config(    timeout  )
##scenario_config = Scenario_Config(  experiment_number, algorithm_name, experiment_iteration_num, dataset_name=None, population_density=None)

#master config and scenario config from "master_config_deploy.py"
master_config   = Master_Config()

# scenario config for algorith, exp no etc
scenario_config = Scenario_Config(  experiment_number, algorithm_name, experiment_iteration_num, dataset_name, population_density)


from gym_collision_avoidance.envs import Config
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env, store_stats, policies

LINEAR      = 0
NonCooperativePolicy = 0
SOCIALFORCE = 1 
RVO         = 2 
CADRL       = 3 
NAVIGAN     = 4

CVM         = 16 
SLSTM       = 17 
SOCIALGAN   = 18 
STGCNN      = 19 
SPEC        = 20 

motion_list = [ "CVM" , "SLSTM" , "SOCIALGAN" , "STGCNN", "SPEC" ]
navigation_list = [ "LINEAR" , "SOCIALFORCE" , "RVO" , "CADRL" ,"NAVIGAN" ]

###########################Dataset output##################################

# visual trajectories add to the plot
def add_traj(agents, trajs, dt, traj_i, max_ts):
    agent_i = 0
    other_agent_i = (agent_i + 1) % 2
    agent = agents[agent_i]
    other_agent = agents[other_agent_i]
    max_t = int(max_ts[agent_i])
    future_plan_horizon_secs = 3.0
    future_plan_horizon_steps = int(future_plan_horizon_secs / dt)

    for t in range(max_t):
        robot_linear_speed = agent.global_state_history[t, 9]
        robot_angular_speed = agent.global_state_history[t, 10] / dt

        t_horizon = min(max_t, t+future_plan_horizon_steps)
        future_linear_speeds = agent.global_state_history[t:t_horizon, 9]
        future_angular_speeds = agent.global_state_history[t:t_horizon, 10] / dt
        predicted_cmd = np.dstack([future_linear_speeds, future_angular_speeds])

        future_positions = agent.global_state_history[t:t_horizon, 1:3]

        d = {
            'control_command': np.array([
                robot_linear_speed,
                robot_angular_speed
                ]),
            'predicted_cmd': predicted_cmd,
            'future_positions': future_positions,
            'pedestrian_state': {
                'position': np.array([
                    other_agent.global_state_history[t, 1],
                    other_agent.global_state_history[t, 2],
                    ]),
                'velocity': np.array([
                    other_agent.global_state_history[t, 7],
                    other_agent.global_state_history[t, 8],
                    ])
            },
            'robot_state': np.array([
                agent.global_state_history[t, 1],
                agent.global_state_history[t, 2],
                agent.global_state_history[t, 10],
                ]),
            'goal_position': np.array([
                agent.goal_global_frame[0],
                agent.goal_global_frame[1],
                ])
        }
        trajs[traj_i].append(d)

    return trajs
#############################################

def reset_env(env, one_env, test_case_fn, test_case_args, test_case,test_case_index, num_agents, agent_policies_list, prev_agents=None, start_from_last_configuration=True):
    #commented this to consider every scenario as a separate one, otherwise the policy list never update
    #if prev_agents is None:

    policy_name_list = [ policies[i]['policy'] for i in agent_policies_list]
    #print(policy_name_list)
    sensor_name_list = [ policies[i]['sensors'][0] for i in agent_policies_list]
    
    prev_agents = tc.small_test_suite(num_agents=num_agents, test_case= test_case, test_case_index=test_case_index, policies=policy_name_list, agents_sensors=sensor_name_list)


    for i in range(len(prev_agents)):

        policy = agent_policies_list[i]
        agent  = prev_agents[i]
        

        if 'checkpt_name' in policies[policy]:
            agent.policy.env = env
            agent.policy.initialize_network(**policies[policy])
        if 'sensor_args' in policies[policy]:
            for sensor in agent.sensors:
                sensor.set_args(policies[policy]['sensor_args'])

    #For filling in the parameter for certain scenario generator            
    test_case_args['agents']              = prev_agents
    test_case_args['test_case']           = test_case
    test_case_args['test_case_index']     = test_case_index 
    test_case_args['num_agents']          = num_agents
    
    #one_env.plot_policy_name = "generated_biwi_ETH" #perhaps it is just the name?
    agents = test_case_fn(**test_case_args)
    one_env.set_agents(agents)
    init_obs = env.reset()
    one_env.test_case_index = test_case_index
    return init_obs

def main():
    
    # initialize custom testcase
    test_case_fn = tc.custom_formation #formation (agents, test_case, test_case_index, num_agents)
    test_case_args = {}
    

    #Formation
    #env test_cases formation
    #For sceneraio generator, can be programmed
    #formation is for 6 agents

    #folder_policy_name = np.unique( np.array(list(scenario_config.scenario)[0])[:,1] )[0]
    folder_policy_name = args.output_name

    env, one_env = create_env()
    dt = one_env.dt_nominal #retrieve framerate
    
    experiment_type = str(folder_policy_name).split("_")[0]

    if any(motion_policy in folder_policy_name for motion_policy in motion_list):
        policy_type = "motion"
        
    if any(navigation_policy in folder_policy_name for navigation_policy in navigation_list):
        policy_type = "navigation"
    
    
    #file_dir = file_dir_template = os.path.dirname(os.path.realpath(__file__)) + '/../results/sam_traj_output_ex/'
    file_dir = file_dir_template = os.path.dirname(os.path.realpath(__file__)) + '/../results/'+experiment_type+"/"+policy_type+"/"+str(folder_policy_name)+'/'

    #one_env.set_plot_save_dir(  os.path.dirname(os.path.realpath(__file__)) + '/../results/sam_traj_output_ex/')
    one_env.set_plot_save_dir(  os.path.dirname(os.path.realpath(__file__)) + '/../results/'+experiment_type+"/"+policy_type+"/"+str(folder_policy_name)+'/')


    pkl_dir = file_dir + '/test/'
    os.makedirs(pkl_dir, exist_ok=True)

    num_of_test_case = len(list(scenario_config.scenario))

    trajs = [[] for _ in range(num_of_test_case)]
    
    #polices will be fixed if use this
    #agent_policies_list = master_config.POLICIES_TO_TEST

    overall_record_list = []
   
    np.random.seed(0)
    prev_agents = None
    for test_case_index in range(num_of_test_case):

        overall_record = {}
        #if test_case_index == 4:
        #    print("caught")
        #print(list(scenario_config.scenario))
        #print(test_case_index)
        #print([test_case_index])
        #print(list(scenario_config.scenario)[test_case_index])


        agent_policies_list = np.array(list(scenario_config.scenario)[test_case_index])[:,1] #this scenario's policies list
        num_agents = len(agent_policies_list)
        print("num_agents")
        print(num_agents)

        print("agent_policies_list")
        print(agent_policies_list)

        print("data")
        print(np.array(list(scenario_config.scenario)[test_case_index]))

        test_case = scenario_config.scenario
        
        _ = reset_env(env, one_env, test_case_fn, test_case_args, test_case, test_case_index, num_agents, agent_policies_list,  prev_agents)

        one_env.plot_policy_name = str(np.unique(agent_policies_list)[0]) #perhaps it is just the name?
        print("before episode")
        episode_stats, prev_agents, presence_dict = run_episode(env, one_env)
        print("after episode, episode stats: ")
        # print(episode_stats)
        #episode_stats returned something like this
        '''
{'total_reward': array([  1.   ,   0.618,  -3.722,   1.   ,   0.813,  -1.878, -11.319,
     0.128,   1.   ,   0.42 ,   0.61 ,  -1.155,  -7.031,  -7.614,
    -3.972,  -6.907,  -0.25 ,   1.   ,   1.   ,  -0.805, -11.   ,
    -1.368,   1.   ,   1.   ]), 'steps': 339, 'num_agents': 24, 'time_to_goal': array([25.3, 19. , 11.2, 15.1, 28.1, 22.8, 20.8, 20.6,  4.8,  6.4, 18. ,
    9.9, 18.3, 11.2, 14.1, 14.6,  9.9, 33.9, 19.9, 20.1, 24.9, 24.4,
   17.7, 24.3]), 'total_time_to_goal': 435.3000000000005, 'extra_time_to_goal': array([ 7.224,  6.7  ,  3.193,  7.093, 15.8  ,  4.724,  4.147, 10.492,
    0.833,  2.433,  7.892, -6.753,  1.647,  1.092, 10.133, 10.633,
   -0.208, 17.247,  1.824,  7.8  , 16.893, 16.393,  5.4  ,  6.224]), 'collision': True, 'all_at_goal': False, 'any_stuck': False, 'outcome': 'collision', 'policies': ['GA3C_CADRL', 'GA3C_CADRL', 'GA3C_CADRL', 'GA3C_CADRL', 'GA3C_CADRL', 'GA3C_CADRL', 'GA3C_CADRL', 'GA3C_CADRL', 'GA3C_CADRL', 'GA3C_CADRL', 'GA3C_CADRL', 'GA3C_CADRL', 'RVO', 'RVO', 'RVO', 'RVO', 'RVO', 'RVO', 'RVO', 'RVO', 'RVO', 'RVO', 'RVO', 'RVO']}
        '''
        total_reward               =  episode_stats['total_reward']
        steps                      =  episode_stats['steps']
        num_agents                 =  episode_stats['num_agents']
        times_to_goal              =  episode_stats['time_to_goal']
        extra_time_to_goal         =  episode_stats['extra_time_to_goal']
        collision                  =  episode_stats['collision']
        all_at_goal                =  episode_stats['all_at_goal']
        any_stuck                  =  episode_stats['any_stuck']
        outcome                    =  episode_stats['outcome']
        policies                   =  episode_stats['policies']

        print("Total steps ",steps)
                                                    
        max_ts = [t / dt for t in times_to_goal]
        trajs = add_traj(prev_agents, trajs, dt, test_case_index, max_ts)


        output_list = []
        
        for agent_index in range(len(prev_agents)): #iterate the agents

            for t in range(len(prev_agents[agent_index].global_state_history)):  #print all the records of a agent

                if t>(steps-1): break

                if agent_index in presence_dict[t]:
                    #output_list.append([ t*10, agent_index, prev_agents[agent_index].global_state_history[t, 1], prev_agents[agent_index].global_state_history[t, 2] ])
                    output_list.append([int(t*10), float(agent_index), round(float(prev_agents[agent_index].global_state_history[t, 1]),2) , round(float(prev_agents[agent_index].global_state_history[t, 2]),2) ] )

        #sort by timestamp
        output_list = np.array(output_list)
        output_list = output_list[np.lexsort((output_list[:, 1], output_list[:, 0]))] #sort by [secondary key, primary key]

        output_list = output_list[~(output_list[:,2:4]==0).all(1)]      #filter out 0 0 useless result from simulation output
        output_list = output_list[~(output_list[:,2:4]>1000).any(1)]    #filter any ridiculous large number
        output_list = output_list[~(output_list[:,2:4]<-1000).any(1)]    #filter any ridiculous small number
        output_list = output_list[~(output_list[:,2:4]==np.nan).any(1)] #filter any ridiculous number
        output_list = output_list[~np.isnan(output_list[:,2:4]).any(1)] #filter any ridiculous number

        #np.savetxt(file_dir+'/trajs/np_test_case_'+str(test_case)+'.txt', output_list, fmt="%d\t%.1f\t%.2f\t%.2f")



        ##########################################
        traj_list = []
        
        for agent_index in range(len(prev_agents)): #iterate the agents

            agent_traj = []

            for t in range(len(prev_agents[agent_index].global_state_history)):  #print all the records of a agent

                if t>(steps-1): break

                #if    prev_agents[agent_index].is_at_goal   or prev_agents[agent_index].was_at_goal_already:
                #    if t>( prev_agents[agent_index].arrival_timestep - 1 ) :break

                #elif  prev_agents[agent_index].in_collision or prev_agents[agent_index].was_in_collision_already:
                #    if t>( prev_agents[agent_index].collision_timestep[0] - 1 ) :break

                agent_traj.append( [round(float(prev_agents[agent_index].global_state_history[t, 1]),2) , round(float(prev_agents[agent_index].global_state_history[t, 2]),2)] )

            traj_list.append(agent_traj)

        ##########################################
        presence_list = []

        #agents leave the scene when they arrive at the goal or run out of time.
        for agent in range(len(prev_agents)): #iterate the agents
            if prev_agents[agent].is_at_goal or prev_agents[agent].was_at_goal_already:
                presence_list.append([prev_agents[agent].start_step_num, prev_agents[agent].arrival_timestep])
            elif prev_agents[agent].ran_out_of_time:
                presence_list.append([prev_agents[agent].start_step_num, prev_agents[agent].timeout_timestep])
            elif prev_agents[agent].is_out_of_bounds:
                #did not arrive and did not timeout, added in at the last timestep
                presence_list.append([prev_agents[agent].start_step_num, prev_agents[agent].out_of_bounds_timestep])
            else:
                presence_list.append([prev_agents[agent].start_step_num, prev_agents[agent].step_num])





            # if    prev_agents[agent].is_at_goal   or prev_agents[agent].was_at_goal_already:
            #     #if arrived but later involved in collision, then collision will override arrival
            #     if (prev_agents[agent].in_collision or prev_agents[agent].was_in_collision_already):
            #         presence_list.append( [0 , prev_agents[agent].collision_timestep[0]+1 ] )
            #     else:
            #         presence_list.append( [0 , prev_agents[agent].arrival_timestep+1 ] )
            # elif  prev_agents[agent].in_collision or prev_agents[agent].was_in_collision_already:
            #     presence_list.append( [0 , prev_agents[agent].collision_timestep[0]+1 ] )
            # elif  prev_agents[agent].ran_out_of_time:
            #     presence_list.append( [0 , -1 ] )
           
        if test_case_index == 5:
            print(":h")

        ##############################################
        overall_record['test_name'] = folder_policy_name
        overall_record['iteration_index'] = test_case_index 

        overall_record['history_len'] = 0
        overall_record['timeout'] = timeout
        overall_record['timestep_size'] = 0.1
        overall_record['field_bound'] = np.array( [[0,5],[0,5]] )
        overall_record['pop_density'] = population_density # people/m^2
        overall_record['target_speed'] = np.array( [ agent.pref_speed for agent in prev_agents ] )
        overall_record['agent_radius'] = np.array( [ agent.radius     for agent in prev_agents ] )
        overall_record['collision_cooldown'] = np.inf
        overall_record['policy'] = np.array( [ eval(p) for p in policies ] )

        overall_record['history_traj'] = []
        overall_record['goal']    =  np.array( [ agent.goal_global_frame for agent in prev_agents ] )
        overall_record['presence'] = np.array( presence_list ) 
        #overall_record['presence'] = np.array( [ [[0,overall_record['timeout']/overall_record['timestep_size']]] * len(prev_agents) ] ) 
        overall_record['arrival'] = np.array( [ agent.arrival_timestep for agent in prev_agents ] )
        overall_record['collision'] = np.array( [ agent.collision_timestep for agent in prev_agents ] )
        overall_record['trajectory']  = np.array( traj_list )
        
        #check data invariants
        assert(len(presence_list) == len(traj_list))
        assert(overall_record['collision'].shape[0] == num_agents)
        assert(overall_record['trajectory'].shape[0] == num_agents)
        assert(overall_record['policy'].size == num_agents)
        assert(overall_record['goal'].shape[0] == num_agents)



        
        ####### TXT output ######
        txt_output_list = list(output_list)
        with open(file_dir + '/test/test_case_'+str(test_case_index)+'.txt', "w") as f:
            for item in txt_output_list:
                timestamp, agent_id, record_x, record_y = item
                #if timestamp >= ((steps-1)*10): break
                f.write(str(int(timestamp))+"\t"+str(float(agent_id))+"\t"+str(record_x)+"\t"+str(record_y)+"\n")
        f.close()

        ########collision / timeout log output for all agents ##########
        log_dir = file_dir + '/logs/'
        os.makedirs(log_dir, exist_ok=True)
        with open(file_dir + '/logs/test_case_'+str(test_case_index)+'_log.txt', "w") as f:

            f.write(str("number of agents: "+str(num_agents)+"\n"))
            f.write(str("steps: "+str(steps)+"\n"))

            collision_count = arrival_count = timeout_count = 0
            for agent in range(num_agents):
                if    prev_agents[agent].is_at_goal   or prev_agents[agent].was_at_goal_already:
                    arrival_count+=1
                elif  prev_agents[agent].in_collision or prev_agents[agent].was_in_collision_already:
                    collision_count+=1
                elif  prev_agents[agent].ran_out_of_time:
                    timeout_count+=1

            
            f.write(str("*"*15))
            f.write(str("Arrival=\n"))
            f.write(str(arrival_count)+"\n")
            f.write(str("Collision=\n"))
            f.write(str(collision_count)+"\n")
            f.write(str("Timeout=\n"))
            f.write(str(timeout_count)+"\n")
            f.write(str("*"*15))
            f.write(str("STATUS during the end\n"))
            for agent in range(num_agents):
                if    prev_agents[agent].is_at_goal   or prev_agents[agent].was_at_goal_already:
                    f.write("Agent "+str(agent)+ " ARRIVED\n")
                elif  prev_agents[agent].in_collision or prev_agents[agent].was_in_collision_already:
                    f.write("Agent "+str(agent)+ " COLLISION\n")
                elif  prev_agents[agent].ran_out_of_time:
                    f.write("Agent "+str(agent)+ " TIMEOUT\n")
                            
            f.write(str("times_to_goal\n"))
            for agent in range(num_agents):
                f.write("Agent "+str(agent)+ " "+str(times_to_goal[agent])+"\n")

            f.write(str("extra_time_to_goal\n"))
            for agent in range(num_agents):
                f.write("Agent "+str(agent)+ " "+str(extra_time_to_goal[agent])+"\n")

            f.write(str("Policies\n"))
            for agent in range(num_agents):
                f.write("Agent "+str(agent)+ " running "+str(policies[agent])+"\n")

            ########output numpy for easier future auto plot generation##################
            np.savez(file_dir + '/logs/test_case_'+str(test_case_index)+".npz", num_agents=num_agents, steps=steps, arrival_count=arrival_count, collision_count=collision_count,
                     timeout_count=timeout_count, times_to_goal=times_to_goal, extra_time_to_goal=extra_time_to_goal, policies=policies)
                     
        f.close()

        overall_record_list.append( overall_record )
    
    np.savez(    file_dir + '/logs/overall_result.npz', data=overall_record_list )        

    
    #fname = pkl_dir+one_env.plot_policy_name+'.pkl'
    #pickle.dump(trajs, open(fname,'wb'))
    #print('dumped {}'.format(fname))
    return True

if __name__ == '__main__':
    main()
    print("Experiment over.")
