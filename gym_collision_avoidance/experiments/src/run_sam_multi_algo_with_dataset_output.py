import os
import numpy as np

import pickle
from tqdm import tqdm

os.environ['GYM_CONFIG_CLASS'] = 'Custom' #'Sam_multi_algo' #'Sam_multi_algo_dataset_ouput'#'Sam_multi_algo'



#Select EvaluateConfig class from config.py
#Contain parameter for simulations like
#NUM_AGENTS_TO_TEST

#if [6]     then run simulation 1 time with 6 agents
#if [2,3,4] then run simulation 1 time for [2,3,4] respectively


#self.NUM_TEST_CASES
# Test how many times

#for setting up predefined goals 



#Scenarios are controlled by tc.formation
#usually a function that return agents



#CONFIG
#TEST CASE

#The legacy cadrl format is a list of
# [start_x, start_y, goal_x, goal_y, pref_speed, radius] for each agent

from gym_collision_avoidance.experiments.src.master_config import Master_Config
from gym_collision_avoidance.experiments.src.master_config import Scenario_Config
master_config   = Master_Config()
scenario_config = Scenario_Config()


from gym_collision_avoidance.envs import Config
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env, store_stats, policies


######## Create simulation for each policy,  not setting unique policy for each agents
##
##def reset_env(env, one_env, test_case_fn, test_case_args, test_case, num_agents, policies, policy, prev_agents=None, start_from_last_configuration=True):
##    if prev_agents is None:
##        prev_agents = tc.small_test_suite(num_agents=num_agents, test_case_index=0, policies=policies[policy]['policy'], agents_sensors=policies[policy]['sensors'])
##        for agent in prev_agents:
##            if 'checkpt_name' in policies[policy]:
##                agent.policy.env = env
##                agent.policy.initialize_network(**policies[policy])
##            if 'sensor_args' in policies[policy]:
##                for sensor in agent.sensors:
##                    sensor.set_args(policies[policy]['sensor_args'])
'''
{'policy': 'GA3C_CADRL', 'checkpt_dir': 'IROS18', 'checkpt_name': 'network_01900000', 'sensors': ['other_agents_states'], 'sensor_args': {'agent_sorting_method': 'closest_last', 'max_num_other_agents_observed': 19}}
GA3C_CADRL
'''

#  a= [ policies.get(i) for i in POLICIES_TO_TEST]

#a= [ policies[i] for i in POLICIES_TO_TEST]


###Policies list transformed into official name
#policies   
#policies = [ policies[i]['sensors'] for i in POLICIES_TO_TEST]



###########################Dataset output##################################
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

#     global_state = np.array([self.t,
#                                  self.pos_global_frame[0],
#                                  self.pos_global_frame[1],
#                                  self.goal_global_frame[0],
#                                  self.goal_global_frame[1],
#                                  self.radius,
#                                  self.pref_speed,
#                                  self.vel_global_frame[0],
#                                  self.vel_global_frame[1],
#                                  self.speed_global_frame,
#                                  self.heading_global_frame])

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
    

    test_case_fn = tc.custom_formation #formation
    test_case_args = {}
    

    #Formation
    #env test_cases formation
    #For sceneraio generator, can be programmed
    #formation is for 6 agents

    folder_policy_name = np.unique( np.array(list(scenario_config.scenario)[0])[:,1] )[0]
    

    env, one_env = create_env()
    dt = one_env.dt_nominal #retrieve framerate
    #file_dir = file_dir_template = os.path.dirname(os.path.realpath(__file__)) + '/../results/sam_traj_output_ex/'
    file_dir = file_dir_template = os.path.dirname(os.path.realpath(__file__)) + '/../results/'+str(folder_policy_name)+'/'

    #one_env.set_plot_save_dir(  os.path.dirname(os.path.realpath(__file__)) + '/../results/sam_traj_output_ex/')
    one_env.set_plot_save_dir(  os.path.dirname(os.path.realpath(__file__)) + '/../results/'+str(folder_policy_name)+'/')

#######Create simulation for each policy,  not setting unique policy for each agents
##
##    for num_agents in Config.NUM_AGENTS_TO_TEST:
##        for policy in Config.POLICIES_TO_TEST:
##            np.random.seed(0)
##            prev_agents = None
##            for test_case in range(Config.NUM_TEST_CASES):
##                _ = reset_env(env, one_env, test_case_fn, test_case_args, test_case, num_agents, policies, policy, prev_agents)
##                episode_stats, prev_agents = run_episode(env, one_env)


    pkl_dir = file_dir + '/test/'
    os.makedirs(pkl_dir, exist_ok=True)

    num_of_test_case = len(list(scenario_config.scenario))

    trajs = [[] for _ in range(num_of_test_case)]
    
    #polices will be fixed if use this
    #agent_policies_list = master_config.POLICIES_TO_TEST
   
    np.random.seed(0)
    prev_agents = None
    for test_case_index in range(num_of_test_case):
        print("XX1XX")
        print(list(scenario_config.scenario))
        print("XX2XX")
        print(test_case_index)
        print("XXXXXX3")
        print([test_case_index])
        print(list(scenario_config.scenario)[test_case_index])
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
        episode_stats, prev_agents = run_episode(env, one_env)
        print("after episode")
        #print(episode_stats)
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
        f.close() 

    
    #fname = pkl_dir+one_env.plot_policy_name+'.pkl'
    #pickle.dump(trajs, open(fname,'wb'))
    #print('dumped {}'.format(fname))
    return True

if __name__ == '__main__':
    main()
    print("Experiment over.")
