import os
import numpy as np
os.environ['GYM_CONFIG_CLASS'] = 'Sam_multi_algo'
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

def reset_env(env, one_env, test_case_fn, test_case_args, test_case, num_agents, agent_policies_list, prev_agents=None, start_from_last_configuration=True):
    if prev_agents is None:
##        print(policies)
##        print(agent_policies_list)
##        print("*"*15)
##        print([ policies[i]['policy'] for i in agent_policies_list])
##        print([ policies[i]['sensors'][0] for i in agent_policies_list])

        policy_name_list = [ policies[i]['policy'] for i in agent_policies_list]
        sensor_name_list = [ policies[i]['sensors'][0] for i in agent_policies_list]
        
        prev_agents = tc.small_test_suite(num_agents=num_agents, test_case_index=0, policies=policy_name_list, agents_sensors=sensor_name_list)
        #print("="*15)
        #print(vars(prev_agents[0]))

##
##        for agent in prev_agents:
##            if 'checkpt_name' in policies[policy]:
##                agent.policy.env = env
##                agent.policy.initialize_network(**policies[policy])
##            if 'sensor_args' in policies[policy]:
##                for sensor in agent.sensors:
##                    sensor.set_args(policies[policy]['sensor_args'])

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
    test_case_args['agents'] = prev_agents
    test_case_args['letter'] = Config.LETTERS[test_case % len(Config.LETTERS)]
    
    one_env.plot_policy_name = "mixed-RVO-CADRL" #perhaps it is just the name?
    agents = test_case_fn(**test_case_args)
    one_env.set_agents(agents)
    init_obs = env.reset()
    one_env.test_case_index = test_case
    return init_obs

def main():
    np.random.seed(0)

    test_case_fn = tc.sam_formation #formation
    test_case_args = {}

    #Formation
    #env test_cases formation
    #For sceneraio generator, can be programmed
    #formation is for 6 agents
    

    env, one_env = create_env()

    one_env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__)) + '/../results/sam/')

#######Create simulation for each policy,  not setting unique policy for each agents
##
##    for num_agents in Config.NUM_AGENTS_TO_TEST:
##        for policy in Config.POLICIES_TO_TEST:
##            np.random.seed(0)
##            prev_agents = None
##            for test_case in range(Config.NUM_TEST_CASES):
##                _ = reset_env(env, one_env, test_case_fn, test_case_args, test_case, num_agents, policies, policy, prev_agents)
##                episode_stats, prev_agents = run_episode(env, one_env)

    for num_agents in Config.NUM_AGENTS_TO_TEST: #if two num agents to test, simulate two times
        
        agent_policies_list = Config.POLICIES_TO_TEST
        
        np.random.seed(0)
        prev_agents = None
        for test_case in range(Config.NUM_TEST_CASES):
            _ = reset_env(env, one_env, test_case_fn, test_case_args, test_case, num_agents, agent_policies_list,  prev_agents)
            episode_stats, prev_agents = run_episode(env, one_env)

    return True

if __name__ == '__main__':
    main()
    print("Experiment over.")
