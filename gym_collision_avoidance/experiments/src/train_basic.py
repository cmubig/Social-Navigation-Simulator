from argparse import ArgumentError
import os
import numpy as np
import gym
# import cv2
import matplotlib.pyplot as plt
gym.logger.set_level(40)
os.environ['GYM_CONFIG_CLASS'] = 'Train'
os.environ["global_timeout"]             = "1000"
os.environ["global_experiment_number"]   = "1"
os.environ["global_dataset_name"]        = "custom"
os.environ["global_population_density"]  = "0.7"
# os.environ['GYM_CONFIG_CLASS'] = 'Example'

# os.environ['GYM_CONFIG_PATH'] = '/home/mjana/ws/src/Social-Navigation-Simulator/gym_collision_avoidance/experiments/src/train_basic.py'

from gym_collision_avoidance.envs import test_cases as tc
from gym_collision_avoidance.envs import Config
# from gym_collision_avoidance.envs.config import Config as EnvConfig
from gym_collision_avoidance.experiments.src.env_utils import create_env

def distance_penalty(start,goal,state):
    return np.abs((goal[0]-start[0])*(start[1]-state[1])-(goal[1]-start[1])*(start[0]-state[0]))/np.sqrt((goal[0]-start[0])**2+(goal[1]-start[1])**2)

def main():
    '''
    Minimum working example:
    2 agents: 1 training a DQN, 1 running external policy
    '''

    env, one_env = create_env()
    # In case you want to save plots, choose the directory
    one_env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__)) + '/../../experiments/results/train/')

    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = tc.get_testcase_one_train()

    init_state = []

    # print(agents[0].to_vector())

    for i in range(len(agents)):
        state = []
        state.append(agents[i].to_vector()[0][1])
        state.append(agents[i].to_vector()[0][2])
        state.append(agents[i].to_vector()[0][10])
        init_state.append(state)
    
    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    one_env.set_agents(agents)
    
    # Training environment parameters
    num_episodes = 500
    num_steps = 150
    eps_start = 0.95                # exploration probability at start
    eps_end = 0.1                   # exploration probability at end
    eps_dec = 0.993                 # exploration probability decay factor
    eps = eps_start                 # exploration probability
    init_pos_min_x = -1.5             # minimum initial position
    init_pos_min_y = -1.5
    init_pos_max_x = 1.5              # maximum initial position
    init_pos_max_y = 1.5
    distance_reward_factor = -0.005     # reward factor for deviating from straight line path

    # Repeatedly send actions to the environment based on agents' observations
    set_point = -0.74*np.pi
    P = 1.0
    int_err = 0.0
    err = 0.0

    # Learning agent parameters
    num_actions = 6
    num_state_vector = 13
    
    agents[0].policy.init_network(num_actions, num_state_vector)

    next_state = np.zeros(num_state_vector, dtype=np.float32)
    curr_state = np.zeros(num_state_vector, dtype=np.float32)

    goal_count = 0

    gx = 1
    # training metrics
    scores = []
    success_rate_list = []
    time_to_goal_list = []
    end = False
    for k in range(num_episodes):
        obs = one_env.reset() # Get agents' initial observations
        for j in range(len(agents)-1):
            agents[j].reset(px=np.random.uniform(init_pos_min_x, init_pos_max_x),
                            py=np.random.uniform(init_pos_min_y, init_pos_max_y),
                            # heading=np.random.uniform(0, 2*np.pi),
                            gx = np.random.uniform(init_pos_min_x, init_pos_max_x), 
                            gy = np.random.uniform(init_pos_min_y, init_pos_max_y))
            agents[j].is_done = False
            agents[j].is_out_of_bounds = False
            agents[j].ran_out_of_time = False

        cumul_reward = 0
        print("agent, x,y,theta, theta_global: ", agents[0].pos_global_frame, agents[0].heading_ego_frame, agents[0].heading_global_frame)

        print("goal: ", agents[0].goal_global_frame)
        time_to_goal = num_steps
        start = agents[0].pos_global_frame
        goal = agents[0].goal_global_frame
        for i in range(num_steps):

            # Query the external agents' policies
            # e.g., actions[0] = external_policy(dict_obs[0])
            actions = {}
            print(":::::::::::::::::")
            print("\nepisode number", k+1, "agents: ", len(agents))

            p_err = err
            curr = agents[1].heading_global_frame
            err = set_point - agents[1].heading_global_frame
            control = np.clip(P*err, -1, 1)/2.0 + 0.5
            # control = -np.pi/6
            actions[1] = np.array([0.0, control])
            
            curr_state = next_state.copy()

            # TODO: get the RL action from the policy [argmax_a Q(s,a)]
            rl_action = agents[0].policy.get_action(curr_state, eps)
            actions[0] = rl_action
            # actions[0] = np.random.randint(0, 11)
            
            # Run a simulation step (check for collisions, move sim agents)
            obs, rewards, game_over, which_agents_done = one_env.step(actions)
            # print("obs", obs)
            # rewards+= distance_reward_factor*distance_penalty(init_state[0], init_state[1], agents[0].pos_global_frame)
            if agents[0].is_at_goal:
                rewards-= i/num_steps
            cumul_reward += rewards
            print("done 0? ", agents[0].is_at_goal, agents[0].is_done, agents[0].ran_out_of_time, agents[0].is_out_of_bounds)
            print("done 1? ", agents[1].is_at_goal, agents[1].is_done, agents[1].ran_out_of_time, agents[1].is_out_of_bounds)
            print("which agents done: ",which_agents_done)
            next_state[0] = obs[0]['dist_to_goal']
            next_state[1] = agents[0].pos_global_frame[0]
            next_state[2] = agents[0].pos_global_frame[1]
            next_state[3] = agents[0].vel_global_frame[0]
            next_state[4] = agents[0].vel_global_frame[1]
            next_state[5] = obs[0]['heading_ego_frame']
            # next_state[5] = agents[0].heading_global_frame
            next_state[6] = agents[0].goal_global_frame[0]
            next_state[7] = agents[0].goal_global_frame[1]
            for i in range(8,num_state_vector):
                next_state[i] = obs[0]['other_agent_states'][i-6]
            print("action control", agents[0].policy.external_action_to_action(agents[0], actions[0]))
            # print("speed",agents[0].speed_global_frame)
            print("s", curr_state)
            print("a", actions[0])
            print("r", rewards)
            print("s'", next_state)
            print("other agents", obs)
            print("cumul_reward", cumul_reward)
            print("at goal?:::::::", agents[0].is_at_goal)
            print("eps = ", eps)
            print("sucess rate%", 100*goal_count/(k+1))
            
            # TODO: use obs and reward to train learning agent (add to experience replay and learn for 1 step)
            if i > 1:
                agents[0].policy.learn_step(curr_state, actions[0], rewards, next_state)

            if(agents[0].is_at_goal):
                goal_count+=1
                print("Agent has reached goal")
                time_to_goal = i
                break

            if agents[0].is_done:
                print("agent stopped working")
                end = True
                # break
            # if game_over:
                # print("All agents finished!")
                # break
        # if end:
            # break
        success_rate_list.append(100*goal_count/(k+1))
        time_to_goal_list.append(time_to_goal)
        if (k+1)%80 == 0:
            eps = max(eps_end, eps - 0.15)
        scores.append(cumul_reward)
    env.reset()

    agents[0].policy.save_checkpoint("sns_dqn_1ag_free")

    plt.figure(figsize=(12,8))
    plt.plot(range(num_episodes), scores)
    plt.xlim(-1, num_episodes+1)
    plt.ylim(-2,3)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward vs Episode')

    plt.figure(figsize=(12,8))
    plt.plot(range(num_episodes), success_rate_list)
    plt.xlim(-1, num_episodes+1)
    plt.ylim(0,100)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs Episode')

    plt.figure(figsize=(12,8))
    plt.plot(range(num_episodes), time_to_goal_list)
    plt.xlim(-1, num_episodes+1)
    plt.ylim(0,255)
    plt.xlabel('Episode')
    plt.ylabel('Time to goal')
    plt.title('Time to goal vs Episode')


    plt.show()

    return True

if __name__ == '__main__':
    main()
    print("Experiment over.")