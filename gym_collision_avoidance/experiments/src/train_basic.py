from argparse import ArgumentError
import os
import numpy as np
import gym
import matplotlib.pyplot as plt
gym.logger.set_level(40)
os.environ['GYM_CONFIG_CLASS'] = 'Train'
os.environ["global_timeout"]             = "200"
os.environ["global_experiment_number"]   = "1"
os.environ["global_dataset_name"]        = "custom"
os.environ["global_population_density"]  = "0.7"
# os.environ['GYM_CONFIG_CLASS'] = 'Example'

# os.environ['GYM_CONFIG_PATH'] = '/home/mjana/ws/src/Social-Navigation-Simulator/gym_collision_avoidance/experiments/src/train_basic.py'

from gym_collision_avoidance.envs import test_cases as tc
from gym_collision_avoidance.envs import Config
# from gym_collision_avoidance.envs.config import Config as EnvConfig
from gym_collision_avoidance.experiments.src.env_utils import create_env

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
    num_episodes = 200
    num_steps = 180

    # Repeatedly send actions to the environment based on agents' observations
    set_point = -0.74*np.pi
    P = 1.0
    int_err = 0.0
    err = 0.0

    # Learning agent parameters
    num_actions = 11
    num_state_vector = 13
    eps_start = 0.9                 # exploration probability at start
    eps_end = 0.01                  # exploration probability at end
    eps_dec = 0.99                  # exploration probability decay factor
    eps = eps_start                 # exploration probability
    agents[0].policy.init_network(num_actions, num_state_vector)

    next_state = np.zeros(13, dtype=np.float32)
    curr_state = np.zeros(13, dtype=np.float32)

    # training metrics
    scores = []

    for k in range(num_episodes):
        for j in range(len(agents)):
            print("init state: ", init_state[j])
            agents[j].reset(px=init_state[j][0], py=init_state[j][1], heading=init_state[j][2])
        obs = one_env.reset() # Get agents' initial observations
        cumul_reward = 0
        print("episode number", one_env.episode_number)
        for i in range(num_steps):

            # Query the external agents' policies
            # e.g., actions[0] = external_policy(dict_obs[0])
            actions = {}

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
            rewards-= 0.0025
            cumul_reward += rewards

            next_state[0] = agents[0].pos_global_frame[0]
            next_state[1] = agents[0].pos_global_frame[1]
            next_state[2] = agents[0].vel_global_frame[0]
            next_state[3] = agents[0].vel_global_frame[1]
            next_state[4] = agents[0].heading_global_frame
            next_state[5] = obs[0]['dist_to_goal']
            for i in range(6,13):
                next_state[i] = obs[0]['other_agent_states'][i-6]
        
            print("s", curr_state)
            print("a", actions[0])
            print("r", rewards)
            print("s'", next_state)
            print("cumul_reward", cumul_reward)
            print("at goal?:::::::", agents[0].is_at_goal)
            # TODO: use obs and reward to train learning agent (add to experience replay and learn for 1 step)
            if i > 1:
                agents[0].policy.learn_step(curr_state, actions[0], rewards, next_state)

            if game_over:
                print("All agents finished!")
                break
        eps = max(eps_end, eps * eps_dec)
        scores.append(cumul_reward)
    env.reset()
    plt.plot(range(num_episodes), scores)
    plt.show()
    return True

if __name__ == '__main__':
    main()
    print("Experiment over.")