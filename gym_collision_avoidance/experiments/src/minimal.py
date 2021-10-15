import os
import numpy as np
import gym
gym.logger.set_level(40)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_name"          , type=str, required=False, help="the output folder name for this run of experiment... e.g. exp1_ETH_CADRL or exp2_0.3_CADRL")
parser.add_argument("--experiment_num"       , type=int, required=True, help="which experiment are we running (1,2,???)")
parser.add_argument("--algorithm_name"       , type=str, required=False, help="which algorithm are we using? CADRL,RVO,SOCIALFORCE,  SPEC, STGCNN, SLSTM, SOCIALGAN")

parser.add_argument("--experiment_iteration" , type=int, required=False, help="for each experiment, how many iteration (how many scenario, from start to goal) should it generate?")
parser.add_argument("--timeout"              , type=int, required=True, help="how many seconds for the experiment to terminate, and declare on-going agents timeout?")

parser.add_argument("--population_density"   , type=float, required=False, default="-1.0" ,help="under exp2, what population density should be used?")
parser.add_argument("--dataset_name"         , type=str  , required=False, default="None" ,help="under exp1, for the exp settings of algortihms, which dataset should they mimick?")
parser.add_argument("--map"                  , type=str  , required=True, help="which static map file to use")
args = parser.parse_args()
print(args)

experiment_number = args.experiment_num
algorithm_name    = args.algorithm_name
map_name = '.'.join(os.path.basename(os.path.normpath(args.map)).split('.')[:-1])
print(f'Map: {map_name}')

experiment_iteration_num = args.experiment_iteration
timeout = args.timeout

dataset_name = args.dataset_name
population_density = args.population_density

os.environ["global_timeout"]             = str(timeout)
os.environ["global_experiment_number"]   = str(experiment_number)
os.environ["global_dataset_name"]        = str(dataset_name)
os.environ["global_population_density"]  = str(population_density)

os.environ['GYM_CONFIG_CLASS'] = 'MinimalMap'
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs import test_cases as tc


def main():
    '''
    Minimum working example, trying to use static map:
    2 agents: Both running GA3C-CADRL
    '''

    # Create single tf session for all experiments
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.Session().__enter__()

    # Instantiate the environment
    env = gym.make("CollisionAvoidance-v0")
    env.set_static_map(args.map)

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__)) + f'/../../experiments/results/minimal/{map_name}/')

    # Set agent configuration (start/goal pos, radius, size, policy)
    policies=['GA3C_CADRL', 'SimpleMap']
    sensors = ['other_agents_states', 'occupancy_grid']
    policy_classes = [tc.policy_dict[policy] for policy in policies]
    agents = tc.get_testcase_two_agents(policies, sensors=sensors)
    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    env.set_agents(agents)

    obs = env.reset() # Get agents' initial observations

    # Repeatedly send actions to the environment based on agents' observations
    num_steps = 500
    for i in range(num_steps):

        actions = {}

        # Run a simulation step (check for collisions, move sim agents)
        obs, rewards, game_over, which_agents_done = env.step(actions)
        agents = env.agents
        for i, agent in enumerate(agents):
            if not agent.is_done and agent.in_collision:
                print(f'\tAgent {i} in collision')

        if game_over:
            print("All agents finished!")
            break
    env.reset()

    return True

if __name__ == '__main__':
    main()
    print("Experiment over.")
