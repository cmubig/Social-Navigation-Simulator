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

os.environ['GYM_CONFIG_CLASS'] = 'Example'
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs import test_cases as tc


def main():
    '''
    Minimum working example:
    2 agents: 1 running external policy, 1 running GA3C-CADRL
    '''

    # Create single tf session for all experiments
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.Session().__enter__()

    # Instantiate the environment
    env = gym.make("CollisionAvoidance-v0")

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__)) + '/../../experiments/results/example/')

    # Set agent configuration (start/goal pos, radius, size, policy)
    policies=['learning', 'GA3C_CADRL']
    policy_classes = [tc.policy_dict[policy] for policy in policies]
    agents = tc.get_testcase_two_agents(policies)
    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    env.set_agents(agents)

    obs = env.reset() # Get agents' initial observations

    # Repeatedly send actions to the environment based on agents' observations
    num_steps = 500
    set_point = np.pi/6.0
    P = 1.0
    int_err = 0.0
    err = 0.0
    learning_idx = 0
    for i in range(num_steps):

        # Query the external agents' policies
        # e.g., actions[0] = external_policy(dict_obs[0])
        actions = {}

        # basic test, have 1st agent go on set heading
        alive_agents = [agent for agent in agents if not agent.is_done]
        learning_agents = [agent for agent in alive_agents if type(agent.policy) == policy_classes[0]]
        # Upate the Learning Agent if it is still alive
        if len(learning_agents):
            learning_agent = learning_agents[0]
            new_learning_idx = agents.index(learning_agent)
            if new_learning_idx != learning_idx:
                learning_idx = new_learning_idx
                P, int_err, err = 1.0, 0.0, 0.0

            p_err = err
            curr = agents[learning_idx].heading_global_frame
            err = set_point - agents[learning_idx].heading_global_frame
            control = np.clip(P*err, -1, 1)/2.0 + 0.5
            actions[learning_idx] = np.array([1.0, control])

        # Internal agents (running a pre-learned policy defined in envs/policies)
        # will automatically query their policy during env.step
        # ==> no need to supply actions for internal agents here

        # Run a simulation step (check for collisions, move sim agents)
        obs, rewards, game_over, which_agents_done = env.step(actions)
        agents = env.agents

        if game_over:
            print("All agents finished!")
            break
    env.reset()

    return True

if __name__ == '__main__':
    main()
    print("Experiment over.")
