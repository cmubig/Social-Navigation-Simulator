import numpy as np
import torch
import copy
import argparse
import configparser
import os

from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.LearningCADRL.utils.trainer import Trainer
from gym_collision_avoidance.envs.policies.LearningCADRL.utils.memory import ReplayMemory
from gym_collision_avoidance.envs.policies.LearningCADRL.policy.policy_factory import policy_factory

class LearningPolicyCADRL(LearningPolicy):
    """ The DQN policy while it's still being trained (an external process provides a discrete action input)
    """
    def __init__(self):
        LearningPolicy.__init__(self)
        self.n_eps    =  500            # no. ofs episodes to train for
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Training on device: ",self.device)
        # self.device = "cpu"
        # self.external_action = np.zeros(self.possible_actions.num_actions)
        self.step_counter = 0
        self.episode = 0
        self.v_pref = 0
        self.dt = 0
        self.parser = argparse.ArgumentParser('Parse configuration file')
        # self.nA = self.possible_actions.num_actions
        self.gamma = 0.9
        # print(self.possible_actions.actions)
        # self.init_network()
    def update_next_lookahead(self, state):
        self.policy.next_step_lookahead = state, 

    def init_network(self, nA,nS):
        self.parser.add_argument('--env_config', type=str, default=os.path.dirname(os.path.realpath(__file__))+ '/configs/env.config')
        self.parser.add_argument('--policy', type=str, default='cadrl')
        self.parser.add_argument('--policy_config', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/configs/policy.ini')
        self.parser.add_argument('--train_config', type=str, default=os.path.dirname(os.path.realpath(__file__))+'/configs/train.config')
        self.parser.add_argument('--output_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/../../experiments/results/train/')
        self.parser.add_argument('--weights', type=str)
        self.parser.add_argument('--resume', default=False, action='store_true')
        self.parser.add_argument('--gpu', default=False, action='store_true')
        self.parser.add_argument('--debug', default=False, action='store_true')
        args = self.parser.parse_args()
        self.nA = nA
        self.nS = nS
        # configure policy
        self.policy_name = 'cadrl'
        self.policy = policy_factory[self.policy_name]()
        # self.policy = policy_factory['sarl']()
        if not self.policy.trainable:
            self.parser.error('Policy has to be trainable')
        if args.policy_config is None:
            self.parser.error('Policy config has to be specified for a trainable network')
        policy_config = configparser.RawConfigParser()
        print(args.policy_config)
        policy_config.read(args.policy_config)
        print("policy_config", policy_config.sections())    
        self.policy.configure(policy_config)
        self.policy.set_device(self.device)

        # read training parameters
        if args.train_config is None:
            self.parser.error('Train config has to be specified for a trainable network')
        train_config = configparser.RawConfigParser()
        train_config.read(args.train_config)
        self.rl_learning_rate = train_config.getfloat('train', 'rl_learning_rate')
        self.train_batches = train_config.getint('train', 'train_batches')
        self.train_episodes = train_config.getint('train', 'train_episodes')
        self.sample_episodes = train_config.getint('train', 'sample_episodes')
        self.target_update_interval = train_config.getint('train', 'target_update_interval')
        self.evaluation_interval = train_config.getint('train', 'evaluation_interval')
        self.capacity = train_config.getint('train', 'capacity')
        self.epsilon_start = train_config.getfloat('train', 'epsilon_start')
        self.epsilon_end = train_config.getfloat('train', 'epsilon_end')
        self.epsilon_decay = train_config.getfloat('train', 'epsilon_decay')
        self.checkpoint_interval = train_config.getint('train', 'checkpoint_interval')

        # configure trainer and explorer
        self.memory = ReplayMemory(self.capacity)
        self.model = self.policy.get_model()
        self.batch_size = train_config.getint('trainer', 'batch_size')
        self.trainer = Trainer(self.model, self.memory, self.device, self.batch_size)
        # self.robot.set_policy(self.policy)
        # self.robot.print_info()
        # reinforcement learning
        self.trainer.set_learning_rate(self.rl_learning_rate)
        self.target_model = copy.deepcopy(self.model)

    def get_action(self, state, eps):
        
        return self.agents[0].eps_greedy(state, self.nA , eps)

    def save_checkpoint(self, name):
        torch.save(self.model, name+".pth")
    
    def act(self, state):
        action = self.policy.predict(state)
        return action

    def update_memory(self, states, action, rewards):
        # print("states,", states[0][0])
        if self.policy_name == 'sarl':
            self.v_pref = states[0][0][1]
        elif self.policy_name == 'cadrl':
            self.v_pref = states[0][1]
        print("v_pref: ", self.v_pref)
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if i == len(states) - 1:
                # terminal state
                value = reward
            else:
                next_state = states[i + 1]
                gamma_bar = pow(self.gamma, self.dt * self.v_pref)
                value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            self.memory.push((state, value))
        
    def update_target_model(self):
        self.target_model = copy.deepcopy(self.model)

    def external_action_to_action(self, agent, external_action):
        """ Convert the discrete external_action into an action for this environment using properties about the agent.

        Args:
            agent (:class:`~gym_collision_avoidance.envs.agent.Agent`): the agent who has this policy
            external_action (int): discrete action between 0-11 directly from the network output

        Returns:
            [speed, heading_change] command

        """
        # print(external_action.v)
        # raw_action = self.possible_actions.actions[int(external_action)]
        action = np.array([external_action.v ,external_action.r])
        return action
