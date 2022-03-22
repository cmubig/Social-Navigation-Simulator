import numpy as np
import torch

from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.DQN import network

class LearningPolicyDQN(LearningPolicy):
    """ The DQN policy while it's still being trained (an external process provides a discrete action input)
    """
    def __init__(self):
        LearningPolicy.__init__(self)
        self.possible_actions = network.Actions()
        self.n_eps    =  500            # no. ofs episodes to train for
        self.eps_start=  0.90           # initial exploration probability
        self.eps_end  =  0.01           # final exploration probability
        self.eps_dec  =  0.991           # decrement of epsilon
        self.external_action = np.zeros(self.possible_actions.num_actions)
        self.step_counter = 0
        self.nA = self.possible_actions.num_actions
        self.episode = 0
        print(self.possible_actions.actions)

    def init_network(self, nA, nS):
        # initialize DQN with nA = number of actions, nS = size of state representation vector
        self.nA = nA
        self.nS = nS
        self.agents = [network.Agent(nA,nS)]
    
    def get_action(self, state, eps):
        return self.agents[0].eps_greedy(state, self.nA , eps)

    def save_checkpoint(self, name):
        torch.save(self.agents[0].Q_policy.state_dict(), name+".pth")

    
    def learn_step(self,state,action,reward,next_state):
        self.step_counter+=1
        print("learner step: ", self.step_counter)
        self.agents[0].step(state,action,reward,next_state)
        print("learner step complete")

    def external_action_to_action(self, agent, external_action):
        """ Convert the discrete external_action into an action for this environment using properties about the agent.

        Args:
            agent (:class:`~gym_collision_avoidance.envs.agent.Agent`): the agent who has this policy
            external_action (int): discrete action between 0-11 directly from the network output

        Returns:
            [speed, heading_change] command

        """

        raw_action = self.possible_actions.actions[int(external_action)]
        action = np.array([agent.pref_speed*raw_action[0], raw_action[1]])
        return action