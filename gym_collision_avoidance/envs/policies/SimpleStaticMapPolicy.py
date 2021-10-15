import numpy as np
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy

# Filter list by Boolean list 
# Using itertools.compress 
from itertools import compress


class SimpleStaticMapPolicy(InternalPolicy):
    """ Simple Static Map agents simply drive at pref speed toward goal,
        with global planning around static obstacles (from initial observation) """
    def __init__(self):
        InternalPolicy.__init__(self, str="SimpleMap")

    def find_next_action(self, obs, agents, agent_index, full_agent_list = None, active_agent_mask = None):
        """ Go at pref_speed, apply a change in heading equal to zero out current ego heading (heading to goal)

        Args:
            obs (dict): ignored
            agents (list): of Agent objects
            i (int): this agent's index in that list

        Returns:
            np array of shape (2,)... [spd, delta_heading]

        """
        
        #check if elements before index contains non active agents, if yes, remove them, thus calculate the index shift
        before_index = np.array(active_agent_mask)[:agent_index]

        #see how many non active agents are before index,  minus them calculate index shift
        agent_index = agent_index - len( before_index[ before_index==False ] )

        agents = list(compress(full_agent_list, active_agent_mask))

        action = np.array([agents[agent_index].pref_speed, -agents[agent_index].heading_ego_frame])
        return action
