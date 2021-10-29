import numpy as np
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy

# Filter list by Boolean list 
# Using itertools.compress 
from itertools import compress

from gym_collision_avoidance.envs import Config

class SimpleStaticMapPolicy(InternalPolicy):
    """ Simple Static Map agents simply drive at pref speed toward goal,
        with global planning around static obstacles (from initial observation) """
    def __init__(self):
        InternalPolicy.__init__(self, str="SimpleMap")
        self.x_width = Config.STATIC_MAP_SIZE
        self.y_width = Config.STATIC_MAP_SIZE
        self.grid_cell_size = Config.STATIC_MAP_GRID_CELL_SIZE
        self.origin_coords = np.array([(self.x_width/2.)/self.grid_cell_size, (self.y_width/2.)/self.grid_cell_size])
        self.map = None


    def find_next_action(self, obs, agents, agent_index, full_agent_list = None, active_agent_mask = None):
        """ Go at pref_speed, apply a change in heading equal to zero out current ego heading (heading to goal)

        Args:
            obs (dict): ignored
            agents (list): of Agent objects
            i (int): this agent's index in that list

        Returns:
            np array of shape (2,)... [spd, delta_heading]

        """
        
        self.map = obs['static_map']
        agent = agents[agent_index]
        grid_idx, in_map = self._coord_to_idx(agent.pos_global_frame)
        import pdb; pdb.set_trace()
        #check if elements before index contains non active agents, if yes, remove them, thus calculate the index shift
        before_index = np.array(active_agent_mask)[:agent_index]

        #see how many non active agents are before index,  minus them calculate index shift
        agent_index = agent_index - len( before_index[ before_index==False ] )

        agents = list(compress(full_agent_list, active_agent_mask))

        action = np.array([agents[agent_index].pref_speed, -agents[agent_index].heading_ego_frame])
        return action

    def _coord_to_idx(self, pos):
        # for a single [px, py] -> [gx, gy]
        gx = int(np.floor(self.origin_coords[0]-pos[1]/self.grid_cell_size))
        gy = int(np.floor(self.origin_coords[1]+pos[0]/self.grid_cell_size))
        grid_coords = np.array([gx, gy])
        in_map = gx >= 0 and gy >= 0 and gx < self.map.shape[0] and gy < self.map.shape[1]
        return grid_coords, in_map