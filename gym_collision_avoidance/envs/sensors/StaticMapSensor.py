import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor

import matplotlib.pyplot as plt

from gym_collision_avoidance.envs import Config

class StaticMapSensor(Sensor):
    """ StaticMap is analogous to OccupancyGridSensor but without being ego-centric

    Currently the grid parameters are mostly hard-coded...

    :param x_width: (float or int) meters of x dimension in returned gridmap (-x_width/2, +x_width/2) from agent's center
    :param x_width: (float or int) meters of y dimension in returned gridmap (-y_width/2, +y_width/2) from agent's center

    """
    def __init__(self):
        if not Config.USE_STATIC_MAP:
            print("StaticMapSensor won't work without static map enabled (Config.USE_STATIC_MAP)")
            assert(0)
        Sensor.__init__(self)
        self.name = 'occupancy_grid'
        # TODO: get from Config
        self.x_width = Config.STATIC_MAP_SIZE
        self.y_width = Config.STATIC_MAP_SIZE

    def sense(self, agents, agent_index, top_down_map):
        """ Use the full top_down_map to compute a smaller occupancy grid centered around agents[agent_index]'s center.

        Args:
            agents (list): all :class:`~gym_collision_avoidance.envs.agent.Agent` in the environment
            agent_index (int): index of this agent (the one with this sensor) in :code:`agents`
            top_down_map (2D np array): binary image with 0 if that pixel is free space, 1 if occupied

        Returns:
            top_down_map (np array): copy of top_down_map.map passed in

        """

        return top_down_map.map.copy()

    def resize(self, og_map):
        """ Currently just copies the gridmap... not sure why this exists.
        """
        resized_og_map = og_map.copy()
        return resized_og_map
