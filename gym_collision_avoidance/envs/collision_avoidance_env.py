'''
Collision Avoidance Environment
Author: Michael Everett
MIT Aerospace Controls Lab
'''

import gym
import gym.spaces
import numpy as np
import itertools
import copy
import os
import inspect

from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import find_nearest, rgba2rgb, l2norm
from gym_collision_avoidance.envs.visualize import plot_episode, animate_episode
from gym_collision_avoidance.envs.agent import Agent
from gym_collision_avoidance.envs.Map import Map
from gym_collision_avoidance.envs import test_cases as tc

# Filter list by Boolean list 
# Using itertools.compress 
from itertools import compress


# Policies
from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy
# from gym_collision_avoidance.envs.policies.DRLLongPolicy import DRLLongPolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.CARRLPolicy import CARRLPolicy
from gym_collision_avoidance.envs.policies.LearningPolicyGA3C import LearningPolicyGA3C

from gym_collision_avoidance.envs.policies.NAVIGANPolicy import NAVIGANPolicy
from gym_collision_avoidance.envs.policies.STGCNNPolicy import STGCNNPolicy
from gym_collision_avoidance.envs.policies.SPECPolicy import SPECPolicy
from gym_collision_avoidance.envs.policies.SOCIALFORCEPolicy import SOCIALFORCEPolicy
from gym_collision_avoidance.envs.policies.SLSTMPolicy import SLSTMPolicy
from gym_collision_avoidance.envs.policies.SOCIALGANPolicy import SOCIALGANPolicy
# from gym_collision_avoidance.envs.policies.GROUPNAVIGANPolicy import GROUPNAVIGANPolicy
from gym_collision_avoidance.envs.policies.CVMPolicy import CVMPolicy

# Dynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamicsMaxTurnRate import UnicycleDynamicsMaxTurnRate
from gym_collision_avoidance.envs.dynamics.ExternalDynamics import ExternalDynamics

# Sensors
from gym_collision_avoidance.envs.sensors.OccupancyGridSensor import OccupancyGridSensor
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor


#for generate new agents to replace old agents (dynamic number of agents)
from gym_collision_avoidance.experiments.src.master_scenario_generator import Scenario_Generator, Seeded_Scenario_Generator, Seeded_Population_Scenario_Generator, Single_Seeded_Population_Scenario_Generator, real_dataset_traj
policy_dict = {
    'RVO': RVOPolicy,
    'LINEAR': NonCooperativePolicy,
    'carrl': CARRLPolicy,
    'external': ExternalPolicy,
    'GA3C_CADRL': GA3CCADRLPolicy,
    'learning': LearningPolicy,
    'learning_ga3c': LearningPolicyGA3C,
    'static': StaticPolicy,
    'CADRL': CADRLPolicy,
    'NAVIGAN' : NAVIGANPolicy,
    'STGCNN' : STGCNNPolicy,
    'SPEC' : SPECPolicy,
    'SOCIALFORCE' : SOCIALFORCEPolicy,
    'SLSTM' : SLSTMPolicy,
    'SOCIALGAN' : SOCIALGANPolicy,
    # 'GROUPNAVIGAN' : GROUPNAVIGANPolicy,
    'CVM' : CVMPolicy,
}

dynamics_dict = {
    'unicycle': UnicycleDynamics,
    'external': ExternalDynamics,
}



class CollisionAvoidanceEnv(gym.Env):
    """ Gym Environment for multiagent collision avoidance

    The environment contains a list of agents.

    :param agents: (list) A list of :class:`~gym_collision_avoidance.envs.agent.Agent` objects that represent the dynamic objects in the scene.
    :param num_agents: (int) The maximum number of agents in the environment.
    """

    # Attributes:
    #     agents: A list of :class:`~gym_collision_avoidance.envs.agent.Agent` objects that represent the dynamic objects in the scene.
    #     num_agents: The maximum number of agents in the environment.

    metadata = {
        # UNUSED !!
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        self.id = 0

        # Initialize Rewards
        self._initialize_rewards()

        # Simulation Parameters
        self.num_agents = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        self.dt_nominal = Config.DT

        # Collision Parameters
        self.collision_dist = Config.COLLISION_DIST
        self.getting_close_range = Config.GETTING_CLOSE_RANGE

        # Plotting Parameters
        self.evaluate = Config.EVALUATE_MODE

        self.plot_episodes = Config.SHOW_EPISODE_PLOTS or Config.SAVE_EPISODE_PLOTS
        self.plt_limits = Config.PLT_LIMITS
        self.plt_fig_size = Config.PLT_FIG_SIZE
        self.test_case_index = 0

        self.set_testcase(Config.TEST_CASE_FN, Config.TEST_CASE_ARGS)

        self.animation_period_steps = Config.ANIMATION_PERIOD_STEPS

        # if Config.TRAIN_ON_MULTIPLE_AGENTS:
        #     self.low_state = np.zeros((Config.FULL_LABELED_STATE_LENGTH))
        #     self.high_state = np.zeros((Config.FULL_LABELED_STATE_LENGTH))
        # else:
        #     self.low_state = np.zeros((Config.FULL_STATE_LENGTH))
        #     self.high_state = np.zeros((Config.FULL_STATE_LENGTH))

        # Upper/Lower bounds on Actions
        self.max_heading_change = np.pi/3
        self.min_heading_change = -self.max_heading_change
        self.min_speed = 0.0
        self.max_speed = 1.0

        ### The gym.spaces library doesn't support Python2.7 (syntax of Super().__init__())
        self.action_space_type = Config.ACTION_SPACE_TYPE
        
        if self.action_space_type == Config.discrete:
            self.action_space = gym.spaces.Discrete(self.actions.num_actions, dtype=np.float32)
        elif self.action_space_type == Config.continuous:
            self.low_action = np.array([self.min_speed,
                                        self.min_heading_change])
            self.high_action = np.array([self.max_speed,
                                         self.max_heading_change])
            self.action_space = gym.spaces.Box(self.low_action, self.high_action, dtype=np.float32)
        

        # original observation space
        # self.observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
        
        # not used...
        # self.observation_space = np.array([gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
                                           # for _ in range(self.num_agents)])
        # observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
        
        # single agent dict obs
        self.observation = {}
        for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
            self.observation[agent] = {}

        # The observation returned by the environment is a Dict of Boxes, keyed by agent index.
        self.observation_space = gym.spaces.Dict({})
        for state in Config.STATES_IN_OBS:
            self.observation_space.spaces[state] = gym.spaces.Box(Config.STATE_INFO_DICT[state]['bounds'][0]*np.ones((Config.STATE_INFO_DICT[state]['size'])),
                Config.STATE_INFO_DICT[state]['bounds'][1]*np.ones((Config.STATE_INFO_DICT[state]['size'])),
                dtype=Config.STATE_INFO_DICT[state]['dtype'])
            for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
                self.observation[agent][state] = np.zeros((Config.STATE_INFO_DICT[state]['size']), dtype=Config.STATE_INFO_DICT[state]['dtype'])

        self.agents = None
        self.default_agents = None
        self.prev_episode_agents = None

        self.static_map_filename = None
        self.map = None

        self.episode_step_number = None
        self.episode_number = 0

        self.plot_save_dir = None
        self.plot_policy_name = None

        self.perturbed_obs = None

        self.active_agent_mask = None
        self.active_agents = None

        self.active_agents_per_timestep = dict()

        self.replaced_agent_mask = None # agent that arrived will be removed and replace by a new agent
        
    def step(self, actions, dt=None):
        """ Run one timestep of environment dynamics.

        This is the main function. An external process will compute an action for every agent
        then call env.step(actions). The agents take those actions,
        then we check if any agents have earned a reward (collision/goal/...).
        Then agents take an observation of the new world state. We compute whether each agent is done
        (collided/reached goal/ran out of time) and if everyone's done, the episode ends.
        We return the relevant info back to the process that called env.step(actions).

        Args:
            actions (list): list of [delta heading angle, speed] commands (1 per agent in env)
            dt (float): time in seconds to run the simulation (defaults to :code:`self.dt_nominal`)

        Returns:
        4-element tuple containing

        - **next_observations** (*np array*): (obs_length x num_agents) with each agent's observation
        - **rewards** (*list*): 1 scalar reward per agent in self.agents
        - **game_over** (*bool*): true if every agent is done
        - **info_dict** (*dict*): metadata that helps in training

        """
        ###################
        #Use active_agent_list as a mask to filter,  only pass the "active" agents to policy and plot
        if self.episode_step_number == 0:
            self.active_agent_mask = np.array( [True] * len(self.agents) )

            self.replaced_agent_mask = np.array( [False] * len(self.agents) )

        #print("self.episode_step_number")
        print(self.episode_step_number)
            
        self.active_agents = list(compress(self.agents, self.active_agent_mask))
        self.active_agents_per_timestep[self.episode_step_number] = set([agent.id for agent in self.active_agents])
        #print("len(self.active_agents)")
        #print(len(self.active_agents))



        ###################
        if dt is None:
            dt = self.dt_nominal

        self.episode_step_number += 1

        # Take action
        self._take_action(actions, dt)

        self.active_agents = list(compress(self.agents, self.active_agent_mask))

        # Collect rewards
        rewards = self._compute_rewards()

        # Take observation
        next_observations = self._get_obs()

        if (Config.ANIMATE_EPISODES and self.episode_step_number % self.animation_period_steps == 0) or np.any([agent.in_collision for agent in self.active_agents]):
            plot_episode(self.agents, True, self.map, self.test_case_index,
                circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ,
                plot_save_dir=self.plot_save_dir,
                plot_policy_name=self.plot_policy_name,
                save_for_animation=True,
                limits=self.plt_limits,
                fig_size=self.plt_fig_size,
                perturbed_obs=self.perturbed_obs,
                show=False,
                save=True,#)
                active_agent_mask= self.active_agent_mask,
                episode_step_num=self.episode_step_number)

        elif Config.SHOW_EPISODE_PLOTS and self.episode_step_number % self.animation_period_steps == 0:
            plot_episode(self.agents, False, self.map, self.test_case_index,
                circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ,
                plot_save_dir=self.plot_save_dir,
                plot_policy_name=self.plot_policy_name,
                save_for_animation=False,
                limits=self.plt_limits,
                fig_size=self.plt_fig_size,
                perturbed_obs=self.perturbed_obs,
                show=True,
                save=True,#)
                active_agent_mask= self.active_agent_mask,
                episode_step_num=self.episode_step_number)

        # Check which agents' games are finished (at goal/collided/out of time)
        which_agents_done, game_over = self._check_which_agents_done()

        which_agents_done_dict = {}
        which_agents_learning_dict = {}
        for i, agent in enumerate(self.agents):
            which_agents_done_dict[agent.id] = which_agents_done[i]
            which_agents_learning_dict[agent.id] = agent.policy.is_still_learning

        

        # Update active agent mask, only keep agents that is still running, and mask out any (at goal/collided/out of time) agents
        agents_still_running = [not done for done in which_agents_done]

        # agents_inside_field = self._check_which_agents_inside_field()

        
        self.active_agent_mask = self.active_agent_mask & agents_still_running
  
        #add agents here
        #take in-active agents and re-add them as new agents with timestamp of now. 
        for i, agent in enumerate(self.agents):

            #if the agent is not active, and the agent haven't been replaced then read its information and add it again
            if (not self.active_agent_mask[i]) and (not self.replaced_agent_mask[i]) :

                #only if the agent arrived the goal and has enough time for newly added agent to reach goal in a straight line:
                #if agent.is_at_goal and (Config.agent_time_out - self.dt_nominal*self.episode_step_number > agent.straight_line_time_to_reach_goal):

                #agent arrived the goal or agent ran outside the field,    then, if there is enough time for newly added agent to reach goal in a straight line:
                if (agent.is_at_goal or agent.is_out_of_bounds) and (Config.agent_time_out - self.dt_nominal*self.episode_step_number > agent.straight_line_time_to_reach_goal):
                    agent_policy     = "LINEAR" if agent.policy.str == "NonCooperativePolicy" else agent.policy.str
                    
                    agent_radius     = agent.radius
                    agent_pref_speed = agent.pref_speed

                    ######respawn with new position
                    bbox = [[0, 5], [0, 5]] #Config.PLT_LIMITS #e.g. [[-1, 6], [-1, 6]]
                    x_min,x_max = bbox[0]
                    y_min,y_max = bbox[1]

                    # Check which agents' games are finished (at goal/collided/out of time)
                    temp_which_agents_done, temp_game_over = self._check_which_agents_done()
                    # Update active agent mask, only keep agents that is still running, and mask out any (at goal/collided/out of time) agents
                    temp_agents_still_running = [not done for done in temp_which_agents_done]
                    temp_active_agent_mask = self.active_agent_mask & temp_agents_still_running


                    global_timeout            = int(os.environ["global_timeout"])
                    global_experiment_number  = int(os.environ["global_experiment_number"])
                    global_dataset_name       = os.environ["global_dataset_name"]
                    global_population_density = float(os.environ["global_population_density"])

                    respawn_scenario = None
                    print(f'RESPAWNING: {i} -> {len(self.agents)}')
                    #if is experiment 1, then respawn with real start&goal from dataset, will also override
                    if global_experiment_number ==1:
                        respawn_scenario = real_dataset_traj( dataset_name=global_dataset_name ).pick_one( list(compress(self.agents, temp_active_agent_mask)) ,random_seed=self.episode_step_number+i*500)

                    else:
                        respawn_scenario = Single_Seeded_Population_Scenario_Generator( 0 , agent_policy, x_min,x_max, y_min, y_max,
                                                                                        agent_pref_speed, agent_radius, 0, list(compress(self.agents, temp_active_agent_mask)), random_seed=self.episode_step_number+i*500, num_agents_override=1 ).population_random_square_edge() 



                    #print("RESPAWN"*15)
                    #print(respawn_scenario)
                    agent_policy     = "LINEAR" if agent.policy.str == "NonCooperativePolicy" else agent.policy.str

                    _,_, start_x, start_y, goal_x, goal_y, past_traj,_,_ = respawn_scenario
                    agent_start      = np.array( [ start_x, start_y ] )
                    agent_goal       = np.array( [ goal_x, goal_y ] )
                    
                    #agent_goal      = agent.start_global_frame
                    #agent_start     = agent.goal_global_frame

                    agent_vec_to_goal = agent_goal - agent_start
                    agent_heading     = np.arctan2(agent_vec_to_goal[1], agent_vec_to_goal[0])

                    agent_px , agent_py  = agent_start
                    agent_gx , agent_gy  = agent_goal

                    agent_dynamics_model = agent.dynamics_model
                    agent_sensors        = agent.sensors



                    number_of_agents  =  len(self.agents)+1
                    algorithm_name    =  agent_policy

##                    self.scenario=[]
##                    if global_experiment_number == 1: #Simulate algorithm using settings from datasets! (e.g. ETH) 
##                        
##                        for i in range(experiment_iteration_num): #set radius from 0.2 to 0.05 to show slstm do better in low radius situation
##                            self.scenario.append( Seeded_Scenario_Generator( self.exp_setting[0], algorithm_name, self.exp_setting[4],self.exp_setting[5], self.exp_setting[6], self.exp_setting[7] , self.exp_setting[2],
##                                                                             0.2 , 0, num_agents_stddev=self.exp_setting[1], pref_speed_stddev=self.exp_setting[3], random_seed=i , num_agents_override=number_of_agents ).random_square_edge() )
##
##                    elif global_experiment_number in [2,3,4]: #population density evaluation
##
##                        for i in range(experiment_iteration_num):         
##                            self.scenario.append( Seeded_Population_Scenario_Generator( global_population_density, algorithm_name, self.exp_setting[4],self.exp_setting[5], self.exp_setting[6], self.exp_setting[7], self.exp_setting[2],
##                                                                                        0.2, 0, random_seed=i , num_agents_override=number_of_agents ).population_random_square_edge() )

                    new_agent = Agent( agent_px, agent_py, agent_gx, agent_gy, agent_radius, agent_pref_speed, agent_heading, policy_dict[agent_policy], UnicycleDynamics, [OtherAgentsStatesSensor], (number_of_agents-1) )
                    if hasattr(new_agent.policy, 'initialize_network'):
                        new_agent.policy.initialize_network()
                    if not hasattr(new_agent, 'max_heading_change'):
                        new_agent.max_heading_change = self.max_heading_change
                    if not hasattr(new_agent, 'max_speed'):
                        new_agent.max_speed = self.max_speed
                    new_agent.reset( px=agent_px, py=agent_py, gx=agent_gx, gy=agent_gy, pref_speed=agent_pref_speed, radius=agent_radius, heading=agent_heading,
                                     start_step_num= self.episode_step_number ,start_t= self.episode_step_number*self.dt_nominal)

                    if global_experiment_number ==1:  new_agent.past_traj = past_traj
                        
                    self.agents.append( new_agent )

                    # since original agent's already handled by "&" case, no need to update agent's state from active to false,
                    # add the replaced agent as a new agent, add it to active agent mask list
                    self.active_agent_mask = np.append(self.active_agent_mask  , [True] )

                    #since this agent is replaced, update to true
                    self.replaced_agent_mask[i] = True
                     
                    #add the replaced agent as a new agnet, add it to "not yet" replaced agent mask list (since it is new)
                    self.replaced_agent_mask = np.append(self.replaced_agent_mask, [False] )
            

        

        return next_observations, rewards, game_over, \
            {
                'which_agents_done': which_agents_done_dict,
                'which_agents_learning': which_agents_learning_dict,
            }

    '''  ####Original step####
    def step(self, actions, dt=None):
        """ Run one timestep of environment dynamics.

        This is the main function. An external process will compute an action for every agent
        then call env.step(actions). The agents take those actions,
        then we check if any agents have earned a reward (collision/goal/...).
        Then agents take an observation of the new world state. We compute whether each agent is done
        (collided/reached goal/ran out of time) and if everyone's done, the episode ends.
        We return the relevant info back to the process that called env.step(actions).

        Args:
            actions (list): list of [delta heading angle, speed] commands (1 per agent in env)
            dt (float): time in seconds to run the simulation (defaults to :code:`self.dt_nominal`)

        Returns:
        4-element tuple containing

        - **next_observations** (*np array*): (obs_length x num_agents) with each agent's observation
        - **rewards** (*list*): 1 scalar reward per agent in self.agents
        - **game_over** (*bool*): true if every agent is done
        - **info_dict** (*dict*): metadata that helps in training

        """
        
        if dt is None:
            dt = self.dt_nominal

        self.episode_step_number += 1

        # Take action
        self._take_action(actions, dt)

        # Collect rewards
        rewards = self._compute_rewards()

        # Take observation
        next_observations = self._get_obs()

        if Config.ANIMATE_EPISODES and self.episode_step_number % self.animation_period_steps == 0:
            plot_episode(self.agents, False, self.map, self.test_case_index,
                circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ,
                plot_save_dir=self.plot_save_dir,
                plot_policy_name=self.plot_policy_name,
                save_for_animation=True,
                limits=self.plt_limits,
                fig_size=self.plt_fig_size,
                perturbed_obs=self.perturbed_obs,
                show=False,
                save=True)

        # Check which agents' games are finished (at goal/collided/out of time)
        which_agents_done, game_over = self._check_which_agents_done()

        which_agents_done_dict = {}
        which_agents_learning_dict = {}
        for i, agent in enumerate(self.agents):
            which_agents_done_dict[agent.id] = which_agents_done[i]
            which_agents_learning_dict[agent.id] = agent.policy.is_still_learning

        return next_observations, rewards, game_over, \
            {
                'which_agents_done': which_agents_done_dict,
                'which_agents_learning': which_agents_learning_dict,
            }
    '''

    def reset(self):
        """ Resets the environment, re-initializes agents, plots episode (if applicable) and returns an initial observation.

        Returns:
            initial observation (np array): each agent's observation given the initial configuration
        """
        if self.episode_step_number is not None and self.episode_step_number > 0 and self.plot_episodes and self.test_case_index >= 0:
            plot_episode(self.agents, self.evaluate, self.map, self.test_case_index, self.id, circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ, plot_save_dir=self.plot_save_dir, plot_policy_name=self.plot_policy_name, limits=self.plt_limits, fig_size=self.plt_fig_size, show=Config.SHOW_EPISODE_PLOTS, save=Config.SAVE_EPISODE_PLOTS)
            if Config.ANIMATE_EPISODES:
                animate_episode(num_agents=len(self.agents), plot_save_dir=self.plot_save_dir, plot_policy_name=self.plot_policy_name, test_case_index=self.test_case_index, agents=self.agents)
            self.episode_number += 1
        self.begin_episode = True
        self.episode_step_number = 0
        self._init_agents()
        if Config.USE_STATIC_MAP:
            self._init_static_map()
        for state in Config.STATES_IN_OBS:
            for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
                self.observation[agent][state] = np.zeros((Config.STATE_INFO_DICT[state]['size']), dtype=Config.STATE_INFO_DICT[state]['dtype'])
        return self._get_obs()

    def _take_action(self, actions, dt):
        """ Some agents' actions come externally through the actions arg, agents with internal policies query their policy here, 
        then each agent takes a step simultaneously.

        This makes it so an external script that steps through the environment doesn't need to
        be aware of internals of the environment, like ensuring RVO agents compute their RVO actions.
        Instead, all policies that are already trained/frozen are computed internally, and if an
        agent's policy is still being trained, it's convenient to isolate the training code from the environment this way.
        Or, if there's a real robot with its own planner on-board (thus, the agent should have an ExternalPolicy), 
        we don't bother computing its next action here and just take what the actions dict said.

        Args:
            actions (dict): keyed by agent indices, each value has a [delta heading angle, speed] command.
                Agents with an ExternalPolicy sub-class receive their actions through this dict.
                Other agents' indices shouldn't appear in this dict, but will be ignored if so, because they have 
                an InternalPolicy sub-class, meaning they can
                compute their actions internally given their observation (e.g., already trained CADRL, RVO, Non-Cooperative, etc.)
            dt (float): time in seconds to run the simulation (defaults to :code:`self.dt_nominal`)

        """
        num_actions_per_agent = 2  # speed, delta heading angle
        all_actions = np.zeros((len(self.agents), num_actions_per_agent), dtype=np.float32)

        # Agents set their action (either from external or w/ find_next_action)
        #print("self.active_agents[0].heading_ego_frame")
        #print(self.active_agents[0].heading_ego_frame)
        #print(f'Num agents: {len(self.agents)}, Num alive agents: {len([agent for agent in self.agents if not agent.is_done])}')
        for agent_index, agent in enumerate(self.agents):
            if agent.is_done:
                continue
            elif agent.policy.is_external:
                agent_action = actions[agent_index]
                ext_actions = agent.policy.external_action_to_action(agent, agent_action)
                all_actions[agent_index, :] = ext_actions
            else:
                dict_obs = self.observation[agent_index]
                action_args = inspect.getfullargspec(agent.policy.find_next_action)[0]
                if 'full_agent_list' in action_args and 'active_agent_mask' in action_args:
                    all_actions[agent_index, :] = agent.policy.find_next_action(dict_obs, self.agents, agent_index,  full_agent_list = self.agents, active_agent_mask = self.active_agent_mask)
                else:
                    all_actions[agent_index, :] = agent.policy.find_next_action(dict_obs, self.agents, agent_index)
        # After all agents have selected actions, run one dynamics update
        for i, agent in enumerate(self.agents):
            agent.take_action(all_actions[i,:], dt)


    '''   #####Original#########
    def _take_action(self, actions, dt):
        """ Some agents' actions come externally through the actions arg, agents with internal policies query their policy here, 
        then each agent takes a step simultaneously.

        This makes it so an external script that steps through the environment doesn't need to
        be aware of internals of the environment, like ensuring RVO agents compute their RVO actions.
        Instead, all policies that are already trained/frozen are computed internally, and if an
        agent's policy is still being trained, it's convenient to isolate the training code from the environment this way.
        Or, if there's a real robot with its own planner on-board (thus, the agent should have an ExternalPolicy), 
        we don't bother computing its next action here and just take what the actions dict said.

        Args:
            actions (dict): keyed by agent indices, each value has a [delta heading angle, speed] command.
                Agents with an ExternalPolicy sub-class receive their actions through this dict.
                Other agents' indices shouldn't appear in this dict, but will be ignored if so, because they have 
                an InternalPolicy sub-class, meaning they can
                compute their actions internally given their observation (e.g., already trained CADRL, RVO, Non-Cooperative, etc.)
            dt (float): time in seconds to run the simulation (defaults to :code:`self.dt_nominal`)

        """
        num_actions_per_agent = 2  # speed, delta heading angle
        all_actions = np.zeros((len(self.agents), num_actions_per_agent), dtype=np.float32)

        # Agents set their action (either from external or w/ find_next_action)
        for agent_index, agent in enumerate(self.agents):
            if agent.is_done:
                continue
            elif agent.policy.is_external:
                all_actions[agent_index, :] = agent.policy.external_action_to_action(agent, actions[agent_index])
            else:
                dict_obs = self.observation[agent_index]
                all_actions[agent_index, :] = agent.policy.find_next_action(dict_obs, self.agents, agent_index)

        # After all agents have selected actions, run one dynamics update
        for i, agent in enumerate(self.agents):
            agent.take_action(all_actions[i,:], dt)
    '''

    def _update_top_down_map(self):
        """ After agents have moved, call this to update the map with their new occupancies. """
        self.map.add_agents_to_map(self.agents)
        # plt.imshow(self.map.map)
        # plt.pause(0.1)

    def set_agents(self, agents):
        """ Set the default agent configuration, which will get used at the start of each episode (and bypass calling self.test_case_fn)

        Args:
            agents (list): of :class:`~gym_collision_avoidance.envs.agent.Agent` objects that should become the self.default_agents
                and thus be loaded in that configuration every time the env resets.

        """
        self.default_agents = agents

    def _init_agents(self):
        """ Set self.agents (presumably at the start of a new episode) and set each agent's max heading change and speed based on env limits.

        self.agents gets set to self.default_agents if it exists.
        Otherwise, self.agents gets set to the result of self.test_case_fn(self.test_case_args).        
        """

        # The evaluation scripts need info about the previous episode's agents
        # (in case env.reset was called and thus self.agents was wiped)
        if self.evaluate and self.agents is not None:
            self.prev_episode_agents = copy.deepcopy(self.agents)

        # If nobody set self.default agents, query the test_case_fn
        if self.default_agents is None:
            self.agents = self.test_case_fn(**self.test_case_args)
        # Otherwise, somebody must want the agents to be reset in a certain way already
        else:
            self.agents = self.default_agents

        # Make every agent respect the same env-wide limits on actions (this probably should live elsewhere...)
        for agent in self.agents:
            agent.max_heading_change = self.max_heading_change
            agent.max_speed = self.max_speed

    def set_static_map(self, map_filename):
        """ If you want to have static obstacles, provide the path to the map image file that should be loaded.
        
        Args:
            map_filename (str or list): full path of a binary png file corresponding to the environment prior map 
                (or list of candidate map paths to randomly choose btwn each episode)
        """
        self.static_map_filename = map_filename

    def _init_static_map(self):
        """ Load the map based on its pre-provided filename, and initialize a :class:`~gym_collision_avoidance.envs.Map.Map` object

        Currently the dimensions of the world map are hard-coded.

        """
        if isinstance(self.static_map_filename, list):
            static_map_filename = np.random.choice(self.static_map_filename)
        else:
            static_map_filename = self.static_map_filename

        x_width = 16 # meters
        y_width = 16 # meters
        grid_cell_size = 0.1 # meters/grid cell
        self.map = Map(x_width, y_width, grid_cell_size, static_map_filename)

    def _compute_rewards(self):
        """ Check for collisions and reaching of the goal here, and also assign the corresponding rewards based on those calculations.
        
        Returns:
            rewards (scalar or list): is a scalar if we are only training on a single agent, or
                      is a list of scalars if we are training on mult agents
        """

        # if nothing noteworthy happened in that timestep, reward = -0.01
        rewards = self.reward_time_step*np.ones(len(self.active_agents))
        collision_with_agent, collision_with_wall, entered_norm_zone, dist_btwn_nearest_agent = \
            self._check_for_collisions()

        for i, agent in enumerate(self.active_agents):
            if agent.is_at_goal:
                if agent.was_at_goal_already is False:
                    # agents should only receive the goal reward once
                    rewards[i] = self.reward_at_goal
                    # #print("Agent %i: Arrived at goal!"
                          # % agent.id)

                if agent.was_in_collision_already is False:
                    if collision_with_agent[i] and (agent.time_since_collision >= agent.collision_cooldown):
                        rewards[i] = self.reward_collision_with_agent                 
                        agent.in_collision = True


                        #other agent is also collided now:
                        #self.agents[i].in_collision = True

                        #agent.collision_timestep = np.array( [agent.step_num] )
                        # agent.arrival_timestep = -1
                        # agent.timeout_timestep = []
                          
            else:
                # agents at their goal shouldn't be penalized if someone else
                # bumps into them
                if agent.was_in_collision_already is False:
                    if collision_with_agent[i] and (agent.time_since_collision >= agent.collision_cooldown):
                        rewards[i] = self.reward_collision_with_agent
                        agent.in_collision = True

                        # #print("Agent %i: Collision with another agent!"
                        #       % agent.id)

                        
                    elif collision_with_wall[i] and (agent.time_since_collision >= agent.collision_cooldown):
                        rewards[i] = self.reward_collision_with_wall
                        agent.in_collision = True

                        # #print("Agent %i: Collision with wall!"
                              # % agent.id)
                    else:
                        # There was no collision
                        if dist_btwn_nearest_agent[i] <= Config.GETTING_CLOSE_RANGE:
                            rewards[i] = -0.1 - dist_btwn_nearest_agent[i]/2.
                            # #print("Agent %i: Got close to another agent!"
                            #       % agent.id)
                        if abs(agent.past_actions[0, 1]) > self.wiggly_behavior_threshold:
                            # Slightly penalize wiggly behavior
                            rewards[i] += self.reward_wiggly_behavior
                        # elif entered_norm_zone[i]:
                        #     rewards[i] = self.reward_entered_norm_zone
        rewards = np.clip(rewards, self.min_possible_reward,
                          self.max_possible_reward)
        if Config.TRAIN_SINGLE_AGENT:
            rewards = rewards[0]
        return rewards

    def _check_for_collisions(self):
        """ Check whether each agent has collided with another agent or a static obstacle in the map 
        
        This method doesn't compute social zones currently!!!!!

        Returns:
            - collision_with_agent (list): for each agent, bool True if that agent is in collision with another agent
            - collision_with_wall (list): for each agent, bool True if that agent is in collision with object in map
            - entered_norm_zone (list): for each agent, bool True if that agent entered another agent's social zone
            - dist_btwn_nearest_agent (list): for each agent, float closest distance to another agent

        """
        collision_with_agent = [False for _ in self.active_agents]
        collision_with_wall = [False for _ in self.active_agents]
        entered_norm_zone = [False for _ in self.active_agents]
        dist_btwn_nearest_agent = [np.inf for _ in self.active_agents]

        #override to see what happens
        #return collision_with_agent, collision_with_wall, entered_norm_zone, dist_btwn_nearest_agent
        agent_shapes = []
        agent_front_zones = []
        agent_inds = list(range(len(self.active_agents))) #always takes 0-7
        agent_pairs = list(itertools.combinations(agent_inds, 2))
        for i, j in agent_pairs:
            dist_btwn = l2norm(self.active_agents[i].pos_global_frame, self.active_agents[j].pos_global_frame)
            combined_radius = self.active_agents[i].radius + self.active_agents[j].radius
            dist_btwn_nearest_agent[i] = min(dist_btwn_nearest_agent[i], dist_btwn - combined_radius)
            if dist_btwn <= combined_radius:
                # Collision with another agent!
                collision_with_agent[i] = True
                collision_with_agent[j] = True
        if Config.USE_STATIC_MAP:
            for i in agent_inds:
                agent = self.active_agents[i]
                [pi, pj], in_map = self.map.world_coordinates_to_map_indices(agent.pos_global_frame)
                mask = self.map.get_agent_map_indices([pi, pj], agent.radius)
                # plt.figure('static map')
                # plt.imshow(self.map.static_map + mask)
                # plt.pause(0.1)
                if in_map and np.any(self.map.static_map[mask]):
                    # Collision with wall!
                    collision_with_wall[i] = True
        return collision_with_agent, collision_with_wall, entered_norm_zone, dist_btwn_nearest_agent

    def _check_which_agents_done(self):
        """ Check if any agents have reached goal, run out of time, or collided.

        Returns:
            - which_agents_done (list): for each agent, True if agent is done, o.w. False
            - game_over (bool): depending on mode, True if all agents done, True if 1st agent done, True if all learning agents done
        """

        for agent in self.agents:
            if (agent.arrival_timestep is None) and (agent.timeout_timestep is None) and (agent.out_of_bounds_timestep is None):

                if agent.ran_out_of_time:

                   agent.timeout_timestep = agent.step_num
                   agent.arrival_timestep = -1
                   agent.out_of_bounds_timestep = -1
                    
                elif agent.is_at_goal:

                   agent.timeout_timestep = []
                   agent.arrival_timestep = agent.step_num
                   agent.out_of_bounds_timestep = -1

                elif agent.is_out_of_bounds:

                    agent.timeout_timestep = []
                    agent.arrival_timestep = -1
                    agent.out_of_bounds_timestep = agent.step_num


            # Only add collisions if the agent is not done and cooldown has expired.
            # Prevents multiple counts when agent is in collision exactly at arrival timestep
            if (agent.in_collision and (agent.time_since_collision >= agent.collision_cooldown) and not agent.is_done):
                agent.collision_timestep.append(self.episode_step_number)


            #if now in collision, but previously already reached goal, then collision will override arrival

        at_goal_condition = np.array(
                [a.is_at_goal for a in self.agents])
        ran_out_of_time_condition = np.array(
                [a.ran_out_of_time for a in self.agents])
        out_of_bounds_condition = np.array(
            [a.is_out_of_bounds for a in self.agents])
        which_agents_done = np.logical_or.reduce((at_goal_condition, ran_out_of_time_condition, out_of_bounds_condition))
        for agent_index, agent in enumerate(self.agents):
            agent.is_done = which_agents_done[agent_index]
        
        if Config.EVALUATE_MODE:
            # Episode ends when every agent is done
            game_over = np.all(which_agents_done)
        elif Config.TRAIN_SINGLE_AGENT:
            # Episode ends when ego agent is done
            game_over = which_agents_done[0]
        else:
            # Episode is done when all *learning* agents are done
            learning_agent_inds = [i for i in range(len(self.agents)) if self.agents[i].policy.is_still_learning]
            game_over = np.all(which_agents_done[learning_agent_inds])
        
        return which_agents_done, game_over


    #out of field detector
    def _check_which_agents_inside_field(self):
        bbox = Config.PLT_LIMITS #e.g. [[-1, 6], [-1, 6]]
        x_min,x_max = bbox[0]
        y_min,y_max = bbox[1]

        out_of_field_status = []
        for agent in self.agents:
            if     agent.pos_global_frame[0] < x_min or agent.pos_global_frame[0] > x_max:
                out_of_field_status.append(False)
            elif   agent.pos_global_frame[1] < y_min or agent.pos_global_frame[1] > y_max:
                out_of_field_status.append(False)
            else:
                out_of_field_status.append(True)

        #return the boolean status array, indicating which agent in within the field
        return np.array(out_of_field_status)

    def _get_obs(self):
        """ Update the map now that agents have moved, have each agent sense the world, and fill in their observations 

        Returns:
            observation (list): for each agent, a dictionary observation.

        """

        if Config.USE_STATIC_MAP:
            # Agents have moved (states have changed), so update the map view
            self._update_top_down_map()

        # Agents collect a reading from their map-based sensors
        for i, agent in enumerate(self.agents):
            agent.sense(self.agents, i, self.map)

        # Agents fill in their element of the multiagent observation vector
        for i, agent in enumerate(self.agents):
            self.observation[i] = agent.get_observation_dict(self.agents)

        return self.observation

    def _initialize_rewards(self):
        """ Set some class attributes regarding reward values based on Config """
        self.reward_at_goal = Config.REWARD_AT_GOAL
        self.reward_collision_with_agent = Config.REWARD_COLLISION_WITH_AGENT
        self.reward_collision_with_wall = Config.REWARD_COLLISION_WITH_WALL
        self.reward_getting_close = Config.REWARD_GETTING_CLOSE
        self.reward_entered_norm_zone = Config.REWARD_ENTERED_NORM_ZONE
        self.reward_time_step = Config.REWARD_TIME_STEP

        self.reward_wiggly_behavior = Config.REWARD_WIGGLY_BEHAVIOR
        self.wiggly_behavior_threshold = Config.WIGGLY_BEHAVIOR_THRESHOLD

        self.possible_reward_values = \
            np.array([self.reward_at_goal,
                      self.reward_collision_with_agent,
                      self.reward_time_step,
                      self.reward_collision_with_wall,
                      self.reward_wiggly_behavior
                      ])
        self.min_possible_reward = np.min(self.possible_reward_values)
        self.max_possible_reward = np.max(self.possible_reward_values)

    def set_plot_save_dir(self, plot_save_dir):
        """ Set where to save plots of trajectories (will get created if non-existent)
        
        Args:
            plot_save_dir (str): path to directory you'd like to save plots in

        """
        os.makedirs(plot_save_dir, exist_ok=True)
        self.plot_save_dir = plot_save_dir

    def set_perturbed_info(self, perturbed_obs):
        """ Used for robustness paper to pass info that could be visualized. Too hacky.
        """
        self.perturbed_obs = perturbed_obs

    def set_testcase(self, test_case_fn_str, test_case_args):
        """ 

        Args:
            test_case_fn_str (str): name of function in test_cases.py
        """

        # Provide a fn (which returns list of agents) and the fn's args,
        # to be called on each env.reset()
        test_case_fn = getattr(tc, test_case_fn_str, None)
        assert(callable(test_case_fn))

        # Before running test_case_fn, make sure we didn't provide any args it doesn't accept
        test_case_fn_args = inspect.signature(test_case_fn).parameters
        test_case_args_keys = list(test_case_args.keys())
        for key in test_case_args_keys:
            # #print("checking if {} accepts {}".format(test_case_fn, key))
            if key not in test_case_fn_args:
                # #print("{} doesn't accept {} -- removing".format(test_case_fn, key))
                del test_case_args[key]
        self.test_case_fn = test_case_fn
        self.test_case_args = test_case_args

if __name__ == '__main__':
    print("See example.py for a minimum working example.")
