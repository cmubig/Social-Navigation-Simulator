import numpy as np
#from gym_collision_avoidance.experiments.src.master_config import Master_Config
from gym_collision_avoidance.experiments.src.master_config_deploy import Master_Config
master_config = Master_Config()

class Config(object):
    def __init__(self):
        #########################################################################
        # GENERAL PARAMETERS
        self.COLLISION_AVOIDANCE = True
        self.continuous, self.discrete = range(2) # Initialize game types as enum
        self.ACTION_SPACE_TYPE   = self.continuous

        ### DISPLAY
        self.ANIMATE_EPISODES    = True
        self.SHOW_EPISODE_PLOTS = False
        self.SAVE_EPISODE_PLOTS = True
        if not hasattr(self, "PLOT_CIRCLES_ALONG_TRAJ"):
            self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.ANIMATION_PERIOD_STEPS = 2 # plot every n-th DT step (if animate mode on)
        self.PLT_LIMITS = None
        self.PLT_FIG_SIZE = (10, 8)

        if not hasattr(self, "USE_STATIC_MAP"):
            self.USE_STATIC_MAP = False
        
        ### TRAIN / PLAY / EVALUATE
        self.TRAIN_MODE           = True # Enable to see the trained agent in action (for testing)
        self.PLAY_MODE           = False # Enable to see the trained agent in action (for testing)
        self.EVALUATE_MODE       = False # Enable to see the trained agent in action (for testing)
        
        ### REWARDS
        self.REWARD_AT_GOAL = 1.0 # reward given when agent reaches goal position
        self.REWARD_COLLISION_WITH_AGENT = -0.25 # reward given when agent collides with another agent
        self.REWARD_COLLISION_WITH_WALL = -0.25 # reward given when agent collides with wall
        self.REWARD_GETTING_CLOSE   = -0.1 # reward when agent gets close to another agent (unused?)
        self.REWARD_ENTERED_NORM_ZONE   = -0.05 # reward when agent enters another agent's social zone
        self.REWARD_TIME_STEP   = 0.0 # default reward given if none of the others apply (encourage speed)
        self.REWARD_WIGGLY_BEHAVIOR = 0.0
        self.WIGGLY_BEHAVIOR_THRESHOLD = np.inf
        self.COLLISION_DIST = 0.0 # meters between agents' boundaries for collision
        self.GETTING_CLOSE_RANGE = 0.2 # meters between agents' boundaries for collision
        # self.SOCIAL_NORMS = "right"
        # self.SOCIAL_NORMS = "left"
        self.SOCIAL_NORMS = "none"

        ### SIMULATION
        self.DT             = 0.2 # seconds between simulation time steps
        self.NEAR_GOAL_THRESHOLD = 0.2
        self.MAX_TIME_RATIO = 2. # agent has this number times the straight-line-time to reach its goal before "timing out"
        
        ### TEST CASE SETTINGS
        self.TEST_CASE_FN = "get_testcase_random"
        self.TEST_CASE_ARGS = {
            'policy_to_ensure': 'learning_ga3c',
            'policies': ['noncoop', 'learning_ga3c', 'static'],
            'policy_distr': [0.05, 0.9, 0.05],
            'speed_bnds': [0.5, 2.0],
            'radius_bnds': [0.2, 0.8],
            'side_length': [
                {'num_agents': [0,5], 'side_length': [4,5]}, 
                {'num_agents': [5,np.inf], 'side_length': [6,8]},
                ],
            # 'agents_sensors': ['other_agents_states_encoded'],
        }

        if not hasattr(self, "MAX_NUM_AGENTS_IN_ENVIRONMENT"):
            self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 4
        if not hasattr(self, "MAX_NUM_AGENTS_TO_SIM"):
            self.MAX_NUM_AGENTS_TO_SIM = 4
        self.MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        if not hasattr(self, "MAX_NUM_OTHER_AGENTS_OBSERVED"):
            self.MAX_NUM_OTHER_AGENTS_OBSERVED = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1

        ### EXPERIMENTS
        self.PLOT_EVERY_N_EPISODES = 100 # for tensorboard visualization

        ### SENSORS
        self.SENSING_HORIZON  = np.inf
        # self.SENSING_HORIZON  = 3.0
        self.LASERSCAN_LENGTH = 512 # num range readings in one scan
        self.LASERSCAN_NUM_PAST = 3 # num range readings in one scan
        self.STATIC_MAP_SIZE = 16
        self.STATIC_MAP_GRID_CELL_SIZE = 0.1
        #occupancy_grid_sz = 2*(int(self.STATIC_MAP_SIZE/self.STATIC_MAP_GRID_CELL_SIZE))
        static_map_sz = int(self.STATIC_MAP_SIZE/self.STATIC_MAP_GRID_CELL_SIZE)
        occupancy_grid_sz = int(5/self.STATIC_MAP_GRID_CELL_SIZE)
        self.NUM_STEPS_IN_OBS_HISTORY = 1 # number of time steps to store in observation vector
        self.NUM_PAST_ACTIONS_IN_STATE = 0

        ### RVO AGENTS
        self.RVO_TIME_HORIZON = 20.0 #5.0
        self.RVO_COLLAB_COEFF = 0.5
        self.RVO_ANTI_COLLAB_T = 1.0

        ### OBSERVATION VECTOR
        self.TRAIN_SINGLE_AGENT = False
        self.STATE_INFO_DICT = {
            'dist_to_goal': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("dist_to_goal")',
                'std': np.array([5.], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
                },
            'radius': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([0.5], dtype=np.float32)
                },
            'heading_ego_frame': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.pi, np.pi],
                'attr': 'get_agent_data("heading_ego_frame")',
                'std': np.array([3.14], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
                },
            'pref_speed': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("pref_speed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
                },
            'num_other_agents': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("num_other_agents_observed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
                },
            'other_agent_states': {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)
                },
            'other_agents_states': {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED,7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32), (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32), (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                },
            'laserscan': {
                'dtype': np.float32,
                'size': (self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH),
                'bounds': [0., 6.],
                'attr': 'get_sensor_data("laserscan")',
                'std': 5.*np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32),
                'mean': 5.*np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32)
                },
            'occupancy_grid': {
                'dtype': bool,
                'size': (occupancy_grid_sz, occupancy_grid_sz),
                'bounds': [0., 1.],
                'attr': 'get_sensor_data("occupancy_grid")',
                'std': 0.5*np.ones((occupancy_grid_sz, occupancy_grid_sz), dtype=bool),
                'mean': 0.5*np.ones((occupancy_grid_sz, occupancy_grid_sz), dtype=bool)
                },
            'static_map': {
                'dtype': bool,
                'size': (static_map_sz, static_map_sz),
                'bounds': [0., 1.],
                'attr': 'get_sensor_data("static_map")',
                'std': 0.5*np.ones((static_map_sz, static_map_sz), dtype=bool),
                'mean': 0.5*np.ones((static_map_sz, static_map_sz), dtype=bool)
                },
            'is_learning': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0., 1.],
                'attr': 'get_agent_data_equiv("policy.str", "learning")'
                },
            'other_agents_states_encoded': {
                'dtype': np.float32,
                'size': 100.,
                'bounds': [0., 1.],
                'attr': 'get_sensor_data("other_agents_states_encoded")'
                }
            }
        self.setup_obs()
    
        # self.AGENT_SORTING_METHOD = "closest_last"
        self.AGENT_SORTING_METHOD = "closest_first"
        # self.AGENT_SORTING_METHOD = "time_to_impact"

    def setup_obs(self):
        if not hasattr(self, "STATES_IN_OBS"):
            self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'other_agents_states', 'occupancy_grid', 'static_map']
            # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agent_states', 'use_ppo', 'laserscan']
            # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agent_states', 'use_ppo'] # 2-agent net
            # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agents_states', 'use_ppo', 'num_other_agents', 'laserscan'] # LSTM
        if not hasattr(self, "STATES_NOT_USED_IN_POLICY"):
            self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        self.MEAN_OBS = {}; self.STD_OBS = {}
        for state in self.STATES_IN_OBS:
            if 'mean' in self.STATE_INFO_DICT[state]:
                self.MEAN_OBS[state] = self.STATE_INFO_DICT[state]['mean']
            if 'std' in self.STATE_INFO_DICT[state]:
                self.STD_OBS[state] = self.STATE_INFO_DICT[state]['std']

class EvaluateConfig(Config):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT   = master_config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        Config.__init__(self)
        self.EVALUATE_MODE                   = master_config.EVALUATE_MODE
        self.TRAIN_MODE                      = master_config.TRAIN_MODE
        self.DT                              = master_config.DT
        self.MAX_TIME_RATIO                  = master_config.MAX_TIME_RATIO

class Example(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS                = master_config.SAVE_EPISODE_PLOTS
        self.SHOW_EPISODE_PLOTS                = master_config.SHOW_EPISODE_PLOTS
        self.ANIMATE_EPISODES                  = master_config.ANIMATE_EPISODES
        self.NEAR_GOAL_THRESHOLD               = master_config.NEAR_GOAL_THRESHOLD
        self.PLT_LIMITS                        = [[-10, 10], [-10, 10]]
        self.PLT_FIG_SIZE                      = master_config.PLT_FIG_SIZE
        self.PLOT_CIRCLES_ALONG_TRAJ           = master_config.PLOT_CIRCLES_ALONG_TRAJ
        self.NUM_AGENTS_TO_TEST                = master_config.NUM_AGENTS_TO_TEST
        self.POLICIES_TO_TEST                  = master_config.POLICIES_TO_TEST
        self.NUM_TEST_CASES                    = master_config.NUM_TEST_CASES


        self.MAX_NUM_AGENTS_IN_ENVIRONMENT     = master_config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        self.MAX_NUM_OTHER_AGENTS_OBSERVED     = master_config.MAX_NUM_OTHER_AGENTS_OBSERVED
        self.agent_time_out                    = master_config.agent_time_out

class Formations(EvaluateConfig):   #CADRL
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = True
        self.NEAR_GOAL_THRESHOLD = 0.2
        self.PLT_LIMITS = [[-5, 6], [-2, 7]]
        self.PLT_FIG_SIZE = (10,10)
        self.PLOT_CIRCLES_ALONG_TRAJ = False
        self.NUM_AGENTS_TO_TEST = [6]
        self.POLICIES_TO_TEST = ['GA3C-CADRL-10']
        self.NUM_TEST_CASES = 2
        self.LETTERS = ['C', 'A', 'D', 'R', 'L']

############################
class Custom(EvaluateConfig):  
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS                = master_config.SAVE_EPISODE_PLOTS
        self.SHOW_EPISODE_PLOTS                = master_config.SHOW_EPISODE_PLOTS 
        self.ANIMATE_EPISODES                  = master_config.ANIMATE_EPISODES
        self.NEAR_GOAL_THRESHOLD               = master_config.NEAR_GOAL_THRESHOLD
        self.PLT_LIMITS                        = master_config.PLT_LIMITS
        self.PLT_FIG_SIZE                      = master_config.PLT_FIG_SIZE 
        self.PLOT_CIRCLES_ALONG_TRAJ           = master_config.PLOT_CIRCLES_ALONG_TRAJ
        self.NUM_AGENTS_TO_TEST                = master_config.NUM_AGENTS_TO_TEST
        self.POLICIES_TO_TEST                  = master_config.POLICIES_TO_TEST 
        self.NUM_TEST_CASES                    = master_config.NUM_TEST_CASES 
 

        self.MAX_NUM_AGENTS_IN_ENVIRONMENT     = master_config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        self.MAX_NUM_OTHER_AGENTS_OBSERVED     = master_config.MAX_NUM_OTHER_AGENTS_OBSERVED 
        self.agent_time_out                    = master_config.agent_time_out 
#######################################
        
class SmallTestSuite(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = False
        self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.NUM_TEST_CASES = 4

class FullTestSuite(EvaluateConfig):   #FULL_TEST
    def __init__(self):
        self.MAX_NUM_OTHER_AGENTS_OBSERVED = 19
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = False
        self.PLOT_CIRCLES_ALONG_TRAJ = True

        self.NUM_TEST_CASES = 4
        self.NUM_AGENTS_TO_TEST = [2,3,4]
        self.RECORD_PICKLE_FILES = False

        # # DRLMACA
        # self.FIXED_RADIUS_AND_VPREF = True
        # self.NEAR_GOAL_THRESHOLD = 0.8

        # Normal
        self.POLICIES_TO_TEST = [
            'CADRL', 'RVO', 'GA3C-CADRL-10'
            # 'GA3C-CADRL-4-WS-4-1', 'GA3C-CADRL-4-WS-4-2', 'GA3C-CADRL-4-WS-4-3', 'GA3C-CADRL-4-WS-4-4', 'GA3C-CADRL-4-WS-4-5',
            # 'GA3C-CADRL-4-WS-6-1', 'GA3C-CADRL-4-WS-6-2', 'GA3C-CADRL-4-WS-6-3', 'GA3C-CADRL-4-WS-6-4',
            # 'GA3C-CADRL-4-WS-8-1', 'GA3C-CADRL-4-WS-8-2', 'GA3C-CADRL-4-WS-8-3', 'GA3C-CADRL-4-WS-8-4',
            # 'GA3C-CADRL-4-LSTM-1', 'GA3C-CADRL-4-LSTM-2', 'GA3C-CADRL-4-LSTM-3', 'GA3C-CADRL-4-LSTM-4', 'GA3C-CADRL-4-LSTM-5',
            # 'GA3C-CADRL-10-WS-4-1', 'GA3C-CADRL-10-WS-4-2', 'GA3C-CADRL-10-WS-4-3', 'GA3C-CADRL-10-WS-4-4', 'GA3C-CADRL-10-WS-4-5',
            # 'GA3C-CADRL-10-WS-6-1', 'GA3C-CADRL-10-WS-6-2', 'GA3C-CADRL-10-WS-6-3', 'GA3C-CADRL-10-WS-6-4',
            # 'GA3C-CADRL-10-WS-8-1', 'GA3C-CADRL-10-WS-8-2', 'GA3C-CADRL-10-WS-8-3', 'GA3C-CADRL-10-WS-8-4',
            # 'GA3C-CADRL-10-LSTM-1', 'GA3C-CADRL-10-LSTM-2', 'GA3C-CADRL-10-LSTM-3', 'GA3C-CADRL-10-LSTM-4', 'GA3C-CADRL-10-LSTM-5',
            # 'CADRL', 'RVO'
            ]
        self.FIXED_RADIUS_AND_VPREF = False
        self.NEAR_GOAL_THRESHOLD = 0.2


class CollectRegressionDataset(EvaluateConfig):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 4
        self.MAX_NUM_AGENTS_TO_SIM = 4
        self.DATASET_NAME = ""

        # # Laserscan mode
        # self.USE_STATIC_MAP = True
        # self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'laserscan']
        # self.DATASET_NAME = "laserscan_"

        EvaluateConfig.__init__(self)
        self.TEST_CASE_ARGS['policies'] = 'CADRL'
        self.AGENT_SORTING_METHOD = "closest_first"

        # # Laserscan mode
        # self.TEST_CASE_ARGS['agents_sensors'] = ['laserscan', 'other_agents_states']


class MinimalMap(Example):
    def __init__(self):
        Example.__init__(self)
        self.USE_STATIC_MAP = True