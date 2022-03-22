from gym_collision_avoidance.envs.policies.LearningCADRL.policy.sim_policy.policy_factory import policy_factory
from gym_collision_avoidance.envs.policies.LearningCADRL.policy.cadrl import CADRL
# from gym_collision_avoidance.envs.policies.LearningCADRL.policy.lstm_rl import LstmRL
# from gym_collision_avoidance.envs.policies.LearningCADRL.policy.sarl import SARL

policy_factory['cadrl'] = CADRL
# policy_factory['lstm_rl'] = LstmRL
# policy_factory['sarl'] = SARL
