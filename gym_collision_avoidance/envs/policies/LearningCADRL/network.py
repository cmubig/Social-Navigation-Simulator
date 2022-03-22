import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym

from gym_collision_avoidance.envs.policies.LearningCADRL.sim_utils.robot import Robot
from gym_collision_avoidance.envs.policies.LearningCADRL.utils.trainer import Trainer
from gym_collision_avoidance.envs.policies.LearningCADRL.utils.memory import ReplayMemory
from gym_collision_avoidance.envs.policies.LearningCADRL.utils.explorer import Explorer
from gym_collision_avoidance.envs.policies.LearningCADRL.policy.policy_factory import policy_factory

class Train():
    def __init__(self):
        self.episode = 0
        self.parser = argparse.ArgumentParser('Parse configuration file')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    
    def initialize(self):
        
        