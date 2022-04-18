import abc
import numpy as np


class Policy(object):
    def __init__(self):
        """
        Base class for all policies, has an abstract method predict().
        """
        self.trainable = False
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None
        # if agent is assumed to know the dynamics of real world
        self.env = None

    @abc.abstractmethod
    def configure(self, config):
        return

    def set_phase(self, phase):
        self.phase = phase

    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        self.env = env

    def get_model(self):
        return self.model

    @abc.abstractmethod
    def predict(self, state):
        """
        Policy takes state as input and output an action

        """
        return

    def predict_reward(self, state_agent, state_other, action):
        collision = False
        discomfort_dist = state_agent.radius/2
        dmin = float('inf')
        print("predict: ",state_other[0])
        for i in range(len(state_other)):
            px = state_other[i].px - state_agent.px
            py = state_other[i].py - state_agent.py
        
            vx = state_other[i].vx - action.v * np.cos(action.r + state_agent.theta)
            vy = state_other[i].vy - action.v * np.sin(action.r + state_agent.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = self.point_to_segment_dist(px, py, ex, ey, 0, 0) - state_agent.radius - state_other.radius
            if closest_dist < 0:
                collision = True
                break

            elif closest_dist < dmin:
                dmin = closest_dist


        reaching_goal = np.linalg.norm(np.array(state_agent.px,state_agent.py) - np.array(state_agent.gx,state_agent.gy)) < 0.3

        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1.0
        # elif dmin < discomfort_dist:
        #     # only penalize agent for getting too close if it's visible
        #     # adjust the reward based on FPS
        #     reward = -0.1
        else:
            reward = 0
        
        print("preidicted reward: ", reward)

        return reward


    @staticmethod
    def reach_destination(state):
        self_state = state.self_state
        if np.linalg.norm((self_state.py - self_state.gy, self_state.px - self_state.gx)) < self_state.radius/2:
            print("dist2 goal: ",np.linalg.norm((self_state.py - self_state.gy, self_state.px - self_state.gx)))
            return True
        else:
            return False
