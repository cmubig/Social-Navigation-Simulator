#!/usr/bin/env python

from __future__ import print_function
import roslib
import rospy
import math
import numpy as np
import sys
##sys.path.append('/home/sam/proj/robot_software/social_navigation/src/navigan/scripts/sgan')

import torch
from collections import deque, defaultdict
from threading import Lock

from sgan.models import TrajectoryGenerator, TrajectoryIntention
from sgan.various_length_models import LateAttentionFullGenerator
from sgan.utils import relative_to_abs

from attrdict import AttrDict

from navigan.msg import PointArray
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Pose, Twist, PointStamped, Point
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, Joy
import sensor_msgs.point_cloud2 as pc2

###########################
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PoseStamped, Vector3, Point, PointStamped, Twist, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray  #for visualization
from geometry_msgs.msg import Vector3, Pose, Point, Quaternion
from std_msgs.msg import Header, ColorRGBA

from gazebo_msgs.msg import ModelStates #gazebo_msgs/ModelStates

from gazebo_msgs.srv import GetWorldProperties
############################



VERBOSE = True
RADIUS = 0.5
VALIDSCAN = 20.0 #20.0



class naviGAN(object):
    def __init__(self):

        CHECKPOINT = '/home/sam/proj/robot_software/social_navigation/src/navigan/models/benchmark_zara1_with_model.pt'

        self.obs_rate = 0.4#0.4
        self.goal_reached = False

        #self.exe_rate = 0.15
        self.OBS_LEN = 4
        self.mutex = Lock()
        # Initializing variables for server
        self.state_queue = deque(maxlen=10000)
        self.predictGoal = True
        self.next_waypoint = Point()
        if VERBOSE:
            print('Loading Attention Generator...')
        self.intention_generator = self.get_attention_generator(torch.load(CHECKPOINT))
        if VERBOSE:
            print('Done.')

        self.output_cmd = Twist()

        self.max_linear_speed = 0.4
        self.speed_multiplier = 0.4

        #self.world_info = rospy.ServiceProxy('/gazebo/get_world_properties',GetWorldProperties)
        self.pathPub = rospy.Publisher("/navigan_path", Path, queue_size=1)



    def predict(self, agent_index, agent_poses,agent_goals,num_step, agent_poses_history ):
        #current_time = self.world_info().sim_time

        ########TrackedPt callback   mainly for other agents, not this agent
        goals_ = [ [pose.position.x, pose.position.y]    for pose  in agent_goals.pose  ]
        poses_ = [ [pose.position.x, pose.position.y]    for pose  in agent_poses.pose  ]

        orientation_ = [ pose.orientation    for pose  in agent_poses.pose  ]
        
        twists_= [ [twist.linear.x, twist.linear.y]  for twist in agent_poses.twist ]
        
        total_agents_num = len(poses_)

        #this agent's position and goal
        robot_pose = poses_[agent_index]
        robot_goal = goals_[agent_index]
        robot_orientation = orientation_[agent_index]

        other_agent_poses = poses_.copy()
        other_agent_goals = goals_.copy()

        #other agent's position and goal
        other_agent_poses.pop(agent_index)
        other_agent_goals.pop(agent_index)

        other_agents_num = len(other_agent_poses)

        #Get past position history, only need newest 3 records in this case  (3 past and 1 current position = 4 poses => obs_trajectory)
        past_poses_history = list(agent_poses_history)[: (self.OBS_LEN-1) ] #get the newest 3 record
        #Get current position 
        now_poses = [agent_poses]
        #Combine current position + past 3 record => 4 in total => obs_traj
        poses_history = now_poses + past_poses_history
        #[ now , new, old , oldest ]
        
        #if poses_history is less than 4, clone more of the oldest data and append that to the end
        if len(poses_history)< (self.OBS_LEN):               
            poses_history.extend(  [ poses_history[-1]] * ( self.OBS_LEN - len(poses_history) ) )

        #[ now , new, old , oldest ]

        #reverse the order of time to match the style of navigan....
        #becomes [oldest , old, new ,now]
        poses_history = np.flip(poses_history.copy())
        

##        print("now_poses =>",now_poses)
##        print("past_poses_history =>",past_poses_history)
##        print("poses_history =>",poses_history)
            
        rearranged_history = [np.nan] * self.OBS_LEN


        newest_all_agents_poses = [ [pose.position.x, pose.position.y]    for pose  in poses_history[-1].pose  ]         
        newest_robot_pose = [newest_all_agents_poses[agent_index]]

        newest_other_agent_poses = newest_all_agents_poses.copy()
        newest_other_agent_poses.pop(agent_index)   #remove agent's position, leaving only other's position

        
        for i in range(self.OBS_LEN):
            #process the position of all agents in Time[i], re-arrange such that  poses = self , other_0, other_1, 2,3,4...
            past_all_agents_poses = [ [pose.position.x, pose.position.y]    for pose  in poses_history[i].pose  ]         
            past_robot_pose = [past_all_agents_poses[agent_index]]
##            print("past_robot_pose")
##            print(past_robot_pose)

           
            past_other_agent_poses = past_all_agents_poses.copy()
            past_other_agent_poses.pop(agent_index)   #remove agent's position, leaving only other's position

            ####exaggeration...  exaggerate peds agents, by comparing to robot
            #################
            for j in range(len(past_other_agent_poses)):
                #BIG mistake! the position tuple for agent_index is removed before referenced!
                #displacement = np.array( past_robot_pose[0] ) - np.array( past_other_agent_poses[j] )

                #Fixed
                displacement = np.array( past_robot_pose[0] ) - np.array( past_other_agent_poses[j] )
                
                #should not be using newest , should be relative to oldest position
                #newest_xxxx_xxx are not in use currently, consider remove it
                
                #displacement = np.array( newest_robot_pose[0] ) - np.array( newest_other_agent_poses[j] )   
                norm_disp = np.sqrt(displacement[0]**2+displacement[1]**2)

                if norm_disp > 2:
                    displacement *= 1.75/norm_disp
                elif norm_disp > 1:
                    displacement *= 0.75/norm_disp
                elif norm_disp > 0.5:
                    displacement *= 0.40/norm_disp
                else:
                    displacement /= 2
            ##############
                past_other_agent_poses[j] = list( np.array( past_other_agent_poses[j] ) + displacement)

            #################
            #Output array of all poses in time series,  each element in array contain all agent poses, [0] element in rearranged_history is the oldest 
            rearranged_history[i] = past_robot_pose + past_other_agent_poses
##            print("combined=>")
##            print(rearranged_history[i])

            


            
#############################################################        
##[INFO] [1587987020.665589]: TTTTTTTTTTTTTTTT
##[INFO] [1587987020.666708]: [[1.0564772892265575, -2.9439042539070246], [1.1145057294792293, -5.99606735577477], [1.1153218711699553, 2.640532728710876e-06]]
##obs_traj => [[[ 1.13049939e+00 -2.86988351e+00]
##  [ 1.39572241e+00 -5.99669114e+00]
##  [ 1.39613869e+00 -2.94107679e-06]]
##
## [[ 1.11235920e+00 -2.88802337e+00]
##  [ 1.32198718e+00 -5.99649834e+00]
##  [ 1.32250828e+00 -1.21340563e-06]]
##
## [[ 1.07441882e+00 -2.92596304e+00]
##  [ 1.22978925e+00 -5.99629044e+00]
##  [ 1.23044144e+00  6.46133517e-07]]
##
## [[ 1.05647729e+00 -2.94390425e+00]
##  [ 1.11450573e+00 -5.99606736e+00]
##  [ 1.11532187e+00  2.64053273e-06]]]
##shape (4, 3, 2)
##
########################################################
      
        obs_traj = np.array(rearranged_history)

        try:
            if np.isnan(np.sum(obs_traj)):
                print("nan")
                return Twist()
        except TypeError:
            print("TypeError")
            return Twist()
##        print("obs_traj =>",obs_traj)
##        print("shape",np.array(obs_traj).shape)
            
##            obs_traj = [robot_pos]+peds_array
##            obs_traj = np.array(obs_traj).transpose((1,0,2))

        self.goalData = robot_goal
        goal = np.array([self.goalData[0], self.goalData[1]])

##        rospy.loginfo("[goal]")
##        rospy.loginfo(goal)
        
        goal = torch.from_numpy(goal).type(torch.float)

        obs_traj = torch.from_numpy(obs_traj).type(torch.float)
        obs_traj_rel = obs_traj - obs_traj[0,:,:]

##        rospy.loginfo("[obs_traj]")
##        rospy.loginfo(obs_traj)
##
##        rospy.loginfo("[obs_traj_rel]")
##        rospy.loginfo(obs_traj_rel)

        seq_start_end = torch.from_numpy(np.array([[0,obs_traj.shape[1]]]))

        goals_rel = goal - obs_traj[0,0,:]
        goals_rel = goals_rel.repeat(1,obs_traj.shape[1],1)


        # move everything to GPU
        obs_traj = obs_traj.cuda()
        obs_traj_rel = obs_traj_rel.cuda()
        seq_start_end = seq_start_end.cuda()
        goals_rel = goals_rel.cuda()

        pred_traj_fake = self.feedforward(obs_traj, obs_traj_rel, seq_start_end, goals_rel)
        # print(pred_traj_fake.size())
        ptf = pred_traj_fake[:,0,:].cpu().numpy()  #[12,2]

        # pick the 3rd position as robot predicted step
        # self.next_waypoint = Point(ptf[2,0],ptf[2,1], 0)
        self.next_waypoint = Point(ptf[2,0],ptf[2,1], 0)   #change to 1st [psotopm as robot prediction step
        #reach the goal check  need to be tuned
        #rospy.loginfo("waypoint is x: {}, y:{}".format(self.next_waypoint.x, self.next_waypoint.y))

        self.output_cmd.linear.x = (self.next_waypoint.x - robot_pose[0]) * self.speed_multiplier             
        if    self.next_waypoint.x > robot_pose[0]   : self.output_cmd.linear.x =  self.max_linear_speed
        if    self.next_waypoint.x < robot_pose[0]   : self.output_cmd.linear.x = -self.max_linear_speed

        self.output_cmd.linear.y = (self.next_waypoint.y - robot_pose[1]) * self.speed_multiplier
        if    self.next_waypoint.y > robot_pose[1]   : self.output_cmd.linear.y =  self.max_linear_speed
        if    self.next_waypoint.y < robot_pose[1]   : self.output_cmd.linear.y = -self.max_linear_speed
        return self.output_cmd

        #for visualization in rviz
        nextPath = Path()
        nextPath.header.frame_id = "/map"
        nextPath.header.stamp = current_time

        for index in range(0, ptf.shape[0]):
            nextPath.poses.append(PoseStamped())
            nextPath.poses[index].pose.position.x = ptf[index,0]
            nextPath.poses[index].pose.position.y = ptf[index,1]
            nextPath.poses[index].pose.position.z = 0.5
        self.pathPub.publish(nextPath)
        return

        # late attention model by full state (actually used)
    def get_attention_generator(self, checkpoint, best=False):
        args = AttrDict(checkpoint['args'])
        generator = LateAttentionFullGenerator(
            goal_dim=(2,),
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            pooling_type=args.pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            batch_norm=args.batch_norm,
            spatial_dim=2)
        if best:
            generator.load_state_dict(checkpoint['g_waypointbest_state'])
        else:
            generator.load_state_dict(checkpoint['g_state'])
        generator.cuda()
        generator.train()
        return generator



    def reachGoalCheck(self, robotPosition, goalPosition, _r=RADIUS):
        if (robotPosition.x-goalPosition[0])**2 + (robotPosition.y-goalPosition[1])**2 < _r**2:
            return True
        else:
            return False



    def feedforward(self, obs_traj, obs_traj_rel, seq_start_end, goals_rel):
        """
        obs_traj: torch.Tensor([4, num_agents, 2])
        obs_traj_rel: torch.Tensor([4, num_agents, 2])
        seq_start_end: torch.Tensor([batch_size, 2]) #robot+#pedstrains
        goals_rel: torch.Tensor([1, num_agents, 2])
        """

        with torch.no_grad():
            pred_traj_fake_rel, _ = self.intention_generator(obs_traj, obs_traj_rel, seq_start_end, goal_input=goals_rel)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0])
        return pred_traj_fake


