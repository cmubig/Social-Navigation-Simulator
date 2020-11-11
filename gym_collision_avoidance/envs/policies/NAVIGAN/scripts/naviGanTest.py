#!/usr/bin/env python

from __future__ import print_function
import roslib
import rospy
import math
import numpy as np
import tf
import sys
sys.path.append('/home/sam/proj/robot_software/social_navigation/src/navigan/scripts/sgan')
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
VERBOSE = True
RADIUS = 0.5
VALIDSCAN = 20.0 #20.0

# Not used
# def get_intention_generator(checkpoint, best=False):
#     args = AttrDict(checkpoint['args'])
#     generator = TrajectoryIntention(
#         obs_len=args.obs_len,
#         pred_len=args.pred_len,
#         embedding_dim=args.embedding_dim,
#         encoder_h_dim=args.encoder_h_dim_g,
#         decoder_h_dim=args.decoder_h_dim_g,
#         mlp_dim=args.mlp_dim,
#         num_layers=args.num_layers,
#         goal_dim=(2,),
#         dropout=args.dropout,
#         bottleneck_dim=args.bottleneck_dim,
#         batch_norm=args.batch_norm)
#     if best:
#         generator.load_state_dict(checkpoint['i_best_state'])
#     else:
#         generator.load_state_dict(checkpoint['i_state'])
#     generator.cuda()
#     generator.train()
#     return generator

# Not used
# def get_force_generator(checkpoint, best=False):
#     args = AttrDict(checkpoint['args'])
#     generator = TrajectoryGenerator(
#         obs_len=args.obs_len,
#         pred_len=args.pred_len,
#         embedding_dim=args.embedding_dim,
#         encoder_h_dim=args.encoder_h_dim_g,
#         decoder_h_dim=args.decoder_h_dim_g,
#         mlp_dim=args.mlp_dim,
#         num_layers=args.num_layers,
#         noise_dim=args.noise_dim,
#         noise_type=args.noise_type,
#         noise_mix_type=args.noise_mix_type,
#         pooling_type=args.pooling_type,
#         pool_every_timestep=args.pool_every_timestep,
#         dropout=args.dropout,
#         bottleneck_dim=args.bottleneck_dim,
#         neighborhood_size=args.neighborhood_size,
#         grid_size=args.grid_size,
#         batch_norm=args.batch_norm)
#     if best:
#         generator.load_state_dict(checkpoint['g_best_state'])
#     else:
#         generator.load_state_dict(checkpoint['g_state'])
#     generator.cuda()
#     generator.train()
#     return generator

# late attention model by full state (actually used)
def get_attention_generator(checkpoint, best=False):
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

class naviGan(object):
    def __init__(self, checkpoint):
        self.waypointPub = rospy.Publisher("/way_point", PointStamped, queue_size=5)
        self.pathPub = rospy.Publisher("/navigan_path", Path, queue_size=1)

        self.trackedPtSub = rospy.Subscriber("/tracked_points", PointCloud2, self.trackedPtCallback, queue_size=2)
        self.goalSub = rospy.Subscriber("/navigan_goal", PointStamped, self.goalCallback, queue_size=10)
        self.odomSub = rospy.Subscriber("/state_estimation", Odometry, self.odomCallback, queue_size=1)
        self.joySub = rospy.Subscriber("/joy", Joy, self.joyCallback, queue_size=1)


        self.obs_rate = 0.4
        self.goal_reached = False

        #self.exe_rate = 0.15
        self.OBS_LEN = 4
        self.mutex = Lock()
        # Initializing variables for server
        self.state_queue = deque(maxlen=10000)
        self.goalInit = False
        self.scanInit = False
        self.odomInit = False
        self.predictGoal = True
        self.next_waypoint = Point()
        if VERBOSE:
            print('Loading Attention Generator...')
        self.intention_generator = get_attention_generator(checkpoint)
        if VERBOSE:
            print('Done.')



    def trackedPtCallback(self, _scanIn):
        if (self.odomInit is False or self.goalInit is False):
            return
        scanTime = _scanIn.header.stamp

        if (self.scanInit is False):
            self.scanInitTime = scanTime
            self.scanInit = True
            print("scan init")

        ####################### Mutex Acquire #######################
        self.mutex.acquire()
        active_peds_id = set()
        peds_pos_t = defaultdict(lambda: None)
        minDistance = 9999
        for p in pc2.read_points(_scanIn, field_names = ("x", "y", "z", "h","s","v"), skip_nans=True):
            active_peds_id.add(p[3])
            peds_pos_t[p[3]] = np.array([p[0], p[1]])
            dist = math.sqrt((self.odomData.position.x-p[0])**2 + (self.odomData.position.y-p[1])**2)
            if dist < minDistance:
                minDistance = dist


        self.state_queue.append({'active_peds_id': active_peds_id,
                                 'peds_pos_t': peds_pos_t,
                                 'time_stamp': scanTime,
                                 'robot_pos_t': np.array([self.odomData.position.x, self.odomData.position.y]),
                                 'robot_rot_t': self.odomData.orientation})
        self.mutex.release()
        ####################### Mutex Release #######################
        #rospy.loginfo('[{}] state queue append, active id ={}, duration since started {}'.format( \
        #                        rospy.get_name(), len(active_peds_id), (scanTime - self.scanInitTime).to_sec()))
        print("minidist",minDistance)
        if (True and minDistance <= VALIDSCAN):
            # if( scanTime - self.scanInitTime > rospy.Duration.from_sec(5*self.obs_rate)):
            if( scanTime - self.scanInitTime > rospy.Duration.from_sec(5*self.obs_rate)):
                rospy.loginfo('[{}] predicting'.format(rospy.get_name()))
                self.predict(scanTime)
        else:
            rospy.loginfo('[{}] Using goalpoint'.format(rospy.get_name()))
            self.next_waypoint = self.goalData

        tmpPoint = PointStamped()
        tmpPoint.header.frame_id = "/map"
        tmpPoint.header.stamp = scanTime
        tmpPoint.point = self.next_waypoint
        self.waypointPub.publish(tmpPoint)
        return

    def predict(self, current_time):
        #current_time = rospy.get_rostime()
        ####################### Mutex Block #######################
        # Acquire mutex and collect information
        self.mutex.acquire()
        state = self.state_queue[-1]
        # most recent state
        active_peds_id = state['active_peds_id']
        peds_pos = defaultdict(deque)
        robot_pos = deque()
        for id in active_peds_id:
            peds_pos[id].appendleft(state['peds_pos_t'][id])
        robot_pos.appendleft(state['robot_pos_t'])

        runner = len(self.state_queue)-1
        # load the previous step
        for i in range(1, self.OBS_LEN):

            #rospy.loginfo('duration looking for is {}, length of queue is {}'.format(i*self.obs_rate, len(self.state_queue)))
            while current_time - self.state_queue[runner]['time_stamp'] < \
                rospy.Duration.from_sec(i*self.obs_rate):
            #    rospy.loginfo('runner is {}, current duration {}'.format(runner, (current_time - self.state_queue[runner]['time_stamp']).to_sec() ))
                runner -= 1
            state = self.state_queue[runner]
            for id in active_peds_id:
                if id in state['active_peds_id']:
                    peds_pos[id].appendleft(state['peds_pos_t'][id])
            robot_pos.appendleft(state['robot_pos_t'])
        self.mutex.release()
        ####################### Mutex Block #######################

        robot_pos = np.array(robot_pos)

        peds_array = []
        for id in active_peds_id:
            tmp = np.array(peds_pos[id])

            # Exaggerate Peds' distance to robot
            displacement = robot_pos[-1,:] - tmp[-1,:]

            norm_disp = np.sqrt(displacement[0]**2+displacement[1]**2)

            if norm_disp > 2:
                displacement *= 1.75/norm_disp
            elif norm_disp > 1:
                displacement *= 0.75/norm_disp
            elif norm_disp > 0.5:
                displacement *= 0.40/norm_disp
            else:
                displacement /= 2

            tmp += displacement.reshape((1,2))
            displacement = robot_pos[-1,:] - tmp[-1,:]
            '''
            rospy.loginfo('[{}] dist to ped: {}, after modified: {}'.format( \
                                rospy.get_name(), norm_disp, \
                                np.sqrt(displacement[0]**2+displacement[1]**2)))
            '''
            tmp = np.pad(tmp, ((self.OBS_LEN-len(tmp),0), (0,0)), 'edge')
            peds_array.append(tmp)


        try:
            obs_traj = [robot_pos]+peds_array
            obs_traj = np.array(obs_traj).transpose((1,0,2))
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()

        goal = np.array([self.goalData.x, self.goalData.y])
        goal = torch.from_numpy(goal).type(torch.float)

        obs_traj = torch.from_numpy(obs_traj).type(torch.float)
        obs_traj_rel = obs_traj - obs_traj[0,:,:]

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
        self.next_waypoint = Point(ptf[2,0],ptf[2,1], 0)
        #reach the goal check  need to be tuned
        rospy.loginfo("waypoint is x: {}, y:{}".format(self.next_waypoint.x, self.next_waypoint.y))
        if self.reachGoalCheck(self.next_waypoint, self.goalData, 0.3) is not True:
            dist = math.sqrt((self.odomData.position.x-self.goalData.x)**2 + (self.odomData.position.y-self.goalData.y)**2)
            length =  math.sqrt((self.next_waypoint.x-self.odomData.position.x)**2 + (self.next_waypoint.y-self.odomData.position.y)**2)
            self.next_waypoint.x = dist / length * (self.next_waypoint.x - self.odomData.position.x) + self.odomData.position.x
            self.next_waypoint.y = dist / length * (self.next_waypoint.y - self.odomData.position.y) + self.odomData.position.y
        rospy.loginfo("rescale waypoint x: {}, y:{}".format(self.next_waypoint.x, self.next_waypoint.y))

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


    def joyCallback(self, _joyIn):
        if(_joyIn.buttons[4] > 0.5):
            self.predictGoal = True
        else:
            self.predictGoal = False

    def reachGoalCheck(self, robotPosition, goalPosition, _r=RADIUS):
        if (robotPosition.x-goalPosition.x)**2 + (robotPosition.y-goalPosition.y)**2 < _r**2:
            return True
        else:
            return False

    def goalCallback(self, _goalIn):
        if(self.goalInit is False):
            self.goalInit = True
            print("goal init")
        goal_time = _goalIn.header.stamp
        self.goalData = _goalIn.point
        return

    def odomCallback(self, _odomIn):
        if (self.odomInit is False):
            self.odomInit = True
            print("odom init")
        odom_time = _odomIn.header.stamp

        self.odomData = _odomIn.pose.pose
        if (self.goalInit and self.odomInit):
            self.goal_reached = self.reachGoalCheck(self.odomData.position, self.goalData)
        return

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



if __name__ == '__main__':
    try:
        rospy.get_master().getPid()
    except:
        print("roscore is offline, exit")
        sys.exit(-1)

    CHECKPOINT = '/home/sam/proj/robot_software/social_navigation/src/navigan/models/benchmark_zara1_with_model.pt'

    rospy.init_node('navigan_local_planner')
    model = naviGan(torch.load(CHECKPOINT))
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('[{}] Shutting down...'.format(rospy.get_name()))
