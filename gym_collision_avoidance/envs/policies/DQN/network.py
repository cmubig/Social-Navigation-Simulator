import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys

# device=T.device("cuda:0")
device=T.device("cpu")
random.seed(42)
np.random.seed(42)
# T.manual_seed(0)

class Actions():
    # Define 11 choices of actions to be:
    # [v_pref,      [-pi/6, -pi/12, 0, pi/12, pi/6]]
    # [0.5*v_pref,  [-pi/6, 0, pi/6]]
    # [0,           [-pi/6, 0, pi/6]]
    def __init__(self):
        self.actions = np.mgrid[1.0:1.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/12].reshape(2, -1).T
        self.actions = np.vstack([self.actions,np.mgrid[0.5:0.6:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.actions = np.vstack([self.actions,np.mgrid[0.0:0.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.num_actions = len(self.actions)

# The Pytorch nn.Module Class for the DQNetwork model
class PatNet(nn.Module):
    ## HYPERPARAMTERS
    def __init__(self, nA, nS):
        super(PatNet,self).__init__()
        self.nA=nA
        self.input_dims=nS
        self.fc1=nn.Linear(self.input_dims, 128)    # 1st Hidden Layer
        self.fc2=nn.Linear(128, 84)                # 2nd Hidden Layer
        self.fc3=nn.Linear(84, self.nA)            # Output Layer
        # self.dropout=nn.Dropout(p=0.2)              # Dropout

    def forward(self, observation):
        observation=F.relu(self.fc1(observation))
        observation=F.relu(self.fc2(observation))
        qsa=self.fc3(observation)

        return qsa

#end of network class

class Agent():
    # HYPERPARAMTERS
    def __init__(self, nA, nS):
        self.GAMMA      = 0.95      # Discounted Return
        self.LEARN_PER  = 1         # Q-policy update time interval
        self.UPD_PER    = 2         # Fixed Q-target update time interval
        self.TAU        = 0.001     # Soft update of target params

        self.mem_size   = 10000    # Replay buffer size
        self.batch_size = 32       # Training batch size
        self.lr         = 0.00075   # opimizer learning rate
        self.momentum   = 0.9       # SGD momentum
        self.exp_gamma  = 0.96      # lr scheduler exp decay rate

        self.action_space = [i for i in range(nA)]
        self.n_actions = nA
        self.state_size = nS
        print("agent action space:", self.action_space)
        # The Q-Networks
        self.Q_policy=PatNet(self.n_actions, self.state_size).to(device)
        self.Q_target=PatNet(self.n_actions, self.state_size).to(device)

        print("Q Networks initialized")
        # Model functions
        # self.optimizer=optim.SGD(self.Q_policy.parameters(), lr=self.lr, momentum=self.momentum)
        self.optimizer=optim.Adam(self.Q_policy.parameters(), lr=self.lr)
        # self.lr_sched = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.exp_gamma)
        self.criterion=nn.MSELoss()

        # Experience replay buffer
        self.memory = Exp_Replay(self.n_actions, self.mem_size, self.batch_size)
        self.t_step = 1

    def step(self, state, action, reward, next_state):
        # One-hot encode actions and save it in replay memory
        self.memory.add(state, action, reward, next_state)

        # Update weights of the Q-networks
        if (self.t_step%self.LEARN_PER)==0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

        self.t_step+=1

    def eps_greedy(self, state, nA, eps=0.):

        # Get the Q(s,a) value predictions from the policy network
        state = T.from_numpy(state).float().unsqueeze(0).to(device)
        self.Q_policy.eval()
        with T.no_grad():
            action_values=self.Q_policy(state)
        self.Q_policy.train()
        action_values=action_values.cpu().data.numpy()
        # print(action_values)
        # action_values=[action_values.reshape(-1)[i] for i in acts]
        # Select action based on exploration-exploitation epsilon-greddy strategy
        print(action_values)
        if random.random() > eps:
            max_acts = np.argwhere(action_values == np.amax(action_values))

            return np.random.choice([i for i in max_acts.reshape(-1)])
        else:
            print("random action !! eps=", eps)
            return np.random.randint(0,nA)

    def learn(self,experiences, gamma):
        # Get input parameters from sampled experience replay
        states, actions, rewards, next_states = experiences

        # Get the next_state target Q-values from target network
        # Calculate current_state target Q-values using Bellman Optimality Equation
        print(next_states.shape)
        Q_targets_next= self.Q_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma*Q_targets_next)

        # Get predicted current state Q-values from policy network
        Q_expects=self.Q_policy(states).gather(1,actions)

        # Compute and minimize the loss and update weights of policy network
        loss = self.criterion(Q_expects,Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Fixed Q-targets learning
        # Update weights of target network after a certain time-step
        if (self.t_step%self.UPD_PER)==0:
            # lr_sched.step()
            self.update(self.Q_policy, self.Q_target, self.TAU)

    def update(self, policy_net, target_net, tau):
        # Copy weights from learned policy network to the target network
        for target_weights, policy_weights in zip(target_net.parameters(), policy_net.parameters()):
            target_weights.data.copy_(tau*policy_weights.data+(1-tau)*target_weights.data)
            # target_weights.data.copy_(policy_weights.data)

#end of agent class

class Exp_Replay():

    def __init__(self, nA, nBuff, batch_size):

        # Initialize the memory deque and experience tuple
        self.nA=nA
        self.memory=deque(maxlen=nBuff)
        self.batch_size=batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])

    def add(self, state, action, reward, next_state):
        # Append (s,a,r,s') tuple to the memory deque
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        # Randomly sample a batch from memory of size = batch_size
        experiences= random.sample(self.memory, k=self.batch_size)

        # Store the sampled data in to a tuple
        states=T.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions=T.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards=T.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states=T.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states)

    def __len__(self):
        # Get length of stored memory deque
        return len(self.memory)

#end of replay class

def get_max_act(acts, Q):
    Q=[Q[i] for i in acts]
    act_max=action[np.argmax(Q)]
#end of fn

def CR_patrol(idle, c, env, an):

    neigh=[idle[i] for i in an]
    print(neigh)
    m = max(neigh)
    idx= [i for i, j in enumerate(neigh) if j == m]
    print('idx: ', idx)
    act_idx=random.choice(idx)
    action_node=an[act_idx]
    print('cur and next: ', c, action_node)
    if c==action_node:
            act_idx=CR_patrol(idle,c,env, an)
    return act_idx
#end of fn

# def train_agent(env, agent, n_ep=500, max_t=1000, eps_start=1.0, eps_end=0.01, eps_dec=0.995):
#     scores = []
#     scores_window=deque(maxlen=50)
#     eps=eps_start
#     for i in range(0,n_ep):
#         state,rou_curr=env.reset(i)
#         sumo_step=1.0
#         # bn=[0, 17, 30, 31, 32, 33]
#         cr=[0.0, 0.0]
#         rl_step=[0,0]
#         data=[[[0. ,0., 0., 0.] for _ in range(3)],[[0. ,0., 0., 0.] for _ in range(3)]]
#         #idle=[np.zeros((25,1)), np.zeros((25,1))]
#         global_idl=np.zeros((25,1))
#         global_v_idl=[[] for _ in range(25)]
#         #v_idle=[[[] for _ in range(25)], [[] for _ in range(25)]]
#         edge=[0,0]
#         prev_node=env.node
#         curr_node=[0.,0.]
#         temp_n=[0.,0.]
#         temp_p=[0,24]
#         ga=[]
#         gav=[]
#         ss=[]
#         while traci.simulation.getMinExpectedNumber()>0:
#             traci.simulationStep()
#             # idle+=1
#             edge[0]=traci.vehicle.getRoadID('veh'+str(2*i))
#             edge[1]=traci.vehicle.getRoadID('veh'+str(2*i+1))
#             if (edge[0] and edge[1]):
#                 global_idl+=1
#                 # idle[0]+=1
#                 # idle[1]+=1
#                 # print('veh edge data: ',ed)
#                 for j, ed in enumerate(edge):
#                     if ed and (ed[0]!=':'):
#                         curr_node[j]= ed.split('to')
#                         curr_node[j]=int(curr_node[j][1])
#                     elif ed[0]==':':
#                         curr_node[j]=ed[1:].split('_')
#                         curr_node[j]=int(curr_node[j][0])
#                 env.node=curr_node.copy()
#                 #print('p_node:',prev_node, 'c_node:',curr_node, 'temp_p: ', temp_p, 'temp_n: ', temp_n)
#                 for j in range(len(agent)):
#                     # Action decision on new edge
#                     if prev_node[j]!=curr_node[j]:
#                         temp_p[j]=prev_node[j]
#                         print(':::::::::::::to next node::::::::::::::::')

#                         #print('Veh angle: ', traci.vehicle.getAngle('veh'+str(i)))
#                         rou_step=[]
#                         print(global_idl[prev_node])
#                         #prev_reward=env.reward_out(global_idl, prev_node)[0].copy()
#                         prev_reward=env.reward_out(global_idl, prev_node[j], j)[0].copy()
#                         global_v_idl[int(prev_node[j])].append(prev_reward)
#                         #glo_reward=env.reward_out(global_idl, prev_node[j],j)[0]
#                         # global_v_idl[int(prev_node[j])].append(glo_reward.copy())
#                         # print(global_v_idl)
#                         global_idl[int(prev_node[j])]=0
#                         # idle[j][int(prev_node[j])]=0
#                         avg_v_idl, max_v_idl, sd_v_idl, glo_v_idl, glo_max_v_idl, glo_sd_v_idl, glo_idl, glo_max_idl = eval_met(global_idl, global_v_idl,sumo_step, 25)
#                         print('global avg node visit idleness: ', glo_v_idl, '\nglobal max node visit idleness: ', glo_max_v_idl)
#                         print('global avg instant idleness: ', glo_idl, '\nglobal max instant idleness: ', glo_max_idl)
#                         #print(np.array(v_idle).reshape(5,5))
#                         gav.append(glo_v_idl)
#                         ga.append(glo_idl)
#                         ss.append(sumo_step)

#                         print('global idleness:\n',global_idl.reshape(5,5))

#                         links = traci.lane.getLinks(traci.vehicle.getLaneID('veh'+str(2*i+j)), extended=False)
#                         s_lanes = [j[0] for j in links]
#                         n_edges=[]
#                         n_nodes=[]
#                         for nodes in s_lanes:
#                             n_edges.append(nodes.split('_')[0])
#                         for edges in n_edges:
#                             n_nodes.append(int(edges.split('to')[1]))
#                         print(n_edges, n_nodes)

#                         #acts, space = env.get_state_action(np.array(prev_node), np.array(curr_node), np.array(n_nodes),np.array(global_idl))
#                         acts, space = env.get_state_action(np.array(prev_node[j]), np.array(curr_node[j]), np.array(n_nodes),np.array(global_idl))
#                         print("look here: ", acts, "and here: " ,space)

#                         # max_act = get_max_act(acts, Q)

#                         env.set_actionSpace(n_edges)
#                         #return index while sampling, get act[index] for exp replay
#                         c_state=np.array([prev_node[j], curr_node[j]]+list(space), dtype=np.float32)
#                         # print(c_state)

#                         # act_idx=CR_patrol(idle,curr_node,env, n_nodes)
#                         # act_idx = env.sample(n_nodes)
#                         act_id=agent[j].eps_greedy(c_state,acts, eps)
#                         print(act_id)
#                         act_idx=np.argwhere(acts==act_id).reshape(-1)[0]
#                         # print(act_idx)
#                         action = n_edges[act_idx]
#                         #next_state, reward= env.step(n_nodes[act_idx], global_idl)
#                         next_state, reward= env.step(n_nodes[act_idx], global_idl,j)
#                         temp_n[j]=next_state

#                         print('reward on prev step: ', prev_reward)
#                         data[j][rl_step[j]%3][0]=c_state
#                         data[j][rl_step[j]%3][1]=act_id
#                         data[j][(rl_step[j]-1)%3][3]=c_state
#                         data[j][(rl_step[j]-2)%3][2]=(prev_reward-21)*((abs(prev_reward-21))**0.5)/glo_idl
#                         # data[j][(rl_step[j]-2)%3][2]=prev_reward-22

#                         # print("delayed replay: ",data)
#                         ##Experience saved as [s,a,r',s']
#                         exp=data[j][(rl_step[j]-2)%3]
#                         print("\nexperience of agent ",j,": " , exp)
#                         if(rl_step[j]>1):
#                             agent[j].step(exp[0],exp[1], exp[2], exp[3])
#                         print(len(agent[j].memory))

#                         print('action: ', action, 'next_state: ', next_state, 'reward: ', reward)
#                         #print('curr_node after step: ',curr_node, env.state)
#                         rou_new=action
#                         rou_step.append(rou_curr[j])
#                         rou_step.append(rou_new)
#                         print('next_route: ', rou_step)
#                         traci.vehicle.setRoute(vehID = 'veh'+str(2*i+j), edgeList = rou_step)
#                         rou_curr[j]=rou_new
#                         rl_step[j]+=1

#                 prev_node=curr_node.copy()
#                 #print('curr route: ',rou_curr)
#                 sumo_step+=1
#                 if sumo_step>=max_t:
#                     break


#         eps=max(eps_end,eps_dec*eps)
#         if i % 100 == 0:
#             plt.plot(ss,ga, "-r", linewidth=0.6,label="Global Average Idleness")
#             plt.plot(ss,gav, "-b", linewidth=4, label="Global Average Node Visit Idleness")
#             plt.legend(loc="lower right")
#             up=np.ceil(max(ga)/10)*10
#             plt.yticks(np.linspace(0,up,(up/10)+1, endpoint=True))
#             plt.xlabel('Unit Time')
#             plt.ylabel('Idleness')
#             plt.title('Performance')
#             plt.show()
#         scores_window.append(ga[-1])
#         scores.append(ga[-1])
#     T.save(agent[0].Q_policy.state_dict(), 'pat_dqn_5_agt_1.pth')
#     T.save(agent[1].Q_policy.state_dict(), 'pat_dqn_5_agt_2.pth')
#     return scores
# #end of fn

# if __name__ == '__main__':
#     # rou_curr=["0to"+str(random.choice([1,5]))]
#     # rou_curr=["0to5"]
#     # traci.route.add('rou_0', rou_curr)
#     # traci.vehicle.add(vehID = 'veh0',routeID = 'rou_0', typeID = "car1")
#     agents = [Agent(11,7)]
#     # HYPERPARAMTERS
#     n_eps    =  500            # no. of episodes to train for
#     max_t    =  6000           # max time steps per episode
#     eps_start=  0.90           # initial exploration probability
#     eps_end  =  0.01           # final exploration probability
#     eps_dec  =  0.991           # decrement of epsilon

#     scores = train_agent(env, agents, n_eps, max_t, eps_start, eps_end, eps_dec)

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.plot(np.arange(len(scores)), scores)
#     plt.ylabel('Global Instant Idleness')
#     plt.xlabel('Episode #')
#     plt.show()


