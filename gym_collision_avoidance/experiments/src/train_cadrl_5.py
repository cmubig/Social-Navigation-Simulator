from argparse import ArgumentError
import os
from turtle import setworldcoordinates
import numpy as np
import gym
# import cv2
import matplotlib.pyplot as plt

gym.logger.set_level(40)
os.environ['GYM_CONFIG_CLASS'] = 'Train_5'
os.environ["global_timeout"]             = "1000"
os.environ["global_experiment_number"]   = "1"
os.environ["global_dataset_name"]        = "custom"
os.environ["global_population_density"]  = "0.7"
# os.environ['GYM_CONFIG_CLASS'] = 'Example'

# os.environ['GYM_CONFIG_PATH'] = '/home/mjana/ws/src/Social-Navigation-Simulator/gym_collision_avoidance/experiments/src/train_basic.py'

from gym_collision_avoidance.envs import test_cases as tc
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.policies.LearningCADRL.sim_utils.action import ActionRot, ActionXY

# from gym_collision_avoidance.envs.config import Config as EnvConfig
from gym_collision_avoidance.experiments.src.env_utils import create_env
from gym_collision_avoidance.envs.policies.LearningCADRL.sim_utils.state import JointState, ObservableState

def distance_penalty(start,goal,state):
    return np.abs((goal[0]-start[0])*(start[1]-state[1])-(goal[1]-start[1])*(start[0]-state[0]))/np.sqrt((goal[0]-start[0])**2+(goal[1]-start[1])**2)

def run_k_episodes(one_env, num_episodes, phase, agents, update_memory=False, episode=0):
    # Training environment parameters
    # num_episodes = 500
    agents[0].policy.policy.set_phase(phase)
    agents[0].policy.policy.time_step = Config.DT
    agents[0].policy.dt = Config.DT
    agents[0].policy.policy.set_env(one_env)
    init_pos_min_x = -5             # minimum initial position
    init_pos_min_y = -5
    init_pos_max_x = 5             # maximum initial position
    init_pos_max_y = 5
    distance_reward_factor = -0.005     # reward factor for deviating from straight line path

    # Learning agent parameters
    num_state_vector = 9+5*(len(agents)-1)
    num_steps = 150    

    next_state = np.zeros(num_state_vector, dtype=np.float32)
    curr_state = np.zeros(num_state_vector, dtype=np.float32)

    goal_count = 0
    coll_count = 0

    gx = 1
    # training metrics
    scores = []
    success_rate_list = []
    time_to_goal_list = []
    end = False

    for k in range(num_episodes):
        obs = one_env.reset() # Get agents' initial observations
        states = []
        acts = []
        rews = []
        for j in range(len(agents)):
            agents[j].reset(px=np.random.uniform(init_pos_min_x, init_pos_max_x),
                            py=np.random.uniform(init_pos_min_y, init_pos_max_y),
                            # heading=np.random.uniform(0, 2*np.pi),
                            gx = np.random.uniform(init_pos_min_x, init_pos_max_x), 
                            gy = np.random.uniform(init_pos_min_y, init_pos_max_y))
            agents[j].is_done = False
            agents[j].is_out_of_bounds = False
            agents[j].ran_out_of_time = False
        print(obs[0]['other_agents_states'])
        dist2goal = np.linalg.norm(np.array([agents[0].goal_global_frame[0], agents[0].goal_global_frame[1]]) - np.array([agents[0].pos_global_frame[0], agents[0].pos_global_frame[1]]))
        next_state[0] = agents[0].pos_global_frame[0]
        next_state[1] = agents[0].pos_global_frame[1]
        next_state[2] = agents[0].vel_global_frame[0]
        next_state[3] = agents[0].vel_global_frame[1]
        next_state[4] = agents[0].radius
        next_state[5] = agents[0].goal_global_frame[0]
        next_state[6] = agents[0].goal_global_frame[1]
        next_state[7] = agents[0].pref_speed
        next_state[8] = agents[0].heading_ego_frame
        for pp in range(len(agents)-1):
            for ll in range(5):
                next_state[9+ll+pp*5] = obs[0]['other_agents_states'][pp][ll]
        print("stored state: ",next_state)
        cumul_reward = 0
        print("agent, x,y,theta, theta_global: ", agents[0].pos_global_frame, agents[0].heading_ego_frame, agents[0].heading_global_frame)

        print("goal: ", agents[0].goal_global_frame)
        time_to_goal = num_steps
        start = agents[0].pos_global_frame
        goal = agents[0].goal_global_frame
        
        agent_dirn = [-4*np.pi/5, -2*np.pi/5, 0 , 2*np.pi/5, 4*np.pi/5]

        for i in range(num_steps):

            # Query the external agents' policies
            # e.g., actions[0] = external_policy(dict_obs[0])
            actions = {}
            print(":::::::::::::::::")
            print("\nepisode number", episode, "agents: ", len(agents))
            other_agent_state = []
            # Repeatedly send actions to the environment based on agents' observations
            for m in range(1,len(agents)):
                set_point = agent_dirn[m-1]
                P = 1.0
                int_err = 0.0
                err = 0.0

                p_err = err
                curr = agents[m].heading_global_frame
                err = set_point - agents[m].heading_global_frame
                control = np.clip(P*err, -1, 1)/2.0 + 0.5

                # control = -np.pi/24
                v_x = 0.5
                actions[m] = np.array([v_x, control])

                other_action = ActionRot(v_x,control)

                other_agent_state.append(agents[m].get_next_observable_state(other_action))
            

            agents[0].policy.update_next_lookahead(other_agent_state)

            curr_state = next_state.copy()
    
            ob_state = [agents[n+1].get_observable_state() for n in range(len(agents)-1)]

            state_obj = JointState(agents[0].get_full_state(),ob_state)

            # for q in range(len(agents)-1):
            #     print("print state", ob_state[q])

            # # TODO: get the RL action from the policy [argmax_a Q(s,a)]
            rl_action = agents[0].policy.act(state_obj)
            actions[0] = rl_action
            # actions[0] = ActionRot(v_x,control)
            
            # Run a simulation step (check for collisions, move sim agents)
            obs, rewards, game_over, which_agents_done = one_env.step(actions)
            # print("obs", obs)
            if agents[0].is_at_goal:
                rewards-= (i-dist2goal/agents[0].pref_speed)/num_steps
            states.append(agents[0].policy.policy.last_state)
            # acts.append(rl_action)
            acts.append(actions[0])
            rews.append(rewards)

            # rewards+= distance_reward_factor*distance_penalty(init_state[0], init_state[1], agents[0].pos_global_frame)
            
            cumul_reward += rewards
            print("done 0? ", agents[0].is_at_goal, agents[0].is_done, agents[0].ran_out_of_time, agents[0].is_out_of_bounds)
            print("done 1? ", agents[1].is_at_goal, agents[1].is_done, agents[1].ran_out_of_time, agents[1].is_out_of_bounds)
            print("which agents done: ",which_agents_done)
            
            next_state[0] = agents[0].pos_global_frame[0]
            next_state[1] = agents[0].pos_global_frame[1]
            next_state[2] = agents[0].vel_global_frame[0]
            next_state[3] = agents[0].vel_global_frame[1]
            next_state[4] = agents[0].radius
            next_state[5] = agents[0].goal_global_frame[0]
            next_state[6] = agents[0].goal_global_frame[1]
            next_state[7] = agents[0].pref_speed
            next_state[8] = agents[0].heading_ego_frame
            
            for pp in range(len(agents)-1):
                for ll in range(5):
                    next_state[9+ll+pp*5] = obs[0]['other_agents_states'][pp][ll]

            # cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                        #    * reward for t, reward in enumerate(rewards)]))

            print("action control", agents[0].policy.external_action_to_action(agents[0], actions[0]))
            # print("speed",agents[0].speed_global_frame)
            print("s", curr_state)
            print("a", actions[0])
            print("r", rewards)
            print("s'", next_state)
            print("cumul_reward", cumul_reward)
            print("at goal?:::::::", agents[0].is_at_goal)
            # print("eps = ", eps)
            print("sucess rate%", 100*goal_count/(k+1))
            
            # TODO: use obs and reward to train learning agent (add to experience replay and learn for 1 step)
            if update_memory:
                if agents[0].is_at_goal or agents[0].in_collision:
                    # only add positive(success) or negative(collision) experience in experience set
                    print("updating memory of size: ", len(states)-1)
                    agents[0].policy.update_memory(states, acts, rews)
            
            if(agents[0].is_at_goal):
                goal_count+=1
                print("Agent has reached goal")

                time_to_goal = i-dist2goal/agents[0].pref_speed
                break
            
            if(agents[0].in_collision):
                coll_count+=1
                print("Agent has collided")
                break


            if agents[0].is_done:
                print("agent stopped working")
                end = True
                # break
            # if game_over:
                # print("All agents finished!")
                # break
        # if end:
            # break
    success_rate_list.append(100*goal_count/(k+1))
    time_to_goal_list.append(time_to_goal)
    scores.append(cumul_reward)
    
    return goal_count, time_to_goal_list, scores, coll_count
        # if phase in ['val', 'test']:
            # num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
            # logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                        #  too_close / num_step, average(min_dist))

        # if print_failure:
            # logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            # logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

def main():
    '''
    Minimum working example:
    2 agents: 1 training a DQN, 1 running external policy
    '''

    env, one_env = create_env()
    # In case you want to save plots, choose the directory
    one_env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__)) + '/../../experiments/results/train/')

    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = tc.get_testcase_one_train_5()
    num_actions = 6
    num_state_vector = 8

    init_state = []

    for i in range(len(agents)):
        state = []
        state.append(agents[i].to_vector()[0][1])
        state.append(agents[i].to_vector()[0][2])
        state.append(agents[i].to_vector()[0][10])
        init_state.append(state)
    
    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    one_env.set_agents(agents)
    
    agents[0].policy.init_network(num_actions, num_state_vector)

    train_episodes = 100
    val_size = 100

    eps_start = 0.5                # exploration probability at start
    eps_end = 0.1                   # exploration probability at end
    eps_dec = 1000                 # exploration probability decay factor
    eps = eps_start                 # exploration probability

    # explorer

    agents[0].policy.policy.set_epsilon(0.6)
    run_k_episodes(one_env, 2, 'train', agents, update_memory=True, episode=0)
    # env.reset()
    episode = 0 
    total_goals = 0
    total_colls = 0
    num_episodes = 0
    scores = []
    time_to_goal_list = []
    success_rate_list = []
    while episode < agents[0].policy.train_episodes:
        agents[0].policy.policy.set_epsilon(eps)
        # if episode % agents[0].policy.evaluation_interval == 0:
                # run_k_episodes(one_env, val_size, 'val', agents, episode=episode)
        goals, time_to_goal, score_list, colls = run_k_episodes(one_env,agents[0].policy.sample_episodes, 'train', agents, update_memory=True, episode=episode)
        total_goals+=goals
        total_colls+=colls
        num_episodes+=agents[0].policy.sample_episodes
        for score in score_list:
            scores.append(score)
        for time in time_to_goal:
            time_to_goal_list.append(time)
        print("BACK PROPAGAAAAAAAATTTIIIIOOOONNNNNNNN ! LET'S GOOOOOOOOOOOOOOOOOO!!!")
        agents[0].policy.trainer.optimize_batch(agents[0].policy.train_batches)
        # env.reset()
        # one_env.reset()
        episode += 1
        print("ep no" ,episode)

        eps = max(eps_end, eps_start + (eps_end - eps_start) / eps_dec * episode)
    
        if episode % agents[0].policy.target_update_interval == 0:
            agents[0].policy.update_target_model()

        if episode != 0 and episode % agents[0].policy.checkpoint_interval == 0:
            agents[0].policy.save_checkpoint('trained_checkpoint')
    agents[0].policy.save_checkpoint('trained_checkpoint')
    plt.figure(figsize=(12,8))
    plt.plot(range(num_episodes), scores)
    plt.xlim(-1, num_episodes+1)
    plt.ylim(-2,3)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward vs Episode')
    plt.show()
    plt.savefig('reward_plot.png')

    print("Test: Success Rate and Collisions: ",total_goals, total_colls)

    plt.figure(figsize=(12,8))
    plt.plot(range(num_episodes), time_to_goal_list)
    plt.xlim(-1, num_episodes+1)
    plt.ylim(0,255)
    plt.xlabel('Episode')
    plt.ylabel('Time to goal')
    plt.title('Time to goal vs Episode')
    plt.show()
    plt.savefig('time_to_goal.png')

    total_goals = 0
    total_colls = 0
    num_episodes = 100
    scores = []
    time_to_goal_list = []
    success_rate_list = []

    goals, time_to_goal, score_list, colls = run_k_episodes(one_env, 100, 'test', agents, episode=episode)
    for score in score_list:
        scores.append(score)
    for time in time_to_goal:
        time_to_goal_list.append(time)
    success_rate_list.append(100*goals/(num_episodes+1))
    
    plt.figure(figsize=(12,8))
    plt.plot(range(num_episodes), scores)
    plt.xlim(-1, num_episodes+1)
    plt.ylim(-2,3)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Test: Cumulative Reward vs Episode')
    plt.savefig('reward_plot_test.png')
    plt.show()
    print("Test: Success Rate and Collisions: ",total_goals, total_colls)

    plt.figure(figsize=(12,8))
    plt.plot(range(num_episodes), time_to_goal_list)
    plt.xlim(-1, num_episodes+1)
    plt.ylim(0,255)
    plt.xlabel('Episode')
    plt.ylabel('Time to goal')
    plt.title('Test: Time to goal vs Episode')
    plt.show()
    plt.savefig('time_plot_test.png')

    # plt.show()

    return True

if __name__ == '__main__':
    main()
    print("Experiment over.")