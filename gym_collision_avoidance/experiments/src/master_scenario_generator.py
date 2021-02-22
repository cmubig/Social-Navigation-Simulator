import numpy as np
import random
import os
#sg = Scenario_Generator(10, "RVO", -10, 10 ,-10, 10, 0.5, 0.05, 0)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class Scenario_Generator(object):
    def __init__(self, num_agents, policy_list, x_min, x_max, y_min, y_max, pref_speed, agent_radius, start_timestamp, num_agents_stddev=0, pref_speed_stddev=0):

        self.num_agents       = int(np.round( np.random.normal(num_agents, num_agents_stddev) ))
        if self.num_agents <=1: self.num_agents=2
        self.policy_list      = policy_list
        self.x_min            = x_min
        self.x_max            = x_max
        self.y_min            = y_min
        self.y_max            = y_max
        self.pref_speed       = pref_speed
        self.agent_radius     = agent_radius
        self.start_timestamp  = start_timestamp

        self.pref_speed_stddev = pref_speed_stddev
        
        pass

    def random_square(self):

        scenario = []

        for i in range(self.num_agents):
            if type(self.policy_list)==list:
                policy = self.policy_list[i]
            else:
                policy = self.policy_list

            if type(self.start_timestamp)==list:
                start_timestamp = self.start_timestamp[i]
            else:
                start_timestamp = self.start_timestamp

            if type(self.pref_speed)==list:
                pref_speed = np.random.normal(self.pref_speed[i], self.pref_speed_stddev)
            else:
                pref_speed = np.random.normal(self.pref_speed, self.pref_speed_stddev)

            start_x = round(np.random.uniform(self.x_min, self.x_max), 1)  #1 decimal place
            start_y = round(np.random.uniform(self.x_min, self.x_max), 1)  #1 decimal place

            goal_x  = round(np.random.uniform(self.y_min, self.y_max), 1)  #1 decimal place
            goal_y  = round(np.random.uniform(self.y_min, self.y_max), 1)  #1 decimal place

            agent_radius = self.agent_radius
            

            scenario.append( [ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ] )

        return scenario

    def random_square_edge(self):
        scenario = []

        for i in range(self.num_agents):
            if type(self.policy_list)==list:
                policy = self.policy_list[i]
            else:
                policy = self.policy_list

            if type(self.start_timestamp)==list:
                start_timestamp = self.start_timestamp[i]
            else:
                start_timestamp = self.start_timestamp

            if type(self.pref_speed)==list:
                pref_speed = np.random.normal(self.pref_speed[i], self.pref_speed_stddev)
            else:
                pref_speed = np.random.normal(self.pref_speed, self.pref_speed_stddev)

            if np.random.choice([True,False]):
                # along x axis (left right edge)
                start_x = round(np.random.uniform(self.x_min, self.x_max), 1)
                start_y = np.random.choice([self.y_min, self.y_max])
            else:
                # along y axis (top bottom edge)
                start_x = np.random.choice([self.x_min, self.x_max])
                start_y = round(np.random.uniform(self.y_min, self.y_max), 1)       

            if np.random.choice([True,False]):
                # along x axis (left right edge)
                goal_x = round(np.random.uniform(self.x_min, self.x_max), 1)

                #prevent any goal on same side
                choice_list = [self.y_min, self.y_max]
                if start_y==self.y_min or start_y==self.y_max:
                    choice_list.remove(start_y)
 
                goal_y = np.random.choice(choice_list)
            else:
                # along y axis (top bottom edge)
                goal_y = round(np.random.uniform(self.y_min, self.y_max), 1)

                #prevent any goal on same side
                choice_list = [self.x_min, self.x_max]
                if start_x==self.x_min or start_x==self.x_max:
                    choice_list.remove(start_x)
 
                goal_x = np.random.choice(choice_list)

            agent_radius = self.agent_radius
            
            #print([ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ])
            scenario.append( [ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ] )

        return scenario
                
######################################
class Seeded_Scenario_Generator(object):
    def __init__(self, num_agents, policy_list, x_min, x_max, y_min, y_max, pref_speed, agent_radius, start_timestamp, num_agents_stddev=0, pref_speed_stddev=0, random_seed=0 , num_agents_override=None):
        self.random_seed      = random_seed*1000
        np.random.seed(self.random_seed)

        if num_agents_override is None:
            self.num_agents       = int(np.round( np.random.normal(num_agents, num_agents_stddev) ))
            if self.num_agents <=1: self.num_agents=2
        else:
            self.num_agents       = num_agents_override
            
        self.policy_list      = policy_list
        self.x_min            = x_min
        self.x_max            = x_max
        self.y_min            = y_min
        self.y_max            = y_max
        self.pref_speed       = pref_speed
        self.agent_radius     = agent_radius
        self.start_timestamp  = start_timestamp

        self.pref_speed_stddev = pref_speed_stddev
        
        pass

    def seed(self):
        self.random_seed+=1
        np.random.seed(self.random_seed)

    def random_square(self):

        scenario = []

        for i in range(self.num_agents):
            if type(self.policy_list)==list:
                policy = self.policy_list[i]
            else:
                policy = self.policy_list

            if type(self.start_timestamp)==list:
                start_timestamp = self.start_timestamp[i]
            else:
                start_timestamp = self.start_timestamp

            self.seed()
            if type(self.pref_speed)==list:
                pref_speed = np.random.normal(self.pref_speed[i], self.pref_speed_stddev)
            else:
                pref_speed = np.random.normal(self.pref_speed, self.pref_speed_stddev)

            self.seed()
            start_x = round(np.random.uniform(self.x_min, self.x_max), 1)  #1 decimal place
            self.seed()
            start_y = round(np.random.uniform(self.x_min, self.x_max), 1)  #1 decimal place

            self.seed()
            goal_x  = round(np.random.uniform(self.y_min, self.y_max), 1)  #1 decimal place
            self.seed()
            goal_y  = round(np.random.uniform(self.y_min, self.y_max), 1)  #1 decimal place

            agent_radius = self.agent_radius
            

            scenario.append( [ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ] )

        return scenario

    def random_square_edge(self):
        scenario = []

        for i in range(self.num_agents):

            #modify the random seed for each agent
            
            if type(self.policy_list)==list:
                policy = self.policy_list[i]
            else:
                policy = self.policy_list

            if type(self.start_timestamp)==list:
                start_timestamp = self.start_timestamp[i]
            else:
                start_timestamp = self.start_timestamp

            self.seed()

            # speed_rescaler = 4.5

            if type(self.pref_speed)==list:
                pref_speed = np.random.normal(self.pref_speed[i], self.pref_speed_stddev) #* speed_rescaler
            else:
                pref_speed = np.random.normal(self.pref_speed, self.pref_speed_stddev) #* speed_rescaler

            #make sure speed is not too slow
            speed_threshold = 0.2
            if (pref_speed < speed_threshold ): pref_speed = speed_threshold

            #To make sure it is not colliding with other start point
            collide = True
            while collide:
                self.seed()
                if np.random.choice([True,False]):
                    # along x axis (left right edge)
                    self.seed()
                    start_x = round(np.random.uniform(self.x_min, self.x_max), 1)
                    self.seed()
                    start_y = np.random.choice([self.y_min, self.y_max])
                else:
                    # along y axis (top bottom edge)
                    self.seed()
                    start_x = np.random.choice([self.x_min, self.x_max])
                    self.seed()
                    start_y = round(np.random.uniform(self.y_min, self.y_max), 1)

                if len(scenario)==0:
                    collide=False
                    break

                past_start_points = np.array(scenario)[:,2:4].astype(np.float)
                closet_distance_to_other_start_point = 999
                for past_start_point in past_start_points:
                    distance = np.linalg.norm(past_start_point - np.array([start_x,start_y]) )
                    if distance < closet_distance_to_other_start_point: closet_distance_to_other_start_point = distance

                if closet_distance_to_other_start_point >= 0.7: #default=1, reduce if it is a crowded scene
                    collide=False
                    break

            #To make sure it is not colliding with other goal point
            collide = True
            while collide:
                self.seed()
                if np.random.choice([True,False]):
                    # along x axis (left right edge)
                    self.seed()
                    goal_x = round(np.random.uniform(self.x_min, self.x_max), 1)

                    #prevent any goal on same side
                    choice_list = [self.y_min, self.y_max]
                    if start_y==self.y_min or start_y==self.y_max:
                        choice_list.remove(start_y)

                    self.seed()
                    goal_y = np.random.choice(choice_list)
                else:
                    # along y axis (top bottom edge)
                    self.seed()
                    goal_y = round(np.random.uniform(self.y_min, self.y_max), 1)

                    #prevent any goal on same side
                    choice_list = [self.x_min, self.x_max]
                    if start_x==self.x_min or start_x==self.x_max:
                        choice_list.remove(start_x)

                    self.seed()
                    goal_x = np.random.choice(choice_list)

                if len(scenario)==0:
                    collide=False
                    break

                past_goal_points = np.array(scenario)[:,4:6].astype(np.float)
                closet_distance_to_other_goal_point = 999
                for past_goal_point in past_goal_points:
                    distance = np.linalg.norm(past_goal_point - np.array([goal_x,goal_y]) )
                    if distance < closet_distance_to_other_goal_point: closet_distance_to_other_goal_point = distance

                if closet_distance_to_other_goal_point >= 0.7: #default=1, reduce if it is a crowded scene
                    collide=False
                    break

            agent_radius = self.agent_radius
            
            #print([ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ])
            scenario.append( [ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ] )

        return scenario

##################################################################
class Seeded_Population_Scenario_Generator(object):
    def __init__(self, population_density, policy_list, x_min, x_max, y_min, y_max, pref_speed, agent_radius, start_timestamp, random_seed=0, num_agents_override=None):
        self.random_seed      = random_seed*1000
        np.random.seed(self.random_seed)

        if num_agents_override is None:
            self.num_agents       = int(round(population_density * ( ( x_max - x_min )  *  ( y_max - y_min )  )))
            print(str(self.num_agents)+" agents created under population density of "+str(population_density))
            if self.num_agents <=1: self.num_agents=2
            #override
            if self.num_agents <=2: self.num_agents=3
        else:
            self.num_agents = num_agent_override
            
        self.policy_list      = policy_list
        self.x_min            = x_min
        self.x_max            = x_max
        self.y_min            = y_min
        self.y_max            = y_max
        self.pref_speed       = pref_speed
        self.agent_radius     = agent_radius
        self.start_timestamp  = start_timestamp

        self.border_relaxer = 1
        self.spawn_distance_threshold = 0.7
        
        pass

    def seed(self):
        self.random_seed+=1
        np.random.seed(self.random_seed)

    def population_random_square_edge(self):
        scenario = []

        for i in range(self.num_agents):

            #modify the random seed for each agent
            
            if type(self.policy_list)==list:
                policy = self.policy_list[i]
            else:
                policy = self.policy_list

            if type(self.start_timestamp)==list:
                start_timestamp = self.start_timestamp[i]
            else:
                start_timestamp = self.start_timestamp

            self.seed()

            # speed_rescaler = 4.5

            if type(self.pref_speed)==list:
                pref_speed = self.pref_speed[i] #* speed_rescaler
            else:
                pref_speed = self.pref_speed #* speed_rescaler

            #make sure speed is not too slow
            speed_threshold = 0.2
            if (pref_speed < speed_threshold ): pref_speed = speed_threshold

            #To make sure it is not colliding with other start point
            collide = True
            while collide:
                self.seed()
                if np.random.choice([True,False]):
                    # along x axis (left right edge)
                    self.seed()
                    start_x = round(np.random.uniform(self.x_min, self.x_max), 1)

                    self.seed()
                    start_y = np.random.choice([self.y_min, self.y_max])

                    # more space for agents in same axis (give extra room. not fixing them in line)
                    self.seed()
                    if start_y == self.y_min: start_y += np.random.random() * self.border_relaxer
                    if start_y == self.y_max: start_y -= np.random.random() * self.border_relaxer

                else:
                    # along y axis (top bottom edge)
                    self.seed()
                    start_x = np.random.choice([self.x_min, self.x_max])

                    # more space for agents in same axis (give extra room. not fixing them in line)
                    self.seed()
                    if start_x == self.x_min: start_x += np.random.random() * self.border_relaxer
                    if start_x == self.x_max: start_x -= np.random.random() * self.border_relaxer

                    self.seed()
                    start_y = round(np.random.uniform(self.y_min, self.y_max), 1)

                if len(scenario)==0:
                    collide=False
                    break

                past_start_points = np.array(scenario)[:,2:4].astype(np.float)
                closet_distance_to_other_start_point = 999
                for past_start_point in past_start_points:
                    distance = np.linalg.norm(past_start_point - np.array([start_x,start_y]) )
                    if distance < closet_distance_to_other_start_point: closet_distance_to_other_start_point = distance

                if closet_distance_to_other_start_point >= self.spawn_distance_threshold: #0.7 #default=1, reduce if it is a crowded scene
                    collide=False
                    break

            #To make sure it is not colliding with other goal point
            collide = True
            while collide:
                self.seed()
                if np.random.choice([True,False]):
                    # along x axis (left right edge)
                    self.seed()
                    goal_x = round(np.random.uniform(self.x_min, self.x_max), 1)

                    #prevent any goal on same side
                    choice_list = np.array([self.y_min, self.y_max])

                    #if start and goal is on the same axis, set goal to opposite side of start
                    if (abs(start_y-self.y_min)<=1) or (abs(start_y-self.y_max)<=1):
                        duplicate_choice = choice_list[ int(np.where( np.abs(choice_list- start_y )<=1 )[0][0]) ]
                        choice_list = list(choice_list)
                        choice_list.remove(duplicate_choice)

                    self.seed()
                    goal_y = np.random.choice(choice_list)

                    # more space for goals in same axis (give extra room. not fixing them in line)
                    self.seed()
                    if goal_y == self.y_min: goal_y += np.random.random()
                    if goal_y == self.y_max: goal_y -= np.random.random()
                else:
                    # along y axis (top bottom edge)
                    self.seed()
                    goal_y = round(np.random.uniform(self.y_min, self.y_max), 1)

                    #prevent any goal on same side
                    choice_list = np.array([self.x_min, self.x_max])

                     #if start and goal is on the same axis, set goal to opposite side of start
                    if (abs(start_x-self.x_min)<=1) or (abs(start_x-self.x_max)<=1):
                        duplicate_choice = choice_list[ int(np.where( abs(choice_list- start_x )<=1 )[0][0]) ]
                        choice_list = list(choice_list)
                        choice_list.remove(duplicate_choice)

                    self.seed()
                    goal_x = np.random.choice(choice_list)

                    # more space for goals in same axis (give extra room. not fixing them in line)
                    self.seed()
                    if goal_x == self.x_min: goal_x += np.random.random()
                    if goal_x == self.x_max: goal_x -= np.random.random()

                if len(scenario)==0:
                    collide=False
                    break

                past_goal_points = np.array(scenario)[:,4:6].astype(np.float)
                closet_distance_to_other_goal_point = 999
                for past_goal_point in past_goal_points:
                    distance = np.linalg.norm(past_goal_point - np.array([goal_x,goal_y]) )
                    if distance < closet_distance_to_other_goal_point: closet_distance_to_other_goal_point = distance

                if closet_distance_to_other_goal_point >= self.spawn_distance_threshold: #0.7 #default=1, reduce if it is a crowded scene
                    collide=False
                    break

            agent_radius = self.agent_radius
            
            #print([ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ])
            scenario.append( [ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ] )

        return scenario
       
###### Single agent scenario return for respawn at new position
class Single_Seeded_Population_Scenario_Generator(object):
    def __init__(self, population_density, policy_list, x_min, x_max, y_min, y_max, pref_speed, agent_radius, start_timestamp, agents, random_seed=0, num_agents_override=None):
        self.random_seed      = random_seed*1000
        np.random.seed(self.random_seed)

        if num_agents_override is None:
            self.num_agents       = int(round(population_density * ( ( x_max - x_min )  *  ( y_max - y_min )  )))
            print(str(self.num_agents)+" agents created under population density of "+str(population_density))
            if self.num_agents <=1: self.num_agents=2
            #override
            if self.num_agents <=2: self.num_agents=3
        else:
            self.num_agents = num_agents_override
            
        self.policy_list      = policy_list
        self.x_min            = x_min
        self.x_max            = x_max
        self.y_min            = y_min
        self.y_max            = y_max
        self.pref_speed       = pref_speed
        self.agent_radius     = agent_radius
        self.start_timestamp  = start_timestamp

        self.border_relaxer = 1
        self.spawn_distance_threshold = 1.5 #0.7
        
        pass

    def seed(self):
        self.random_seed+=1
        np.random.seed(self.random_seed)

    def population_random_square_edge(self):
        scenario = []

        for i in range(self.num_agents):

            #modify the random seed for each agent
            
            if type(self.policy_list)==list:
                policy = self.policy_list[i]
            else:
                policy = self.policy_list

            if type(self.start_timestamp)==list:
                start_timestamp = self.start_timestamp[i]
            else:
                start_timestamp = self.start_timestamp

            self.seed()

            # speed_rescaler = 4.5

            if type(self.pref_speed)==list:
                pref_speed = self.pref_speed[i] #* speed_rescaler
            else:
                pref_speed = self.pref_speed #* speed_rescaler

            #make sure speed is not too slow
            speed_threshold = 0.2
            if (pref_speed < speed_threshold ): pref_speed = speed_threshold

            #To make sure it is not colliding with other start point
            collide = True
            while collide:
                self.seed()
                if np.random.choice([True,False]):
                    # along x axis (left right edge)
                    self.seed()
                    start_x = round(np.random.uniform(self.x_min, self.x_max), 1)

                    self.seed()
                    start_y = np.random.choice([self.y_min, self.y_max])

                    # more space for agents in same axis (give extra room. not fixing them in line)
                    self.seed()
                    if start_y == self.y_min: start_y += np.random.random() * self.border_relaxer
                    if start_y == self.y_max: start_y -= np.random.random() * self.border_relaxer

                else:
                    # along y axis (top bottom edge)
                    self.seed()
                    start_x = np.random.choice([self.x_min, self.x_max])

                    # more space for agents in same axis (give extra room. not fixing them in line)
                    self.seed()
                    if start_x == self.x_min: start_x += np.random.random() * self.border_relaxer
                    if start_x == self.x_max: start_x -= np.random.random() * self.border_relaxer

                    self.seed()
                    start_y = round(np.random.uniform(self.y_min, self.y_max), 1)

                if len(scenario)==0:
                    collide=False
                    break

                #past_start_points = np.array(scenario)[:,2:4].astype(np.float)
                past_start_points          =   np.array( [ agent.start_global_frame for agent in agents ] ) .astype(np.float)
                past_goal_points           =   np.array( [ agent.goal_global_frame for agent in agents ] ) .astype(np.float)
                past_current_points        =   np.array( [ agent.pos_global_frame for agent in agents ] ) .astype(np.float)

                past_start_points = np.concatenate((past_start_points,past_goal_points,past_current_points))

                if len(past_start_points)==0: break
                
                closet_distance_to_other_start_point = 999
                for past_start_point in past_start_points:
                    distance = np.linalg.norm(past_start_point - np.array([start_x,start_y]) )
                    if distance < closet_distance_to_other_start_point: closet_distance_to_other_start_point = distance

                if closet_distance_to_other_start_point >= self.spawn_distance_threshold: #0.7 #default=1, reduce if it is a crowded scene
                    collide=False
                    break

            #To make sure it is not colliding with other goal point
            collide = True
            while collide:
                self.seed()
                if np.random.choice([True,False]):
                    # along x axis (left right edge)
                    self.seed()
                    goal_x = round(np.random.uniform(self.x_min, self.x_max), 1)

                    #prevent any goal on same side
                    choice_list = np.array([self.y_min, self.y_max])

                    #if start and goal is on the same axis, set goal to opposite side of start
                    if (abs(start_y-self.y_min)<=1) or (abs(start_y-self.y_max)<=1):
                        duplicate_choice = choice_list[ int(np.where( np.abs(choice_list- start_y )<=1 )[0][0]) ]
                        choice_list = list(choice_list)
                        choice_list.remove(duplicate_choice)

                    self.seed()
                    goal_y = np.random.choice(choice_list)

                    # more space for goals in same axis (give extra room. not fixing them in line)
                    self.seed()
                    if goal_y == self.y_min: goal_y += np.random.random()
                    if goal_y == self.y_max: goal_y -= np.random.random()
                else:
                    # along y axis (top bottom edge)
                    self.seed()
                    goal_y = round(np.random.uniform(self.y_min, self.y_max), 1)

                    #prevent any goal on same side
                    choice_list = np.array([self.x_min, self.x_max])

                     #if start and goal is on the same axis, set goal to opposite side of start
                    if (abs(start_x-self.x_min)<=1) or (abs(start_x-self.x_max)<=1):
                        duplicate_choice = choice_list[ int(np.where( abs(choice_list- start_x )<=1 )[0][0]) ]
                        choice_list = list(choice_list)
                        choice_list.remove(duplicate_choice)

                    self.seed()
                    goal_x = np.random.choice(choice_list)

                    # more space for goals in same axis (give extra room. not fixing them in line)
                    self.seed()
                    if goal_x == self.x_min: goal_x += np.random.random()
                    if goal_x == self.x_max: goal_x -= np.random.random()

                if len(scenario)==0:
                    collide=False
                    break

                #past_goal_points = np.array(scenario)[:,4:6].astype(np.float)
                past_start_points          =   np.array( [ agent.start_global_frame for agent in agents ] ) .astype(np.float)
                past_goal_points           =   np.array( [ agent.goal_global_frame for agent in agents ] ) .astype(np.float)
                past_current_points        =   np.array( [ agent.pos_global_frame for agent in agents ] ) .astype(np.float)

                past_goal_points = np.concatenate((past_start_points,past_goal_points,past_current_points))

                if len(past_goal_points)==0: break

                closet_distance_to_other_goal_point = 999
                for past_goal_point in past_goal_points:
                    distance = np.linalg.norm(past_goal_point - np.array([goal_x,goal_y]) )
                    if distance < closet_distance_to_other_goal_point: closet_distance_to_other_goal_point = distance

                if closet_distance_to_other_goal_point >= self.spawn_distance_threshold: #0.7 #default=1, reduce if it is a crowded scene
                    collide=False
                    break

            agent_radius = self.agent_radius
            
            #print([ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ])
            scenario.append( [ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ] )

        return [ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ]
        
########
class real_dataset_traj(object):

    def __init__( self, dataset_name="hotel"):
        self.random_seed=0
        self.spawn_distance_threshold = 0.7 #0.7
        
        data_dir = os.path.dirname(__file__)+"/datasets/"+dataset_name.lower()+"/test/"
        delim='\t'
        
        all_files = os.listdir(data_dir)
        all_files = [os.path.join(data_dir, _path) for _path in all_files]

        print(all_files)


        filtered_frame_data = []
        for path in all_files:
            data = read_file(path, delim)

            agent_list = np.unique(data[:, 1]).tolist()
            print(agent_list)

            frame_data = []
            for frame in agent_list:
                frame_data.append(data[frame == data[:, 1], :])

            #start filtering data
            
            #iterate through agents
            for i,record in enumerate(frame_data):
                diff = np.array(record[-1][2:]) - np.array(record[0][2:])

                #remove record if start and goal are too close     or if the agent record have less than 8 timestamps
                if (np.linalg.norm(diff) < 2) or (len(record)<8): continue
                filtered_frame_data.append( record[:,2:]  )


        self.dataset_traj = np.array(filtered_frame_data)
        
    def seed(self):
        self.random_seed+=1
        np.random.seed(self.random_seed)
        
    def pick_one(self, agents, random_seed=0):
        self.random_seed      = random_seed*1000
        np.random.seed(self.random_seed)
        #get target agent's start, goal, current position, so that when picking new traj, prevent choosing one that is close to these points
        num_traj = len(self.dataset_traj)

        collide = True
        while collide:
            self.seed()
            index = np.random.randint(0,num_traj)

            picked_agent_traj = self.dataset_traj[index]

            start_x,start_y = picked_agent_traj[7] #[0]  not 0, because we want history before the starting point, so we use [7], the number 8 element as starting point
            goal_x,goal_y   = picked_agent_traj[-1]

            past_traj = picked_agent_traj[:8]

            #past_goal_points = np.array(scenario)[:,4:6].astype(np.float)
            past_start_points          =   np.array( [ agent.start_global_frame for agent in agents ] ) .astype(np.float)
            past_goal_points           =   np.array( [ agent.goal_global_frame for agent in agents ] ) .astype(np.float)
            past_current_points        =   np.array( [ agent.pos_global_frame for agent in agents ] ) .astype(np.float)

            own_goal_points            = np.array( [[ goal_x, goal_y  ]]).astype(np.float)

            past_start_points = np.concatenate((past_start_points,past_goal_points,past_current_points,own_goal_points))

            if len(past_goal_points)==0: break

            closet_distance_to_other_start_point = 999
            for past_start_point in past_start_points:
                distance = np.linalg.norm(past_start_point - np.array([start_x,start_y]) )
                if distance < closet_distance_to_other_start_point: closet_distance_to_other_start_point = distance

            if closet_distance_to_other_start_point >= self.spawn_distance_threshold: #0.7 #default=1, reduce if it is a crowded scene
                collide=False
                break

        return [ None , None ,start_x, start_y, goal_x, goal_y, past_traj, None, None ]

    def pick_start(self, population_density, policy_list, x_min, x_max, y_min, y_max, pref_speed, agent_radius, start_timestamp, random_seed=0, num_agents_override=None):

        self.random_seed      = random_seed*1000
        np.random.seed(self.random_seed)

        if num_agents_override is None:
            self.num_agents       = int(round(population_density * ( ( x_max - x_min )  *  ( y_max - y_min )  )))
            print(str(self.num_agents)+" agents created under population density of "+str(population_density))
            if self.num_agents <=1: self.num_agents=2
            #override
            if self.num_agents <=2: self.num_agents=3
        else:
            self.num_agents = num_agents_override
            
        self.policy_list      = policy_list
        self.x_min            = x_min
        self.x_max            = x_max
        self.y_min            = y_min
        self.y_max            = y_max
        self.pref_speed       = pref_speed
        self.agent_radius     = agent_radius
        self.start_timestamp  = start_timestamp

        scenario = []

        ###while loop till we find the start position for all agents so it doesnot collide when spawn
    
        for i in range(self.num_agents):

            #modify the random seed for each agent
            
            if type(self.policy_list)==list:
                policy = self.policy_list[i]
            else:
                policy = self.policy_list

            if type(self.start_timestamp)==list:
                start_timestamp = self.start_timestamp[i]
            else:
                start_timestamp = self.start_timestamp

            if type(self.pref_speed)==list:
                pref_speed = self.pref_speed[i] #* speed_rescaler
            else:
                pref_speed = self.pref_speed #* speed_rescaler

            #make sure speed is not too slow
            speed_threshold = 0.2
            if (pref_speed < speed_threshold ): pref_speed = speed_threshold

            #To make sure it is not colliding with other start point
            collide = True
            while collide:
                self.seed()
                num_traj = len(self.dataset_traj)
                index = np.random.randint(0,num_traj)

                picked_agent_traj = self.dataset_traj[index]

                start_x,start_y = picked_agent_traj[7] #[0]  not 0, because we want history before the starting point, so we use [7], the number 8 element as starting point
                goal_x,goal_y   = picked_agent_traj[-1]

                past_traj = picked_agent_traj[:8]

                #First agent traj added, no collision yet, so add directly
                if len(scenario)==0:
                    collide=False
                    break

                past_start_points = np.array(scenario)[:,2:4].astype(np.float)
                past_goal_points  = np.array(scenario)[:,4:6].astype(np.float)

                past_start_points = np.concatenate((past_start_points,past_goal_points))

                if len(past_start_points)==0: break
                
                closet_distance_to_other_start_point = 999
                for past_start_point in past_start_points:
                    distance = np.linalg.norm(past_start_point - np.array([start_x,start_y]) )
                    if distance < closet_distance_to_other_start_point: closet_distance_to_other_start_point = distance

                if closet_distance_to_other_start_point >= self.spawn_distance_threshold: #0.7 #default=1, reduce if it is a crowded scene
                    collide=False
                    break


            
            #print([ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, start_timestamp ])
            scenario.append( [ i, policy, start_x, start_y, goal_x, goal_y, pref_speed, agent_radius, past_traj ] )

        return scenario
