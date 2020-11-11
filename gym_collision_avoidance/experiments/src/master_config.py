number_of_agent = 60

from master_scenario_generator import Scenario_Generator, Seeded_Scenario_Generator, Seeded_Population_Scenario_Generator

class Master_Config(object):
    def __init__(self):
        #EvaluateConfig Level
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = number_of_agent+1
        self.EVALUATE_MODE = True
        self.TRAIN_MODE = False
        self.DT = 0.1 #0.1
        self.MAX_TIME_RATIO = 3. #8.
        #Formations config Level

        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = True
        self.NEAR_GOAL_THRESHOLD = 0.2
        #ETH   [[-10, 17], [-6, 16]]
        #HOTEL [[-7,7],[-13,7]]
        #UNIV  [[-3,17],[-3,15]]
        #ZARA1 [[-3,17],[-3,15]]
        #ZARA2 [[-3,17],[-3,15]]

        #Population [[-1,11],[-1,11]]

        #Motion prediction [[-10,10],[-10,10]]
        
        self.PLT_LIMITS = [[-15,15],[-15,15]] #[[-10,10],[-10,10]] #[[-15, 15], [-15, 15]]  #display graph grid showing dimension limit
        self.PLT_FIG_SIZE = (10,10) #Actual hidden limit
        self.PLOT_CIRCLES_ALONG_TRAJ = False
        self.NUM_AGENTS_TO_TEST = [60]
        #self.POLICIES_TO_TEST = ['GA3C-CADRL-10']
        self.POLICIES_TO_TEST = ['GA3C-CADRL-10']*60 #['RVO']*number_of_agent#['STGCNN']*number_of_agent #['CADRL']*7#['NAVIGAN']*7#['RVO']*7#['GA3C-CADRL-10']*7
        self.NUM_TEST_CASES = 2 #correspond to how many letters are there

        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = number_of_agent+1 
        self.MAX_NUM_OTHER_AGENTS_OBSERVED = number_of_agent

        self.agent_time_out          = 180 #180  #30 motion prediction

#       num mean        num std dev      vel mean       vel std dev   x_min             x_max   y_min   y_max
#ETH    6.312138728	4.536521361      2.339926573	0.7502205478  -7.69	        14.42	-3.17	13.21
#HOTEL  5.598098531	3.418910729      1.137002131	0.6487607538  -3.25	        4.35	-10.31	4.31
#UNIV   40.83024533	6.734736777      0.6817478507	0.2481828799  -0.4619709156	15.46918556	-0.3183721728	13.89190962
#ZARA1  5.87224158	3.213275774      1.12739064	0.2946279183  -0.1395383677	15.48055067	-0.3746958856	12.38644361
#ZARA2  9.314121037	3.926104465      1.096467485	0.3849301882  -0.3577906864	15.55842276	-0.2737427903	13.94274416

class Scenario_Config(object):
    def __init__(self):
        #generate random scenario here, write a function to generate and pass to self.scenario

        self.scenario=[]
        for i in range(20):  #100
            #######
            #random seed
            #
            #(ETH) GA3C-CADRL   
            #self.scenario.append( Scenario_Generator( 6.312138728, "GA3C-CADRL-10", -7.69, 14.42, -3.17, 13.21 , 2.339926573, 0.05 , 0, num_agents_stddev=4.536521361, pref_speed_stddev=0.7502205478 ).random_square_edge() )

            ###################################FULL traj output 100  0.05 radius  ###########################
            #fixed seed
            #(ETH)   GA3C-CADRL  fixed seed
            #self.scenario.append( Seeded_Scenario_Generator( 6.312138728, "GA3C-CADRL-10", -7.69, 14.42, -3.17, 13.21 , 2.339926573, 0.05 , 0, num_agents_stddev=4.536521361, pref_speed_stddev=0.7502205478, random_seed=i ).random_square_edge() )

            #(HOTEL) GA3C-CADRL  fixed seed
            #self.scenario.append( Seeded_Scenario_Generator( 5.598098531, "GA3C-CADRL-10", -3.25, 4.35, -10.31, 4.31 ,  1.137002131, 0.05 , 0, num_agents_stddev=3.418910729, pref_speed_stddev=0.6487607538, random_seed=i ).random_square_edge() )
            #(UNIV)  GA3C-CADRL  fixed seed
            #self.scenario.append( Seeded_Scenario_Generator( 40.83024533, "CADRL", -0.4619709156, 15.46918556, -0.3183721728, 13.89190962 ,  0.6817478507, 0.05 , 0, num_agents_stddev=6.734736777, pref_speed_stddev=0.2481828799, random_seed=i ).random_square_edge() )


            #(ZARA1) GA3C-CADRL  fixed seed
            #self.scenario.append( Seeded_Scenario_Generator( 5.87224158 , "RVO", -0.1395383677, 15.48055067,	-0.3746958856,	12.38644361 ,  1.12739064, 0.05 , 0, num_agents_stddev=3.213275774, pref_speed_stddev=0.2946279183, random_seed=i ).random_square_edge() )

            #(ZARA2) GA3C-CADRL  fixed seed
            #self.scenario.append( Seeded_Scenario_Generator( 9.314121037, "CADRL", -0.3577906864,  15.55842276,-0.2737427903,	13.94274416 ,  1.096467485, 0.05 , 0, num_agents_stddev=3.926104465, pref_speed_stddev=0.3849301882, random_seed=i ).random_square_edge() )


            ###################################FULL traj output 20  0.2 radius  ###########################
            #fixed seed
            #(ETH)   GA3C-CADRL  fixed seed
            #self.scenario.append( Seeded_Scenario_Generator( 6.312138728, "RVO", -7.69, 14.42, -3.17, 13.21 , 2.339926573, 0.2 , 0, num_agents_stddev=4.536521361, pref_speed_stddev=0.7502205478, random_seed=i ).random_square_edge() )

            #(HOTEL) GA3C-CADRL  fixed seed
            #self.scenario.append( Seeded_Scenario_Generator( 5.598098531, "GA3C-CADRL-10", -3.25, 4.35, -10.31, 4.31 ,  1.137002131, 0.2 , 0, num_agents_stddev=3.418910729, pref_speed_stddev=0.6487607538, random_seed=i ).random_square_edge() )
            
            #(UNIV)  GA3C-CADRL  fixed seed
            #self.scenario.append( Seeded_Scenario_Generator( 40.83024533, "RVO", -0.4619709156, 15.46918556, -0.3183721728, 13.89190962 ,  0.6817478507, 0.2 , 0, num_agents_stddev=6.734736777, pref_speed_stddev=0.2481828799, random_seed=i ).random_square_edge() )


            #(ZARA1) GA3C-CADRL  fixed seed
            #self.scenario.append( Seeded_Scenario_Generator( 5.87224158 , "RVO", -0.1395383677, 15.48055067,	-0.3746958856,	12.38644361 ,  1.12739064, 0.2 , 0, num_agents_stddev=3.213275774, pref_speed_stddev=0.2946279183, random_seed=i ).random_square_edge() )

            #(ZARA2) GA3C-CADRL  fixed seed
            #self.scenario.append( Seeded_Scenario_Generator( 9.314121037, "RVO", -0.3577906864,  15.55842276,-0.2737427903,	13.94274416 ,  1.096467485, 0.2 , 0, num_agents_stddev=3.926104465, pref_speed_stddev=0.3849301882, random_seed=i ).random_square_edge() )

            #(ZARA2) testing with SPEC / STGCNN
            #self.scenario.append( Seeded_Scenario_Generator( 15, "SLSTM", -5, 5,-5,	5 ,  1.096467485, 0.2 , 0, num_agents_stddev=0.001, pref_speed_stddev=0.3849301882, random_seed=i ).random_square_edge() )
            #self.scenario.append( Seeded_Scenario_Generator( 15, "SOCIALGAN", -5, 5,-5,	5 ,  1.096467485, 0.2 , 0, num_agents_stddev=0.001, pref_speed_stddev=0.3849301882, random_seed=i ).random_square_edge() )


            #self.scenario.append( Seeded_Scenario_Generator( 30, "SLSTM", -6, 6,-6,	6 ,  1.096467485, 0.2 , 0, num_agents_stddev=0.001, pref_speed_stddev=0.3849301882, random_seed=i ).random_square_edge() )
            #self.scenario.append( Seeded_Scenario_Generator( 5, "SOCIALGAN", -3, 3,-3,	3 ,  1.096467485, 0.2 , 0, num_agents_stddev=0.001, pref_speed_stddev=0.3849301882, random_seed=i ).random_square_edge() )
            self.scenario.append( Seeded_Scenario_Generator( 30, "SPEC", -6, 6,-6,	6 ,  1.096467485, 0.2 , 0, num_agents_stddev=0.001, pref_speed_stddev=0.3849301882, random_seed=i ).random_square_edge() )

            #self.scenario.append( Seeded_Scenario_Generator( 20, "SPEC", -10,  10,-10,	10 ,  1, 0.2 , 0, num_agents_stddev=0.01, pref_speed_stddev=0, random_seed=i ).random_square_edge() )



            ################Population density fixed seed fixed speed (1m/s), 0.2m radius gradually increase density, 10x10m #############################
            #0.1
            #self.scenario.append( Seeded_Population_Scenario_Generator( 0.1, "RVO", 0, 10, 0, 10, 1, 0.2, 0, random_seed=i  ).population_random_square_edge() )
            
            #0.15
            #self.scenario.append( Seeded_Population_Scenario_Generator( 0.15, "GA3C-CADRL-10", 0, 10, 0, 10, 1, 0.2, 0, random_seed=i  ).population_random_square_edge() )

            #0.2
            #self.scenario.append( Seeded_Population_Scenario_Generator( 0.2, "GA3C-CADRL-10", 0, 10, 0, 10, 1, 0.2, 0, random_seed=i  ).population_random_square_edge() )

            #0.25
            #self.scenario.append( Seeded_Population_Scenario_Generator( 0.25, "GA3C-CADRL-10", 0, 10, 0, 10, 1, 0.2, 0, random_seed=i  ).population_random_square_edge() )

            #0.3
            #self.scenario.append( Seeded_Population_Scenario_Generator( 0.3, "RVO", 0, 10, 0, 10, 1, 0.2, 0, random_seed=i  ).population_random_square_edge() )

            #0.35
            #self.scenario.append( Seeded_Population_Scenario_Generator( 0.35, "RVO", 0, 10, 0, 10, 1, 0.2, 0, random_seed=i  ).population_random_square_edge() )

            #0.4
            #self.scenario.append( Seeded_Population_Scenario_Generator( 0.4, "RVO", 0, 10, 0, 10, 1, 0.2, 0, random_seed=i  ).population_random_square_edge() )

            #0.45
            #self.scenario.append( Seeded_Population_Scenario_Generator( 0.45, "RVO", 0, 10, 0, 10, 1, 0.2, 0, random_seed=i  ).population_random_square_edge() )
            
            #0.5
            #self.scenario.append( Seeded_Population_Scenario_Generator( 0.5, "RVO", 0, 10, 0, 10, 1, 0.2, 0, random_seed=i  ).population_random_square_edge() )

            #0.55
            #self.scenario.append( Seeded_Population_Scenario_Generator( 0.55, "GA3C-CADRL-10", 0, 10, 0, 10, 1, 0.2, 0, random_seed=i  ).population_random_square_edge() )

'''
self.scenario = {
        'S':
        [
                [0    ,"GA3C-CADRL-10"     ,-0.73	,4.63        ,10.43	,5.4    ,1.0           ,0.05   ,0],
                [1    ,"GA3C-CADRL-10"     ,-0.57	,5.34        ,9.99 	,6.24   ,1.0           ,0.05   ,0],
                [2    ,"GA3C-CADRL-10"     ,-0.42	,5.89        ,11.64	,5.68   ,1.0           ,0.05   ,0],
                [3    ,"GA3C-CADRL-10"     ,-0.67	,6.63        ,11.78	,6.38   ,1.0           ,0.05   ,0],
                [4    ,"GA3C-CADRL-10"     ,12.45	,4.92        ,0.86 	,1.71   ,1.0           ,0.05   ,0],
                [5    ,"GA3C-CADRL-10"     ,12.7 	,5.68        ,0.85 	,2.58   ,1.0           ,0.05   ,0],
                [6    ,"GA3C-CADRL-10"     ,11.93	,5.99        ,-0.0 	,2.9    ,1.0           ,0.05   ,0]
        ],

        'A':
        [
                [0    ,"GA3C-CADRL-10"     ,-2	,-4        ,2	        ,-4   ,1.0           ,0.05   ,0],
                [1    ,"GA3C-CADRL-10"     ,-2	,-3        ,2	  	,-3   ,1.0           ,0.05   ,0],
                [2    ,"GA3C-CADRL-10"     ,-2	,-2        ,2	  	,-2   ,1.0           ,0.05   ,0],
                [3    ,"RVO"               ,-2	,-1        ,2	  	,-1   ,1.0           ,0.05   ,0],
                [4    ,"RVO"               , 2	,-4        ,-2	   	,-4    ,1.0           ,0.05   ,0],
                [5    ,"GA3C-CADRL-10"     , 2 	,-1        ,-2	   	,-1    ,1.0           ,0.05   ,0],
                [6    ,"GA3C-CADRL-10"     , 2	,-2        ,-2	   	,-2    ,1.0           ,0.05   ,0]
        ]

        }
'''
    
'''

#ID  #Algorithm     #Start_x   Start_y      Goal_x      Goal_y    Pref_Speed    Radius    Start_timestamp
0    "RVO"            -0.73	4.63        10.43	5.4       1.0           0.05      0
1    "RVO"            -0.57	5.34        9.99 	6.24      1.0           0.05      0
2    "RVO"            -0.42	5.89        11.64	5.68      1.0           0.05      0
3    "RVO"            -0.67	6.63        11.78	6.38      1.0           0.05      0
4    "GA3C-CADRL"     12.45	4.92        0.86 	1.71      1.0           0.05      0
5    "GA3C-CADRL"     12.7 	5.68        0.85 	2.58      1.0           0.05      0
6    "GA3C-CADRL"     11.93	5.99        -0.0 	2.9       1.0           0.05      0
'''
