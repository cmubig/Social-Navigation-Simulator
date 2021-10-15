number_of_agent = 100
import os

from .master_scenario_generator import Scenario_Generator, Seeded_Scenario_Generator, Seeded_Population_Scenario_Generator, real_dataset_traj

class Master_Config(object):
    def __init__(self):

        global_timeout            = int(os.getenv("global_timeout", 60))
        global_experiment_number  = int(os.getenv("global_experiment_number", 1))
        global_dataset_name       = os.getenv("global_dataset_name", "ETH")
        global_population_density = float(os.getenv("global_population_density", 0.3))

##        print("FROM MASTER")
##        print(os.environ["global_timeout"] )
##        print(os.environ["global_experiment_number"])
##        print(os.environ["global_dataset_name"])
##        print(global_population_density)

        self.exp_setting = None
        #####################################################################################################################################################
        #                 num mean        num std dev      vel mean       vel std dev   x_min           x_max           y_min           y_max           plot_size
        self.ETH   = [    6.312138728	,4.536521361      ,2.339926573	,0.7502205478  ,-7.69	        ,14.42	        ,-3.17	        ,13.21        , [[-10, 17], [-6, 16]] ]
        self.HOTEL = [    5.598098531	,3.418910729      ,1.137002131	,0.6487607538  ,-3.25	        ,4.35	        ,-10.31	        ,4.31         , [[-7,7],[-13,7]]      ]
        self.UNIV  = [    40.83024533	,6.734736777      ,0.6817478507	,0.2481828799  ,-0.4619709156	,15.46918556	,-0.3183721728	,13.89190962  , [[-3,17],[-3,15]]     ]
        self.ZARA1 = [    5.87224158	,3.213275774      ,1.12739064	,0.2946279183  ,-0.1395383677	,15.48055067	,-0.3746958856	,12.38644361  , [[-3,17],[-3,15]]     ]
        self.ZARA2 = [    9.314121037	,3.926104465      ,1.096467485	,0.3849301882  ,-0.3577906864	,15.55842276	,-0.2737427903	,13.94274416  , [[-3,17],[-3,15]]     ]


        ########################################################################################################
        #                      num mean        num std dev      vel mean       vel std dev   x_min           x_max           y_min           y_max      plot_size
        #self.POPULATION = [    None            ,None            ,1             ,None         ,0              ,10             ,0              ,10      , [[-1,11],[-1,11]]     ]
        self.POPULATION = [    None            ,None            ,1             ,None         ,0              ,5             ,0              ,5      , [[-1,6],[-1,6]]     ]
        #generate random scenario here, write a function to generate and pass to self.scenario

        if global_experiment_number == 1: #Simulate algorithm using settings from datasets! (e.g. ETH)
            #print(global_dataset_name)
            if   global_dataset_name == "ETH"     : self.exp_setting = self.ETH
            elif global_dataset_name == "HOTEL"   : self.exp_setting = self.HOTEL          
            elif global_dataset_name == "UNIV"    : self.exp_setting = self.UNIV
            elif global_dataset_name == "ZARA1"   : self.exp_setting = self.ZARA1 
            elif global_dataset_name == "ZARA2"   : self.exp_setting = self.ZARA2

            self.PLT_LIMITS = self.exp_setting[8]
##            print("FINALLY")
##            print(self.PLT_LIMITS)

        elif global_experiment_number == 2: #population density evaluation

            #####for high population density, reduce size, hence less agents required######
            #if global_population_density >= 0.5:
            #    self.POPULATION = [    None            ,None            ,1             ,None         ,0              ,5             ,0              ,5      , [[-1,6],[-1,6]]     ]
            
            self.exp_setting = self.POPULATION
            self.PLT_LIMITS = self.exp_setting[8]

        elif global_experiment_number == 3: #touranment  1 vs n-1
            self.exp_setting = self.POPULATION
            self.PLT_LIMITS = self.exp_setting[8]

        elif global_experiment_number == 4: #touranment  50% vs 50%
            self.exp_setting = self.POPULATION
            self.PLT_LIMITS = self.exp_setting[8]

        ############################################

            
        #EvaluateConfig Level
        self.EVALUATE_MODE = True
        self.TRAIN_MODE = False
        self.DT = 0.1 #0.1
        self.MAX_TIME_RATIO = 3. #8.
        #Formations config Level

        self.SHOW_EPISODE_PLOTS = False #plot while sim
        self.SAVE_EPISODE_PLOTS  = self.ANIMATE_EPISODES = False  #output gif + mp4
        self.NEAR_GOAL_THRESHOLD = 0.2
        #ETH   [[-10, 17], [-6, 16]]
        #HOTEL [[-7,7],[-13,7]]
        #UNIV  [[-3,17],[-3,15]]
        #ZARA1 [[-3,17],[-3,15]]
        #ZARA2 [[-3,17],[-3,15]]

        #Population [[-1,11],[-1,11]]

        #Motion prediction [[-10,10],[-10,10]]


        self.PLT_FIG_SIZE = (10,10) #Actual hidden limit
        self.PLOT_CIRCLES_ALONG_TRAJ = False
        self.NUM_AGENTS_TO_TEST = [60]
        #self.POLICIES_TO_TEST = ['GA3C-CADRL-10']
        self.POLICIES_TO_TEST = ['GA3C-CADRL-10']*60 #['RVO']*number_of_agent#['STGCNN']*number_of_agent #['CADRL']*7#['NAVIGAN']*7#['RVO']*7#['GA3C-CADRL-10']*7
        self.NUM_TEST_CASES = 2 #correspond to how many letters are there


        self.MAX_NUM_OTHER_AGENTS_OBSERVED = number_of_agent * 3
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = self.MAX_NUM_OTHER_AGENTS_OBSERVED + 1
        
        self.agent_time_out          =  global_timeout #180 seconds normal #30 motion prediction


class Scenario_Config(object):
    def __init__(self, experiment_number, algorithm_name, experiment_iteration_num, dataset_name=None, population_density=None):

        self.exp_setting = None
        #####################################################################################################################################################
        #                 num mean        num std dev      vel mean       vel std dev   x_min           x_max           y_min           y_max           plot_size
        self.ETH   = [    6.312138728	,4.536521361      ,2.339926573	,0.7502205478  ,-7.69	        ,14.42	        ,-3.17	        ,13.21        , [[-10, 17], [-6, 16]] ]
        self.HOTEL = [    5.598098531	,3.418910729      ,1.137002131	,0.6487607538  ,-3.25	        ,4.35	        ,-10.31	        ,4.31         , [[-7,7],[-13,7]]      ]
        self.UNIV  = [    40.83024533	,6.734736777      ,0.6817478507	,0.2481828799  ,-0.4619709156	,15.46918556	,-0.3183721728	,13.89190962  , [[-3,17],[-3,15]]     ]
        self.ZARA1 = [    5.87224158	,3.213275774      ,1.12739064	,0.2946279183  ,-0.1395383677	,15.48055067	,-0.3746958856	,12.38644361  , [[-3,17],[-3,15]]     ]
        self.ZARA2 = [    9.314121037	,3.926104465      ,1.096467485	,0.3849301882  ,-0.3577906864	,15.55842276	,-0.2737427903	,13.94274416  , [[-3,17],[-3,15]]     ]


        ########################################################################################################
        #                      num mean        num std dev      vel mean       vel std dev   x_min           x_max           y_min           y_max      plot_size
        #self.POPULATION = [    None            ,None            ,1             ,None         ,0              ,10             ,0              ,10      , [[-1,11],[-1,11]]     ]
        self.POPULATION = [    None            ,None            ,1             ,None         ,0              ,5             ,0              ,5      , [[-1,6],[-1,6]]     ]
        #generate random scenario here, write a function to generate and pass to self.scenario


        if experiment_number == 1: #Simulate algorithm using settings from datasets! (e.g. ETH)
            #print(dataset_name)
            if   dataset_name == "ETH"     : self.exp_setting = self.ETH
            elif dataset_name == "HOTEL"   : self.exp_setting = self.HOTEL          
            elif dataset_name == "UNIV"    : self.exp_setting = self.UNIV
            elif dataset_name == "ZARA1"   : self.exp_setting = self.ZARA1 
            elif dataset_name == "ZARA2"   : self.exp_setting = self.ZARA2
            
            self.scenario=[]
            for i in range(experiment_iteration_num): #set radius from 0.2 to 0.05 to show slstm do better in low radius situation
                #old approach to gen similar dataset based on speed, num of agents of certain dataset
                #self.scenario.append( Seeded_Scenario_Generator( self.exp_setting[0], algorithm_name, self.exp_setting[4],self.exp_setting[5], self.exp_setting[6], self.exp_setting[7] , self.exp_setting[2], 0.2 , 0, num_agents_stddev=self.exp_setting[1], pref_speed_stddev=self.exp_setting[3], random_seed=i ).random_square_edge() )

                #just use the dataset's real traj
                self.scenario.append( real_dataset_traj( dataset_name=dataset_name ).pick_start( None, algorithm_name, self.exp_setting[4],self.exp_setting[5], self.exp_setting[6], self.exp_setting[7] , self.exp_setting[2],
                                                                                                 0.2 , 0, random_seed=i, num_agents_override= round(self.exp_setting[0]) ) )

        elif experiment_number == 2: #population density evaluation
            
            #####for high population density, reduce size, hence less agents required######
            #if population_density >= 0.5:
            #    self.POPULATION = [    None            ,None            ,1             ,None         ,0              ,5             ,0              ,5      , [[-1,6],[-1,6]]     ]
            
            self.exp_setting = self.POPULATION
            self.scenario=[]
            for i in range(experiment_iteration_num):         
                self.scenario.append( Seeded_Population_Scenario_Generator( population_density, algorithm_name, self.exp_setting[4],self.exp_setting[5], self.exp_setting[6], self.exp_setting[7], self.exp_setting[2], 0.2, 0, random_seed=i  ).population_random_square_edge() )

        elif experiment_number == 3:  #touranment  1 vs n-1
            self.exp_setting = self.POPULATION
            self.scenario=[]
            
            print(algorithm_name)
            algorithm_name = algorithm_name.strip('][').split(',')  #make sure it is transformed back to list
            number_of_agents = int(round(population_density * ( ( self.POPULATION[5] - self.POPULATION[4] )  *  ( self.POPULATION[7] - self.POPULATION[6] )  )))
            
            for i in range(experiment_iteration_num):
                temp_name = []
                for j in range(number_of_agents):   
                    if j==0:
                        temp_name.append( algorithm_name[0] )
                    else:
                        temp_name.append( algorithm_name[1] )

                algorithm_name = temp_name
            
                self.scenario.append( Seeded_Population_Scenario_Generator( population_density, algorithm_name, self.exp_setting[4],self.exp_setting[5], self.exp_setting[6], self.exp_setting[7], self.exp_setting[2], 0.2, 0, random_seed=i  ).population_random_square_edge() )


        elif experiment_number == 4:  #touranment  50% vs 50%
            self.exp_setting = self.POPULATION
            self.scenario=[]
            
            print(algorithm_name)
            algorithm_name = algorithm_name.strip('][').split(',')  #make sure it is transformed back to list
            
            number_of_agents = int(round(population_density * ( ( self.POPULATION[5] - self.POPULATION[4] )  *  ( self.POPULATION[7] - self.POPULATION[6] )  )))
            
            for i in range(experiment_iteration_num):
                temp_name = []

                #number of agents from population density, retrieved from master scenario generator

                for j in range(number_of_agents):   
                    if (j%2)==0:
                        temp_name.append( algorithm_name[0] )
                    else:
                        temp_name.append( algorithm_name[1] )

                algorithm_name = temp_name
                print(algorithm_name)
                
            
                self.scenario.append( Seeded_Population_Scenario_Generator( population_density, algorithm_name, self.exp_setting[4],self.exp_setting[5], self.exp_setting[6], self.exp_setting[7], self.exp_setting[2], 0.2, 0, random_seed=i  ).population_random_square_edge() )

                
        '''
        self.scenario=[]
        for i in range(experiment_iteration_num):  #100
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
