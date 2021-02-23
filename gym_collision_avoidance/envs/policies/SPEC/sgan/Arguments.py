
class Arguments():
    def __init__(self,token):
        self.dataset_name = 'zara1'
        self.delim = '\t'
        self.loader_num_workers = 4
        self.min_ped = 2
        self.hist_len = 8
        self.fut_len = 12
        self.loadNpy = 0
        self.untracked_ratio = 1.0
        # Network design
        self.l2d = 1
        self.tanh = 1
        self.n_ch = 2
        self.use_max = 1
        self.targ_ker_num = [50,80] # [7,28]
        self.targ_ker_size = [2,2]
        self.targ_pool_size = [2,2]
        self.cont_ker_num = [-1,160]
        self.cont_ker_size = [2,2]
        self.cont_pool_size = [2,2]
        self.n_fc = -1
        self.fc_width = [300,120,80,5] # 280,200,120,80
        self.output_size = 11
        self.neighbor = 1
        self.drop_rate = 0.0
        self.lock_l2d = 0
        # HyperParam
        self.seed = 1
        self.ave_spd = 0.5
        self.loss_balance = 75.0
        # Training
        self.loadModel = ''
        self.batch_size = 64
        self.n_epoch = 150
        self.n_iteration = 300
        self.lr = 0.001
        self.start = 0
        # Validation and Output
        self.batch_size_val = 2
        self.batch_size_tst = 2
        self.n_batch_val = 6
        self.n_batch_tst = 4
        self.val_freq = 1
        self.n_guess = 2
        self.n_sample = 20
        self.coef = 1.000000001
        self.task_name = "ut"
        self.plotting_weights=0

        self.token = token


        if token == '':
            pass

        elif token=='univ_best':
            self.lr = 0.0005
            self.n_epoch = 1000
            self.dataset_name = 'univ'
            self.batch_size = 1
            self.batch_size_val = 1
            self.batch_size_tst = 1
            self.n_batch_val = 3
            self.n_batch_tst = 3
            self.loadModel = 'univ_best'

        else:
            print("no token!!!")

        print("task:", self.token, self.dataset_name, self.task_name)








# if args.dataset_name =  = "univ": args.n_batch_tst = 1
# if args.dataset_name =  = "zara2": args.n_batch_tst = 2



# elif token=='all_fr_lrntP':
# self.lr = 0.0005
# self.cont_ker_num = [-1]
# self.n_epoch = 50
# self.dataset_name = 'all'
# self.loadModel = 'all_ep100rc'
#
# elif token=='sp_z2':
# self.dataset_name = 'zara2'
# self.loadModel = 'all_fr_lrntP'
# self.loadNpy = 1
# self.plotting_weights=1
# elif token=='sp_hotel':
# self.dataset_name = 'hotel'
# self.loadModel = 'all_fr_lrntP'
# self.loadNpy = 1
# self.plotting_weights=1
# elif token=='sp_univ':
# self.dataset_name = 'univ'
# self.loadModel = 'all_fr_lrntP'
# self.loadNpy = 1
# self.plotting_weights=1
#
#
# elif token=='z1_fr_lrntP':
# self.lr = 0.0005
# self.cont_ker_num = [-1]
# self.n_epoch = 50
# self.dataset_name = 'zara1'
# self.loadModel = 'all_ep100ra'
#
# elif token=='ptw_z1':
# self.lr = 0.0005
# self.cont_ker_num = [-1]
# self.n_epoch = 100
#
# elif token=='ptw_univ':
# self.dataset_name = 'univ'
# self.lr = 0.0005
# self.cont_ker_num = [-1]
# self.n_epoch = 50
#
# elif token=='ptw_all':
# self.dataset_name = 'all'
# self.lr = 0.003
# self.cont_ker_num = [-1]
# self.n_epoch = 100
# self.task_name = "ep100rc"
#
# elif token=="ppr_z1":
# self.dataset_name = 'zara1'
#
# elif token=="plot_sample_z2":
# self.dataset_name = 'zara2'
# self.loadNpy = 0
# elif token=="plot_sample_z1":
# self.dataset_name = 'zara1'
# self.loadNpy = 0
