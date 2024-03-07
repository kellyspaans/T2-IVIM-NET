"""
September 2020 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved unsupervised physics-informed deep learning for intravoxel-incoherent motion modeling and evaluation in pancreatic cancer patients. MRM 2021)

requirements:
numpy
torch
tqdm
matplotlib
scipy
joblib
"""
import torch
import numpy as np


#most of these are options from the article and explained in the M&M.
class train_pars:
    def __init__(self,nets):
        self.optim='adam' #these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
        if nets == 'optim':
            self.lr = 0.00003 # this is the learning rate.
            # self.lr = 0.00001
        elif nets == 'orig':
            self.lr = 0.001  # this is the learning rate.
        else:
            self.lr = 0.00003 # this is the learning rate.
        self.patience= 10 # this is the number of epochs without improvement that the network waits untill determining it found its optimum
        self.batch_size= 128 # number of datasets taken along per iteration
        self.maxit = 500 # max iterations per epoch
        self.split = 0.9 # split of IVIM_get_parameter_maps.py and validation data
        self.load_nn= False # load the neural network instead of retraining
        self.loss_fun = 'L1' # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
        self.skip_net = False # skip the network training and evaluation
        self.scheduler = False # as discussed in the article, LR is important. This approach allows to reduce the LR itteratively when there is no improvement throughout an 5 consecutive epochs
        # use GPU if available
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = True
        self.freeze_T2 = True # segmented-fit in which we freeze part of the network
        self.stages = 3 # number of stages used in freeze-fitting (if regular training loop, set to 1)
        self.folderfig = ''



class net_pars:
    def __init__(self,nets):
        # select a network setting
        if (nets == 'optim'):
            # the optimized network settings
            self.dropout = 0.1 #0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
            self.batch_norm = False # False/True turns on batch normalistion
            self.parallel = 'parallel' # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
            self.con = 'sigmoid' # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
            self.tri_exp = False
            self.T2 = True
            self.fix_T2p = False
            if self.fix_T2p:
                self.T2p_fixed_val = 0.1 # setting a fixed value for T2p; usually either 100ms or 150ms
            #### only if sigmoid constraint is used!
            if self.tri_exp:
                self.cons_min = [0., 0.0003, 0.0, 0.003, 0.0, 0.08] # F0', D0, F1', D1, F2', D2
                self.cons_max = [2.5, 0.003, 1, 0.08, 1, 5]  # F0', D0, F1', D1, F2', D2
            elif self.T2: # KS: added -- note that in thesis of Betsch the max value of Fp was set to 0.3 instead of 1
                if self.fix_T2p:
                    self.cons_min = [0, 0, 0, 0, 0]  # Dt, Fp, Ds, S0, T2t
                    self.cons_max = [0.003, 1, 0.05, 1, 0.12]  # Dt, Fp, Ds, S0, T2t
                else:
                    self.cons_min = [0, 0, 0, 0, 0, 0]  # Dt, Fp, Ds, S0, T2p, T2t
                    self.cons_max = [0.003, 1.0, 0.05, 1.0, 0.15, 0.12] # Dt, Fp, Ds, S0, T2p, T2t
            else:
                self.cons_min = [0, 0, 0, 0]  # Dt, Fp, Ds, S0
                self.cons_max = [0.003, 1, 0.05, 1]  # Dt, Fp, Ds, S0
            ####
            self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
            self.depth = 2 # number of layers
            self.width = 0 # new option that determines network width. Putting to 0 makes it as wide as the number of b-values
            self.width_first_layer = 0
            self.segmented_fit = True

        elif nets == 'orig':
            # as summarized in Table 1 from the main article for the original network
            self.dropout = 0.0 #0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
            self.batch_norm = True # False/True turns on batch normalistion
            self.parallel = 'single'  # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
            self.con = 'abs' # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
            self.tri_exp = False
            #### only if sigmoid constraint is used!
            if self.tri_exp:
                self.cons_min = [0., 0.0003, 0.0, 0.003, 0.0, 0.08] # F0', D0, F1', D1, F2', D2
                self.cons_max = [2.5, 0.003, 1, 0.08, 1, 5]  # F0', D0, F1', D1, F2', D2
            elif self.T2: # KS: added -- note that in thesis of Betsch the max value of Fp was set to 0.3 instead of 1
                self.cons_min = [0, 0, 0, 0, 0.02, 0.04]  # Dt, Fp, Ds, S0, T2p, T2t
                self.cons_max = [0.003, 1, 0.05, 1, 0.15, 0.12]  # Dt, Fp, Ds, S0, T2p, T2t
            else:
                self.cons_min = [0, 0, 0.005, 0]  # Dt, Fp, Ds, f0
                self.cons_max = [0.005, 0.7, 0.2, 2.0]  # Dt, Fp, Ds, f0
            ####
            self.fitS0 = False # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
            self.depth = 3 # number of layers
            self.width = 0 # new option that determines network width. Putting to 0 makes it as wide as the number of b-values
        else:
            # the optimized network settings
            self.dropout = 0.2 #0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
            self.batch_norm = False # False/True turns on batch normalistion
            self.parallel = 'parallel' # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
            self.con = 'abs' # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
            self.tri_exp = False
            #### only if sigmoid constraint is used!
            if self.tri_exp:
                self.cons_min = [0., 0.0003, 0.0, 0.003, 0.0, 0.08] # F0', D0, F1', D1, F2', D2
                self.cons_max = [2.5, 0.003, 1, 0.08, 1, 5]  # F0', D0, F1', D1, F2', D2
            elif self.T2: # KS: added
                self.cons_min = [0, 0, 0, 0, 0.02, 0.04]  # Dt, Fp, Ds, S0, T2p, T2t
                self.cons_max = [0.003, 1, 0.05, 1, 0.15, 0.12]  # Dt, Fp, Ds, S0, T2p, T2t
            else:
                self.cons_min = [0, 0, 0.005, 0]  # Dt, Fp, Ds, S0
                self.cons_max = [0.005, 0.7, 0.3, 2.0]  # Dt, Fp, Ds, S0
            ####
            self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
            self.depth = 2 # number of layers
            self.width = 0 # new option that determines network width. Putting to 0 makes it as wide as the number of b-values

        boundsrange = 0.3 * (np.array(self.cons_max)-np.array(self.cons_min)) # ensure that we are on the most lineair bit of the sigmoid function
        self.cons_min = np.array(self.cons_min) - boundsrange
        self.cons_max = np.array(self.cons_max) + boundsrange




class lsqfit:
    def __init__(self):
        self.method = 'lsq' #seg, bayes or lsq
        self.model = 'T2' #bi-exp, tri-exp, or T2
        self.do_fit = False #lsq fitting
        self.load_lsq = False # load the last results for lsq fit
        self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 in the least squares fit.
        self.jobs = 1 # number of parallel jobs. If set to 1, no parallel computing is used
        if self.model == 'tri-exp':
            self.bounds = ([0, 0, 0, 0.008, 0, 0.06], [2.5, 0.008, 1, 0.08, 1, 5]) # F0', D0, F1', D1, F2', D2
        elif self.model == 'T2':
            # self.bounds = ([0, 0, 0, 0, 0, 0], [2.5, 0.003, 1, 0.05, 0.15, 0.12])  # Fp0, Dt, Fp, Dp, T2p, T2t
            self.bounds = ([0, 0, 0, 0, 0, 0], [2.5, 0.008, 1, 0.06, 0.2, 0.2]) # loosened boundaries
        else:
            self.bounds = ([0, 0, 0, 0], [0.003, 1, 0.05, 1])  # Dt, Fp, Ds, S0


class sim:
    def __init__(self):
        self.bvalues = np.array([0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700]) # array of b-values
        self.TEs = np.array([60, 60, 60, 60, 76, 76, 76, 76, 110, 110, 110])
        self.SNR = [0] # the SNRs to simulate at
        self.sims = 10000 # number of simulations to run
        self.num_samples_eval = 1000 # number of simualtions te evaluate. This can be lower than the number run. Particularly to save time when fitting. More simulations help with generating sufficient data for the neural network
        self.repeats = 3 # this is the number of repeats for simulations
        self.rician = False # add rician noise to simulations; if false, gaussian noise is added instead
        self.model = 'T2'
        if self.model == 'bi-exp':
            self.range = ([0.0005, 0.05, 0.01], [0.003, 0.55, 0.1])
        elif self.model == 'T2':
            self.range = ([0.0005, 0.05, 0.001, 0.02, 0.04], [0.003, 1, 0.05, 0.15, 0.12]) # Dt, Fp, Ds, T2p, T2t
        else:
            self.range = ([0.0005, 0.05, 0.001, 0.05, 0.08], [0.003, 0.5, 0.05, 0.5, 2]) # D0, F1', D1, F2', D2


class hyperparams:
    def __init__(self):
        self.fig = False # plot results and intermediate steps
        self.save_name = 'optim' # orig or optim (or optim_adsig for in vivo)
        self.net_pars = net_pars(self.save_name)
        self.train_pars = train_pars(self.save_name)
        self.fit = lsqfit()
        self.sim = sim()
