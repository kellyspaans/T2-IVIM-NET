"""
K.E. Spaans 2024

This script contains code for training the T2-IVIM-NET model on simulated data.
"""

import numpy as np
import IVIMNET_model.IVIMNET.deep_T2 as deep
import torch
from IVIMNET_model.hyperparams_orig import hyperparams as hp_orig
from IVIMNET_model.hyperparams_seg import hyperparams as hp_seg
from IVIMNET_model.hyperparams_seg_fix_100 import hyperparams as hp_seg_fix_100
from IVIMNET_model.hyperparams_seg_fix_150 import hyperparams as hp_seg_fix_150


def train_IVIM_sim(model, data_train, bvalues_all, TEs, arg, folder_experiment):
    '''
    This function trains a T2-IVIM-NET model and saves this model within the experiment folder
    map. It takes the following variables as input:
    - model (str): specifies the model used for training
    - data_train: training data
    - bvalues_all: 1D array of b-values
    - TEs: 1D array of echo times
    - arg: necessary arguments
    - folder_experiment: folder name where results of experiment are stored
    '''

    # get according hyperparameters for the model used
    if model == "T2_IVIM_NET_orig":
        arg = hp_orig()
        arg = deep.checkarg(arg)
    elif model == "T2_IVIM_NET_seg":
        arg = hp_seg()
        arg = deep.checkarg(arg)
    elif model == "T2_IVIM_NET_seg_fix_100":
        arg = hp_seg_fix_100()
        arg = deep.checkarg(arg)
    elif model == "T2_IVIM_NET_seg_fix_150":
        arg = hp_seg_fix_150()
        arg = deep.checkarg(arg)
    else:
        print("PLEASE SPECIFY MODEL\n")

    # normalize the data
    datatot = data_train/data_train.max()

    # NETWORK TRAINING
    print('NN fitting\n')

    # remove NaN
    res = [i for i, val in enumerate(datatot != datatot) if not val.any()]

    data = datatot[res]

    # train network
    net = deep.learn_IVIM(data, bvalues_all, TEs, arg)

    # save the model
    torch.save(net, f'{folder_experiment}/trained_{model}.pth')