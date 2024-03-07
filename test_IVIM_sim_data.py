"""
K.E. Spaans 2024

This script contains code for testing the trained T2-IVIM-NET model on simulated data.
"""

import IVIMNET_model.IVIMNET.deep_T2 as deep
import torch


def test_IVIM_sim(model, data_test, bvalues_all, TEs, arg, folder_experiment):
    '''
    This function estimates T2-IVIM parameters based on the test data and
    the trained model.
    It takes the following variables as input:
    - model (str): specifies the model used for training
    - data_test: test data
    - bvalues_all: 1D array of b-values
    - TEs: 1D array of echo times
    - folder_experiment: folder name where results of experiment are stored
    - folder_experiment_data: folder name where data is stored

    returns: predicted parameters.
    '''

    # load the model
    net = torch.load(f'{folder_experiment}/trained_{model}.pth')
    net.eval()

    # normalize data
    datatot2 = data_test/data_test.max()

    # predict parameters
    paramsNN = deep.predict_IVIM(datatot2, bvalues_all, TEs, net, arg)

    return paramsNN

