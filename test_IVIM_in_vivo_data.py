"""
K.E. Spaans 2024

This script contains code for testing the trained T2-IVIM-NET model on in vivo patient data.
"""

import os
import time
import nibabel as nib
import numpy as np
from IVIMNET_model.hyperparams import hyperparams as hp
from IVIMNET_model.hyperparams_orig import hyperparams as hp_orig
from IVIMNET_model.hyperparams_seg import hyperparams as hp_seg
from IVIMNET_model.hyperparams_seg_fix_100 import hyperparams as hp_seg_fix_100
from IVIMNET_model.hyperparams_seg_fix_150 import hyperparams as hp_seg_fix_150
import IVIMNET_model.IVIMNET.deep_T2 as deep
import torch
from analysis import calculate_correlation_matrix, calculate_nrmse_signals


def test_IVIM_in_vivo(model, sim, testing_patients, bvalues_all, TEs, folder_experiment, folder_experiment_data):
    '''
    This function estimates T2-IVIM parameters based on the test data and
    the trained model.
    It takes the following variables as input:
    - model (str): specifies the model used for training
    - testing_patients: test data
    - bvalues_all: 1D array of b-values
    - TEs: 1D array of echo times
    - folder_experiment: folder name where results of experiment are stored
    - folder_experiment_data: folder name where data is stored


    returns: predicted parameters and correlation matrix.
    '''

    calculate_cor = True # only calculate correlation matrix once

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
        arg = hp()
        arg = deep.checkarg(arg)
        print("RUNNING BI-EXP MODEL\n")


    # retrieve test patients
    subjects = sorted([folder_experiment_data + '/' + name for name in os.listdir(folder_experiment_data) if
                       any(name.startswith(patient) for patient in testing_patients)])

    names = sorted([name for name in os.listdir(folder_experiment_data) if
                    any(name.startswith(patient) for patient in testing_patients)])

    subjects_original_data = sorted(
        [folder_experiment_data + '/' + name for name in os.listdir(folder_experiment_data) if
         any(name.startswith(patient) for patient in testing_patients)])

    # load the model
    net = torch.load(f'{folder_experiment}/trained_{model}.pth')
    net.eval()

    # predict the parameter maps for each patient and save them
    for j in range(len(subjects)):

        # real data experiment
        data = nib.load(f'{subjects[j]}/real_signal_with_mask.nii.gz')

        signal = data.get_fdata()
        mask = nib.load(f'{subjects_original_data[j]}/mask.nii.gz').get_fdata()

        # transform data in 2d tensor with voxels and values for corresponding bvalues
        sx, sy, sz, n_b_values = signal.shape
        signal4d = signal
        signal = np.reshape(signal, (sx * sy * sz, n_b_values))

        # same for mask
        mx, my, mz = mask.shape
        mask_reshaped = np.reshape(mask, (mx * my * mz))

        # select voxels of the brain
        valid_id = mask_reshaped == 1
        datatot2 = signal[valid_id, :]

        # normalize data
        datatot2 = datatot2 / datatot2.max()

        # predict parameters
        paramsNN = deep.predict_IVIM(datatot2, bvalues_all, TEs, net, arg)

        names_parameters = ['D', 'f', 'Dp', 'T2t', 'S0'] if "T2_IVIM_NET_seg_fix" in model else ['D', 'f', 'Dp', 'T2p', 'T2t', 'S0']

        if model == "IVIM_NET":
            names_parameters = ['D', 'f', 'Dp', 'S0']

        folder_patient = f'{folder_experiment}/{names[j]}'
        tot = 0

        print(f"***SAVE PARAMETER DISTRIBUTION MAPS***\n")
        # fill image array and make nifti
        for k in range(len(names_parameters)):
            img = np.zeros([sx * sy * sz])
            img[valid_id] = paramsNN[k][tot:(tot + sum(valid_id))]
            img = np.reshape(img, [sx, sy, sz])

            # save parameter maps of the test patients
            nib.save(nib.Nifti1Image(img, data.affine, data.header),
                     f'{folder_patient}/sim{sim}_{names_parameters[k]}_{model}.nii.gz')

        print(f"***CALCULATING THE CORRELATIONS BETWEEN PARAMETERS FOR PATIENT {names[j]}***\n")
        if calculate_cor:
            if model != "IVIM_NET":
                corr_matrix = calculate_correlation_matrix(model, paramsNN[0], paramsNN[1], paramsNN[2], paramsNN[3]) if "T2_IVIM_NET_seg_fix" in model else calculate_correlation_matrix(
                    model, paramsNN[0], paramsNN[1], paramsNN[2], paramsNN[4], paramsNN[3])
                calculate_cor = False
            else:
                corr_matrix = calculate_correlation_matrix(model, paramsNN[0], paramsNN[1], paramsNN[2], 0)
                calculate_cor = False


    return paramsNN, corr_matrix

