"""
K.E. Spaans 2024

This script contains code for training the T2-IVIM-NET model on in vivo patient data.
"""

import nibabel as nib
import numpy as np
import IVIMNET_model.IVIMNET.deep_T2 as deep
import torch
from IVIMNET_model.hyperparams import hyperparams as hp
from IVIMNET_model.hyperparams_orig import hyperparams as hp_orig
from IVIMNET_model.hyperparams_seg import hyperparams as hp_seg
from IVIMNET_model.hyperparams_seg_fix_100 import hyperparams as hp_seg_fix_100
from IVIMNET_model.hyperparams_seg_fix_150 import hyperparams as hp_seg_fix_150


def train_IVIM_in_vivo(model, bvalues_all, TEs, testing_patients, folder_experiment, subjects_study_folder, subjects_experiment_folder, names):
    '''
    This function takes the following variables as input:
    - model: model to be trained
    - bvalues_all: array of TE values
    - TEs (array): array of TE values (same order as b values)
    - testing_patients (str)
    - folder_experiment (str)
    - subjects_study_folder (str)
    - subjects_experiment_folder (str)
    - names (str)
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
        arg = hp()
        arg = deep.checkarg(arg)
        print("TRAINING BI-EXP\n")

    # select subjects except for testing patients
    subjects = [x for x in subjects_study_folder if all(item not in x for item in testing_patients)]
    names = [x for x in names if all(item not in x for item in testing_patients)]


    # save paths of patients within study folder (where real_signal, real_signal_mask, etc. is saved)
    subjects_study_folder = [x for x in subjects_study_folder if all(item not in x for item in testing_patients)]

    # combine all patients and transform it to 2d tensor of the voxels and their values for the corresponding b-values
    signals = np.array([])
    masks = np.array([])

    for g, patient in enumerate(subjects):
        print('processing', names[g])
        signal = nib.load(f'{patient}/real_signal_with_mask.nii.gz').get_fdata()
        mask = nib.load(f'{subjects_study_folder[g]}/mask.nii.gz').get_fdata()

        sx, sy, sz, n_b_values = signal.shape
        signal_reshaped = np.reshape(signal, (sx * sy * sz, n_b_values))
        mx, my, mz = mask.shape
        mask_reshaped = np.reshape(mask, (mx * my * mz))

        if g == 0:
            signals = np.append(signals, signal_reshaped)
            signals = np.reshape(signals, (sx * sy * sz, n_b_values))
            masks = np.append(masks, mask_reshaped)
            masks = np.reshape(masks, (mx * my * mz))
        if g>0:
            signals = np.vstack((signals, signal_reshaped))
            masks = np.hstack((masks, mask_reshaped))

        # retrieve valid voxels of the brain
        valid_id = masks == 1
        datatot = signals[valid_id, :]

        # normalize the data
        datatot = datatot/datatot.max()

    print('Patient data loaded\n')

    # NETWORK TRAINING
    print('NN fitting\n')

    # remove NaN
    res = [i for i, val in enumerate(datatot != datatot) if not val.any()]

    data = datatot[res]

    # train network
    net = deep.learn_IVIM(data, bvalues_all, TEs, arg)

    # save the model
    torch.save(net, f'{folder_experiment}/trained_{model}.pth')

