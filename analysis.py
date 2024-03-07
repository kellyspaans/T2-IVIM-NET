"""
K.E. Spaans 2024
This is code containing the necessary functions for analysis of the experiments.
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import random as rd
import nibabel as nib


def nrmse(true, predicted):
    rmse = np.sqrt(np.mean(np.square(np.subtract(true, predicted))))
    return rmse / (np.max(true) - np.min(true))

def calculate_nrmse_params(D, f, Dp, T2t, params, T2p=None):
    """
    Function that calculates the NRMSE over all parameters, based on the predicted
    parameters and the actual generated parameter values.
    input:
    - D: true value of D
    - f: true value of f
    - Dp: true value of Dp
    - T2t: true value of T2t
    - params: predicted parameters
    - T2p: true value of T2p, default is None (in case of fixed parameter model)
    """

    nrmse_D = nrmse(D, params[0])
    nrmse_f = nrmse(f, params[1])
    nrmse_Dp = nrmse(Dp, params[2])

    if T2p is not None:
        nrmse_T2p = nrmse(T2p, params[3])
        nrmse_T2t = nrmse(T2t, params[4])
        return np.array([nrmse_D, nrmse_f, nrmse_Dp, nrmse_T2p, nrmse_T2t])
    else:
        nrmse_T2t = nrmse(T2t, params[3])
        return np.array([nrmse_D, nrmse_f, nrmse_Dp, nrmse_T2t])


def T2_IVIM_model(model, bvalues, TEs, params, S0):
    """
    Function that calculates the estimated signal.
    """

    bvalues = np.reshape(bvalues, (1, 90, 1, 1))
    TEs = np.reshape(TEs, (1, 90, 1, 1))

    if "T2_IVIM_NET_seg_fix" in model:
        D, f, Dp, T2t = np.split(params, 4, axis=1)
        if model == "T2_IVIM_NET_seg_fix_100":
            T2p = np.full_like(D, 0.1)
        else:
            T2p = np.full_like(D, 0.15)
    else:
        D, f, Dp, T2p, T2t = np.split(params, 5, axis=1)

    S = S0 * ((f * np.exp(-bvalues * Dp) * np.exp(-TEs / T2p)) + ((1-f) * np.exp(-bvalues * D) * np.exp(-TEs / T2t)))

    return S

def calculate_nrmse_signals(subjects_original_data, folder, patients, sim, TEs, b_values, model):
    """
    Function that calculates the NRMSE based on the estimated signal and the true signal within
    the patients.
    """
    if model in ["T2_IVIM_NET_seg_fix_100", "T2_IVIM_NET_seg_fix_150"]:
        parameters = ['D', 'f', 'Dp', 'T2t']
    elif model == "IVIM_NET":
        parameters = ['D', 'f', 'Dp']
    else:
        parameters = ['D', 'f', 'Dp', 'T2p', 'T2t']

    NRMSE_brain_patients = np.array([])


    for p, patient in enumerate(patients):
        print("processing", patient)
        mask = nib.load(f'{subjects_original_data}/{patient}/mask.nii.gz').get_fdata()
        mask = np.moveaxis(mask, -1, 0)

        kappa = np.zeros((mask.shape[0], len(parameters), mask.shape[1], mask.shape[2]))

        for counter, param in enumerate(parameters):
            result = nib.load(f'{folder}/{patient}/sim{sim}_{param}_{model}.nii.gz').get_fdata()
            result = np.moveaxis(result, -1, 0)
            kappa[:, counter, :, :] = result.squeeze()

        signal = nib.load(f'{subjects_original_data}/{patient}/real_signal_with_mask.nii.gz').get_fdata()
        signal = np.moveaxis(signal, (-1, -2), (0, 1))

        # NRMSE calculation
        S0_76 = signal[0].reshape(signal.shape[1], 1, signal.shape[2], signal.shape[3])
        S0_76 = np.repeat(S0_76, 53, axis=1)
        S0_110 = signal[52].reshape(signal.shape[1], 1, signal.shape[2], signal.shape[3])
        S0_110 = np.repeat(S0_110, 31, axis=1)
        S0_60 = signal[83].reshape(signal.shape[1], 1, signal.shape[2], signal.shape[3])
        S0_60 = np.repeat(S0_60, 6, axis=1)

        S0 = np.concatenate((S0_76, S0_110, S0_60), axis=1)

        S_predicted = T2_IVIM_model(model, b_values, TEs, kappa, S0)
        S_predicted = np.moveaxis(S_predicted, 0, 1)
        brain_signal = signal[:, mask == 1]
        NRMSE_brain = nrmse(brain_signal, S_predicted[:, mask == 1])
        NRMSE_brain_patients = np.append(NRMSE_brain_patients, NRMSE_brain) if NRMSE_brain_patients.size else NRMSE_brain

    return NRMSE_brain_patients


def calculate_correlation_matrix(model, D, f, Dp, T2t, T2p=None):
    """
    Calculates the correlation matrix for the parameters and signal and
    returns the correlation values.
    """

    if T2p is not None:
        data = {
            'D': D.squeeze(),
            'f': f.squeeze(),
            'Dp': Dp.squeeze(),
            'T2p': T2p.squeeze(),
            'T2t': T2t.squeeze(),
        }

    elif "T2_IVIM_NET_seg_fix" in model:
        data = {
            'D': D.squeeze(),
            'f': f.squeeze(),
            'Dp': Dp.squeeze(),
            'T2p': np.zeros(len(T2t.squeeze())), # will be zeros in the matrix
            'T2t': T2t.squeeze(),
        }

    elif model == "IVIM_NET":
        data = {
            'D': D.squeeze(),
            'f': f.squeeze(),
            'Dp': Dp.squeeze(),
            'T2p': np.zeros(len(Dp.squeeze())), # will be zeros in the matrix
            'T2t': np.zeros(len(Dp.squeeze())),
        }

    df = pd.DataFrame(data)

    corr_matrix = df.corr(method='spearman')

    return corr_matrix




