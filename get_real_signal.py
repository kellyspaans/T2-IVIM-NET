"""
In this file we get the DWI data, normalize it, apply a mask on it and save it.
"""

from pathlib import Path
import torch
import nibabel as nib
import numpy as np
from check_data import subtraction_histogram
from check_data import signal_map
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

def real_signal(folder_experiment_data, subjects, names):
    """
    This function takes the directory of the original data of the subjects.
    It extracts the data for the subjects, normalizes it, and applies a mask.
    The new data is saved within the experiment folder. 
    """

    for g,patient in enumerate(subjects):
        print('processing',names[g])
        # get real signal and signal where the mask is already applied
        # real_signal = nib.load(f'{patient}/DWI/dwi_final_denoised.nii.gz').get_fdata()
        signal = nib.load(f'{patient}/DWI/dwi_final_notdenoised.nii.gz').get_fdata()

        #add mask
        mask = nib.load(f'{patient}/DWI/dwi_final_notdenoised_brain_mask.nii.gz').get_fdata()
        for x in range(signal.shape[3]):
            signal[:,:,:,x] = signal[:,:,:,x] * mask

        #normalize
        signal = signal / signal.max()

        #name of the dir where we save the data
        directory_save = f'{folder_experiment_data}/{names[g]}'

        #save real signal
        # nib.save(nib.Nifti1Image(real_signal, np.eye(4)), f'{directory_save}/real_signal.nii.gz')
        #save normalized signal with mask
        nib.save(nib.Nifti1Image(signal, np.eye(4)), f'{directory_save}/real_signal_with_mask.nii.gz')
        #save mask
        nib.save(nib.Nifti1Image(mask, np.eye(4)), f'{directory_save}/mask.nii.gz')
