"""
In this file the synthesized signal with noise is made.
"""

import os
import nibabel as nib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_syn_signal(sigma, bvalues_all, TEs, subjects_experiment_folder, subjects_study_folder, names, sim):

    for g,patient in enumerate(subjects_experiment_folder):
        print('processing',names[g])
        #get groundtruth and if the value is nan is is replaced with 0 (happens very rarely)
        D = np.nan_to_num(nib.load(f'{patient}/sim{sim}_D_groundtruth_IVIM_NET.nii.gz').get_fdata(), nan=0)
        f = np.nan_to_num(nib.load(f'{patient}/sim{sim}_f_groundtruth_IVIM_NET.nii.gz').get_fdata(), nan=0)
        Dstar = np.nan_to_num(nib.load(f'{patient}/sim{sim}_Dp_groundtruth_IVIM_NET.nii.gz').get_fdata(), nan = 0)
        T2p = np.nan_to_num(nib.load(f'{patient}/sim{sim}_T2p_groundtruth_IVIM_NET.nii.gz').get_fdata(), nan=0)
        T2t = np.nan_to_num(nib.load(f'{patient}/sim{sim}_T2t_groundtruth_IVIM_NET.nii.gz').get_fdata(), nan=0)

        #add mask
        mask = nib.load(f'{subjects_study_folder[g]}/mask.nii.gz').get_fdata()
        #first and last slice are empty
        mask[:, :, 0] = 0
        mask[:, :, -1] = 0  # first and last slice are empty

        #need to import from another place since I cant save the original data on github
        signal = nib.load(f'{subjects_study_folder[g]}/real_signal_with_mask.nii.gz').get_fdata()

        signal = signal[:, :, :, :52]

        #normalize signal
        signal = signal/signal.max()
        S0 = signal[:, :, :, 0]

        #make synthesized signal
        Sforward = np.zeros((D.shape[0], D.shape[1], D.shape[2], bvalues_all.shape[0]))
        for i in range(len(bvalues_all) - 1):
            b = bvalues_all[i]
            te = TEs[i]
            tmp = S0 * ((f * np.exp(-te/T2p) * np.exp(-b * Dstar)) + ((1 - f) * np.exp(-te/T2t) * np.exp(-b * D)))

            tmp = tmp * mask
            Sforward[..., i] = tmp

        #add noise to synthesized signal
        S_noise = np.zeros((D.shape[0], D.shape[1], D.shape[2], bvalues_all.shape[0]))
        for i in range(len(bvalues_all) - 1):
            S_bvalues = Sforward[:,:,:,i]

            #add noise
            gaussian_noise = np.random.normal(0, sigma, size=S_bvalues.shape)
            noisy_signal = S_bvalues + gaussian_noise
            noisy_signal = np.abs(noisy_signal)
            noisy_signal = noisy_signal * mask
            S_noise[..., i] = noisy_signal

        #save synthesized signal and synthesized signal with noise
        nib.save(nib.Nifti1Image(Sforward, np.eye(4)), f'{patient}/sim{sim}_synthesized_signal.nii.gz')
        nib.save(nib.Nifti1Image(S_noise, np.eye(4)), f'{patient}/sim{sim}_synthesized_signal_noise.nii.gz')

