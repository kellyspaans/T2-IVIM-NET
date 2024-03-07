"""
12-12-2023
K.E. Spaans

Test script for checking the data for IVIM parameter estimation. Histogram is plotted of the subtracted signal values of voxels
with TE=76 and TE=60.
"""

import time
import numpy as np
import os
import sys
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def subtraction_histogram(data, TEs, bvalues_all, values=[110, 76], title_add='', S0=None):

    value_1 = values[0] / 1000
    value_2 = values[1] / 1000

    # retrieve values of according TEs
    te_index_1 = np.where(TEs == value_1)[0]
    te_index_2 = np.where(TEs == value_2)[0]
    print(f"index checker for TE={value_1}: {te_index_1}")

    if len(data.shape) == 2:
        data_1 = data[:, te_index_1]
        data_2 = data[:, te_index_2]

    else:
        data_1 = data[:, :, :, te_index_1]
        data_2 = data[:, :, :, te_index_2]

    print(f"data shape is reduced from {data.shape} to {data_1.shape}")

    # filter on b=0 values
    print(f"bvals for TE = {value_1}: {bvalues_all[te_index_1]}")
    indices_b0_1 = np.where(bvalues_all[te_index_1] == 0)[0]
    indices_b0_2 = np.where(bvalues_all[te_index_2] == 0)[0]

    print(f"npwhere for te={value_1} and b=0: {indices_b0_1}")
    print(f"npwhere for te={value_2} and b=0: {indices_b0_2}")

    if len(data.shape) == 2:
        data_1_b0 = data_1[:, indices_b0_1]
        data_2_b0 = data_2[:, indices_b0_2]

        # if S0 is not None:
        #     S0_b0 = data_2[:, indices_b0_2]

    else:
        data_1_b0 = data_1[:, :, :, indices_b0_1]
        data_2_b0 = data_2[:, :, :, indices_b0_2]

    print(f"data shape after filtering for b=0 (TE=76): {data_1_b0.shape}")
    print(f"data shape after filtering for b=0 (TE=60): {data_2_b0.shape}")

    # find the b value with highest overall signal intensity
    if len(data.shape) == 2:
        col_sum_1 = np.sum(data_1_b0, axis=0)
        col_sum_2 = np.sum(data_2_b0, axis=0)
        max_b0_1 = data_1_b0[:, np.argmax(col_sum_1)]
        max_b0_2 = data_2_b0[:, np.argmax(col_sum_2)]

    else:
        col_sum_1 = np.sum(data_1_b0, axis=(0, 1, 2))
        col_sum_2 = np.sum(data_2_b0, axis=(0, 1, 2))

        max_b0_1 = data_1_b0[:, :, :, np.argmax(col_sum_1)]
        max_b0_2 = data_2_b0[:, :, :, np.argmax(col_sum_2)]

    print(f"index for max b0 value (TE={value_1}): {np.argmax(col_sum_1)}")
    print(f"resulting shape {max_b0_1.shape}")

    # histogram
    subtraction_result = max_b0_2 - max_b0_1
    subtraction_result = subtraction_result.flatten()
    max_b0_1= max_b0_1.flatten()
    max_b0_2 = max_b0_2.flatten()

    # print(f"subtr result: {subtraction_result}")
    print(f"{value_1}: {max_b0_1}")
    print(f"{value_2}: {max_b0_2}")

    # separate positive and negative values
    positive_values = subtraction_result[subtraction_result >= 0]
    negative_values = subtraction_result[subtraction_result < 0]

    # Plot histograms with specified colors
    plt.figure(figsize=(12, 8))

    plt.subplot(1,2,1)
    plt.hist(max_b0_1, bins=50, color='blue', alpha=0.5, label=f'TE={value_1*1000}', edgecolor='white')
    plt.hist(max_b0_2, bins=50, color='orange', alpha=0.5, label=f'TE={value_2*1000}', edgecolor='white')
    # if S0 is not None:
    #     plt.scatter(S0, 0, color='purple', marker='o', label='S0')

    plt.title(f"Histogram: TE={value_1} (max val: {max_b0_1.max():.2f}, min val:{max_b0_1.min():.2f} and\nTE={value_2} (max val: {max_b0_2.max():.2f}, min val:{max_b0_2.min():.2f}) where b=0, {title_add}", fontsize=8)
    plt.xlim()
    plt.ylim()
    plt.legend()

    plt.subplot(1,2,2)
    plt.hist(positive_values, bins=40, color='green', alpha=0.7, label='Positive', edgecolor='white')
    plt.hist(negative_values, bins=40, color='red', alpha=0.7, label='Negative', edgecolor='white')
    plt.title(f'Subtract. Hist.: TE={value_2}\nwith b=0 - TE={value_1} with b=0, {title_add}', fontsize=8)
    plt.xlabel('Signal Intensity Difference')
    plt.ylabel('Frequency')
    plt.xlim()
    plt.ylim()
    plt.legend()

    plt.show()

def signal_map(data, TEs, bvalues_all, values=[110, 76], title_add=''):
    value_1 = values[0] / 1000
    value_2 = values[1] / 1000

    # retrieve values of according TEs
    te_index_1 = np.where(TEs == value_1)[0]
    te_index_2 = np.where(TEs == value_2)[0]
    print(f"index checker for TE={value_1}: {te_index_1}")

    if len(data.shape) == 2:
        data_1 = data[:, te_index_1]
        data_2 = data[:, te_index_2]

    else:
        data_1 = data[:, :, :, te_index_1]
        data_2 = data[:, :, :, te_index_2]

    print(f"data shape is reduced from {data.shape} to {data_1.shape}")

    # filter on b=0 values
    print(f"bvals for TE = {value_1}: {bvalues_all[te_index_1]}")
    indices_b0_1 = np.where(bvalues_all[te_index_1] == 0)[0]
    indices_b0_2 = np.where(bvalues_all[te_index_2] == 0)[0]

    print(f"npwhere for te={value_1} and b=0: {indices_b0_1}")
    print(f"npwhere for te={value_2} and b=0: {indices_b0_2}")

    if len(data.shape) == 2:
        data_1_b0 = data_1[:, indices_b0_1]
        data_2_b0 = data_2[:, indices_b0_2]

    else:
        data_1_b0 = data_1[:, :, :, indices_b0_1]
        data_2_b0 = data_2[:, :, :, indices_b0_2]

    print(f"data shape after filtering for b=0 (TE=76): {data_1_b0.shape}")
    print(f"data shape after filtering for b=0 (TE=60): {data_2_b0.shape}")

    # find the b value with highest overall signal intensity
    if len(data.shape) == 2:
        col_sum_1 = np.sum(data_1_b0, axis=0)
        col_sum_2 = np.sum(data_2_b0, axis=0)
        max_b0_1 = data_1_b0[:, np.argmax(col_sum_1)]
        max_b0_2 = data_2_b0[:, np.argmax(col_sum_2)]

    else:
        col_sum_1 = np.sum(data_1_b0, axis=(0, 1, 2))
        col_sum_2 = np.sum(data_2_b0, axis=(0, 1, 2))

        max_b0_1 = data_1_b0[:, :, :, np.argmax(col_sum_1)]
        max_b0_2 = data_2_b0[:, :, :, np.argmax(col_sum_2)]

    print(f"index for max b0 value (TE={value_1}): {np.argmax(col_sum_1)}")
    print(f"resulting shape {max_b0_1.shape}")

    subtraction_result = max_b0_2 - max_b0_1

    if len(data.shape) == 2:
        subtraction_result = np.reshape(subtraction_result, (112,112,48))

    slices = [3, 11, 25, 33]

    cmap = ListedColormap(['green', 'white', 'red'])

    fig, axes = plt.subplots(2, 2, figsize=(15,10))

    for i, slice in enumerate(slices, start=1):

        plt.subplot(2, 2, i)
        slice_data = subtraction_result[:, :, slice]
        im = plt.imshow(slice_data, cmap='RdBu', vmin=-np.max(np.abs(subtraction_result)), vmax=np.max(np.abs(subtraction_result)))
        plt.title(f"Slice nr {slice}")
        plt.axis('off')
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Intensity', rotation=270, labelpad=15)

    plt.suptitle(title_add)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # get names folders and make necessary folder structure
    maindir='/data/projects/followup-NOIV/data/proc'
    files = 'NOIV'
    no_data = []
    subjects = []
    subjects_study_folder = []
    subjects_experiment_folder = []
    names = []

    # we'll check the availability of these files for the selected subjects
    needed_file1 = "dwi_final_notdenoised.nii.gz"
    needed_file2 = "dwi_final_notdenoised_brain_mask.nii.gz"

    # mainfolder
    folder = '/data/projects/followup-NOIV/students/kespaans/data/Experiments'

    # save paths to study / experiment data
    folder_experiment_data = f'{folder}/T2/DWI_data'
    folder_experiment = f'{folder}/T2/TEST'

    # select subjects for experiment
    for name in os.listdir(maindir):
        # if name.startswith(files):
        if name.startswith('NOIV_10507'): #or name.startswith('NOIV_10169') or name.startswith('NOIV_10220') or name.startswith('NOIV_10374'): # FOR QUICK RUNNNING THIS IS CHANGED TEMPORARILY

            dir_to_data = maindir + '/' + name + '/DWI'

            if os.path.exists(os.path.join(dir_to_data, needed_file1)) and os.path.exists(os.path.join(dir_to_data, needed_file2)):
                print(f'The files {needed_file1} and {needed_file2} exist in the directory for subject {name}. Will be included in experiment.')
                subjects.append(maindir + '/' + name)
                names.append(name)

                # this is the general study folder / directory
                directory_save = f'{folder_experiment_data}/{name}'
                subjects_study_folder.append(f'{folder_experiment_data}/{name}')
                if not os.path.exists(directory_save):
                    os.makedirs(directory_save)

                # this is the folder / directory for the specific experiment
                directory_save2 = f'{folder_experiment}/{name}'
                subjects_experiment_folder.append(f'{folder_experiment}/{name}')
                if not os.path.exists(directory_save2):
                    os.makedirs(directory_save2)

            else:
                print(f'The files {needed_file1} or {needed_file2} do not exist in the directory for subject {name}. Will not be included in experiment.')
                no_data.append(name)

    print(f"subject name list: {names}")

    # assign val/test patient
    testing_patients = ['NOIV_10169', 'NOIV_10220']
    validation_patient = 'NOIV_10507'

    # assign train patient of which figures are shown for comparison
    one_training_patient = 'NOIV_10148'

    # figures of slice of training/validation patients
    slice = '23'
    training_slice = f'{one_training_patient} {slice}'
    validation_slice = f'{validation_patient} {slice}'

    # get b-values for 60, 76, and 110 TE
    b_values = np.genfromtxt(f'{subjects[0]}/DWI/dwi_final_notdenoised.bval')
    TEs = np.genfromtxt(f'{subjects[0]}/DWI/dwi_final_notdenoised.te')

    bvalues_all = np.array(b_values)
    TEs = np.array(TEs) / 1000

    # bvalues close to 0 are set to 0 otherwise IVIM net does not work
    bvalues_all[bvalues_all < 0.01] = 0

    # create dictionaries to store b values for each TE
    b_values_dict = {}
    TEs_dict = {}

    for TE in np.unique(TEs):

        # mask b values based on the position of TE
        mask = (TEs == TE)
        b_values_for_TE = bvalues_all[mask]

        # store values in dictionaries
        TEs_dict[TE*1000] = b_values_for_TE
        b_values_dict[f'bvalsTE{int(TE)}'] = b_values_for_TE


    # get the normalised signal with mask
    print('*****GET REAL SIGNAL*****')

    for g, patient in enumerate(subjects):
        print('processing', names[g])
        # get real signal and signal where the mask is already applied
        signal = nib.load(f'{patient}/DWI/dwi_final_notdenoised.nii.gz').get_fdata()

        subtraction_histogram(signal, TEs, bvalues_all, values=[76, 60], title_add='dwi_final_notdenoised', S0=None)
        signal_map(signal, TEs, bvalues_all, title_add='dwi_final_notdenoised')

        print(f"signal shape: {signal.shape}")

        # add mask
        mask = nib.load(f'{patient}/DWI/dwi_final_notdenoised_brain_mask.nii.gz').get_fdata()
        for x in range(signal.shape[3]):
            signal[:, :, :, x] = signal[:, :, :, x] * mask

        subtraction_histogram(signal, TEs, bvalues_all, [76, 60], 'after mask application')
        signal_map(signal, TEs, bvalues_all, title_add='after mask application')

        # normalize
        signal = signal / signal.max()

        subtraction_histogram(signal, TEs, bvalues_all, [76, 60], 'normalized and saved as real_signal_with_mask')
        signal_map(signal, TEs, bvalues_all, title_add='normalized and saved as real_signal_with_mask')


        # print(f'signal shape: {signal.shape}')
        # print(f"signal type {type(signal)}")

        # name of the dir where we save the data
        directory_save = f'{folder_experiment_data}/{names[g]}'

        # save real signal
        # nib.save(nib.Nifti1Image(real_signal, np.eye(4)), f'{directory_save}/real_signal.nii.gz')
        # save normalized signal with mask
        nib.save(nib.Nifti1Image(signal, np.eye(4)), f'{directory_save}/real_signal_with_mask.nii.gz')
        # save mask
        nib.save(nib.Nifti1Image(mask, np.eye(4)), f'{directory_save}/mask.nii.gz')

    # select all subjects
    subjects = subjects_study_folder

    # combine all patients and transform it to 2d tensor of the voxels and their values for the corresponding b-values
    signals = np.array([])
    masks = np.array([])

    for g, patient in enumerate(subjects):
        print('processing',names[g])

        signal = nib.load(f'{patient}/real_signal_with_mask.nii.gz').get_fdata()
        mask = nib.load(f'{subjects_study_folder[g]}/mask.nii.gz').get_fdata()

        sx, sy, sz, n_b_values = signal.shape
        subtraction_histogram(signal, TEs, bvalues_all, [76, 60], f'retrieved from real_signal_with_mask again, p{g}')

        # nifti_img = nib.Nifti1Image(signal, affine=np.eye(4))
        # nib.save(nifti_img, '/data/projects/followup-NOIV/students/kespaans/data/Experiments/T2/test_image.nii.gz')

        signal_reshaped = np.reshape(signal, (sx * sy * sz, n_b_values))
        mx, my, mz = mask.shape
        mask_reshaped = np.reshape(mask, (mx * my * mz))

        if g == 0:
            signals = np.append(signals, signal_reshaped)
            signals = np.reshape(signals, (sx * sy * sz, n_b_values))
            masks = np.append(masks, mask_reshaped)
            masks = np.reshape(masks, (mx * my * mz))
            print(f"signal shape for first patient: {signals.shape}")
            # dwi_data_4d = signals.reshape((112, 112, 48, 90))
            # nifti_img = nib.Nifti1Image(dwi_data_4d, affine=np.eye(4))
            # nib.save(nifti_img, '/data/projects/followup-NOIV/students/kespaans/data/Experiments/T2/test_image.nii.gz')

        if g>0:
            signals = np.vstack((signals, signal_reshaped))
            masks = np.hstack((masks, mask_reshaped))
            print(f"total signal shape after adding next patient: {signals.shape}")

    # retrieve valid voxels of the brain
    valid_id = masks == 1
    datatot = signals[valid_id, :]
    # datatot = signals

    print(f"datatot shape; {datatot.shape}")
    subtraction_histogram(datatot, TEs, bvalues_all, [76, 60], 'after mask application (datatot)')

    # normalize the data
    datatot = datatot/datatot.max()

    print('Patient data loaded\n')

    # remove NaN
    res = [i for i, val in enumerate(datatot != datatot) if not val.any()]

    print(f"bval: {bvalues_all.shape}")
    print(f"TEs: {TEs.shape}")

    data = datatot[res]

    print(f"data: {data.shape}")
    print(f"data raw; {data}")