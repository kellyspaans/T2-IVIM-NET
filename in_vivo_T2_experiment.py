"""
K.E. Spaans 2024
Code for T2-IVIM parameter estimation based on simulated signal.

Script should run with the following arguments:
in_vivo_T2_experiment.py experiment_name n_sims skip
- experiment_name: name of the experiment
- n_sims: number of simulations
- skip: argument to skip the training phase if trained model is already saved. If 1, training is skipped; if 0, training is not skipped.
"""

import numpy as np
import os
import sys
import IVIMNET_model.IVIMNET.deep_T2 as deep
from train_IVIM_in_vivo_data import train_IVIM_in_vivo
from test_IVIM_in_vivo_data import test_IVIM_in_vivo
from test_IVIM_in_vivo_data import test_IVIM_in_vivo
from IVIMNET_model.hyperparams_orig import hyperparams as hp_orig
from IVIMNET_model.hyperparams_seg import hyperparams as hp_seg
from IVIMNET_model.hyperparams_seg_fix_100 import hyperparams as hp_seg_fix_100
from IVIMNET_model.hyperparams_seg_fix_150 import hyperparams as hp_seg_fix_150
from visualization import distribution_plot, parameter_distribution_plot_updated
from get_real_signal import real_signal
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from analysis import calculate_nrmse_signals
import pandas as pd

experiment = sys.argv[1]
n_sims = int(sys.argv[2])
skip = sys.argv[3]

# define models to be tested
models = ["T2_IVIM_NET_orig", "T2_IVIM_NET_seg", "T2_IVIM_NET_seg_fix_100", "T2_IVIM_NET_seg_fix_150"]

print(
    f"*** Running experiment ({experiment}) with in vivo data ***\n")

# get names folders and make necessary folder structure
maindir = '/data/projects/followup-NOIV/data/proc'
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
folder_experiment = f'{folder}/T2/{experiment}'

# assign test patient
testing_patients = ['NOIV_10169']

# select subjects for experiment
for name in os.listdir(maindir):
    if name.startswith(files):
        dir_to_data = maindir + '/' + name + '/DWI'

        if os.path.exists(os.path.join(dir_to_data, needed_file1)) and os.path.exists(
                os.path.join(dir_to_data, needed_file2)):
            print(
                f'The files {needed_file1} and {needed_file2} exist in the directory for subject {name}. Will be included in experiment.')
            subjects.append(maindir + '/' + name)
            names.append(name)

            # this is the general study folder / directory
            directory_save = f'{folder_experiment_data}/{name}'
            subjects_study_folder.append(f'{folder_experiment_data}/{name}')
            if not os.path.exists(directory_save) and name in testing_patients:
                os.makedirs(directory_save)

            # this is the folder / directory for the specific experiment
            directory_save2 = f'{folder_experiment}/{name}'
            subjects_experiment_folder.append(f'{folder_experiment}/{name}')
            if not os.path.exists(directory_save2):
                os.makedirs(directory_save2)

        else:
            print(
                f'The files {needed_file1} or {needed_file2} do not exist in the directory for subject {name}. Will not be included in experiment.')
            no_data.append(name)

# create folder for created figures
folder_fig = f'{folder_experiment}/figures'

if not os.path.exists(folder_fig):
    os.makedirs(folder_fig)
else:
    print(f"figures map already exists")

# get the normalised signal with mask
print('*****GET REAL SIGNAL*****\n')
real_signal(folder_experiment_data, subjects, names)

# get b-values for 60, 76, and 110 TE
b_values = np.genfromtxt(f'{subjects[0]}/DWI/dwi_final_notdenoised.bval')
TEs = np.genfromtxt(f'{subjects[0]}/DWI/dwi_final_notdenoised.te')

bvalues_all = np.array(b_values)
TEs = np.array(TEs) / 1000

# bvalues close to 0 are set to 0 otherwise IVIM-NET does not work
bvalues_all[bvalues_all < 0.01] = 0

# create dictionaries to save test results
nrmse_data = {model: [] for model in models}
corr_matrices = {model: [] for model in models}

all_params = ['D', 'f', 'Dp', 'T2p', 'T2t']
fixed_params = ['D', 'f', 'Dp', 'T2t']  # for the T2_IVIM_NET_seg_fix models
all_model_predictions = {param: {model: [] for model in models} for param in all_params}

# *** SIMULATIONS ***
for sim in range(n_sims):

    for model in models:
        print(f"***RUNNING MODEL {model}, SIMULATION {sim}***\n")

        # get according hyperparameters for the model used
        if model == "T2_IVIM_NET_orig":
            arg = hp_orig()
            arg = deep.checkarg(arg)
            arg.train_pars.folderfig = folder_fig
        elif model == "T2_IVIM_NET_seg":
            arg = hp_seg()
            arg = deep.checkarg(arg)
            arg.train_pars.folderfig = folder_fig
        elif model == "T2_IVIM_NET_seg_fix_100":
            arg = hp_seg_fix_100()
            arg = deep.checkarg(arg)
            arg.train_pars.folderfig = folder_fig
        elif model == "T2_IVIM_NET_seg_fix_150":
            arg = hp_seg_fix_150()
            arg = deep.checkarg(arg)
            arg.train_pars.folderfig = folder_fig
        else:
            print("PLEASE SPECIFY MODEL\n")

        if skip == '0':
            print(f"***TRAINING {model}***\n")
            # train T2-IVIM-NET model
            train_IVIM_in_vivo(model, bvalues_all, TEs, testing_patients, folder_experiment, subjects_study_folder, subjects_experiment_folder, names)
        else:
            print(f"***TRAINING IS SKIPPED***\n")

        print(f"***EVALUATE MODEL ON TEST PATIENTS***\n")
        paramsNN, corr_matrix = test_IVIM_in_vivo(model, sim, testing_patients, bvalues_all, TEs, folder_experiment, folder_experiment_data)

        corr_matrices[model].append(corr_matrix)

        if sim == 0:
            if "T2_IVIM_NET_seg_fix" in model:
                for i, param in enumerate(fixed_params):
                    all_model_predictions[param][model] = paramsNN[i]
            else:
                for i, param in enumerate(all_params):
                    all_model_predictions[param][model] = paramsNN[i]

        print(f"***CALCULATE NRMSE BETWEEN PREDICTED SIGNAL AND ACTUAL SIGNAL***\n")
        nrmse_signals = calculate_nrmse_signals(folder_experiment_data, folder_experiment, testing_patients, sim, TEs,
                                                bvalues_all, model)
        nrmse_data[model].append(nrmse_signals)

    if sim == 0:
        # save parameter distribution plot of predictions based on testing patients
        parameter_distribution_plot_updated(all_model_predictions, 0, folder_fig)


print(f"***SAVE NRMSE IN CSV FILES***\n")
nrmse_df = pd.DataFrame(nrmse_data)
nrmse_df['sim'] = [f'sim {i + 1}' for i in range(n_sims)]
nrmse_df.set_index('sim', inplace=True)
nrmse_csv = nrmse_df.to_csv(f"{folder_experiment}/nrmse_signals_in_vivo.csv")

print(f"***CALCULATE MEAN AND MEDIAN OVER ALL ACQUIRED CORRELATION MATRICES***\n")

model_names = ['T$_2$-IVIM-NET$_{orig}$', 'T$_2$-IVIM-NET$_{seg}$', 'T$_2$-IVIM-NET$_{seg\_{fix}\_{100}}$', 'T$_2$-IVIM-NET$_{seg\_{fix}\_{150}}$']

for i, model in enumerate(models):
    # convert each correlation matrix to numpy array
    corr_matrices_array = [np.array(matrix) for matrix in corr_matrices[model]]

    # stacking and computing the median
    stacked_matrices = np.stack(corr_matrices_array, axis=-1)
    median_matrix = np.median(stacked_matrices, axis=-1)

    # save correlation matrix as a figure
    labels = ['$D$', '$f$', '$D*$', '$T_{2p}$', '$T_{2t}$']

    plt.figure(figsize=(8, 6))
    sns.heatmap(median_matrix, annot=True, cmap='coolwarm', xticklabels=labels, yticklabels=labels)
    plt.title(f'{model}')
    plt.savefig(f"{folder_fig}/corr_{model}.png")





