"""
K.E. Spaans 2024

Code for T2-IVIM parameter estimation based on simulated signal.

Script should be ran with the following arguments:
simulation_T2_experiment.py experiment_name n_sims SNR skip
- experiment_name: name of the experiment 
- n_sims: number of simulations
- SNR: Signal-to-Noise ratio. If 1, three levels of SNRs are tested for each simulation; if 0, no SNR is taken into account.
- skip: argument to skip the training phase if trained model is already saved. If 1, training is skipped; if 0, training is not skipped.
"""

from IVIMNET_model.IVIMNET.simulations import sim_signal
import numpy as np
import os
import sys
import IVIMNET_model.IVIMNET.deep_T2 as deep
from train_IVIM_sim_data import train_IVIM_sim
from test_IVIM_sim_data import test_IVIM_sim
from IVIMNET_model.hyperparams_orig import hyperparams as hp_orig
from IVIMNET_model.hyperparams_seg import hyperparams as hp_seg
from IVIMNET_model.hyperparams_seg_fix_100 import hyperparams as hp_seg_fix_100
from IVIMNET_model.hyperparams_seg_fix_150 import hyperparams as hp_seg_fix_150
from visualization import T2_sim_example_plot, distribution_plot, parameter_distribution_plot, parameter_distribution_plot_updated
from analysis import calculate_nrmse_params
import pandas as pd


def run_experiment_sim(experiment, n_sims, SNR, skip, cons_max=None, loss=None):

    print(
        f"*** Running experiment ({experiment}) with simulated data ***\n")

    # define models and SNR levels to be tested
    models = ["T2_IVIM_NET_orig", "T2_IVIM_NET_seg", "T2_IVIM_NET_seg_fix_100", "T2_IVIM_NET_seg_fix_150"]

    if SNR == '1':
        SNR_levels = [0, 20, 60, 100]
    elif SNR == '2':
        SNR_levels = [20, 60, 100]
    elif SNR == '20':
        SNR_levels = [20]
    else:
        SNR_levels = [0]

    # get names folders and make necessary folder structure
    maindir = '/data/projects/followup-NOIV/data/proc'

    # create folder for experiment
    folder = '/data/projects/followup-NOIV/students/kespaans/data/Experiments'
    folder_experiment = f'{folder}/T2/{experiment}'

    if not os.path.exists(folder_experiment):
        os.makedirs(folder_experiment)
    else:
        print(f"{experiment} map already exists")

    # create folder for created figures
    folder_fig = f'{folder_experiment}/figures'

    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)
    else:
        print(f"figures map already exists")

    # create a separate directory for each model
    model_folders = {}
    for model in models:
        model_folder = os.path.join(folder_experiment, model)
        model_folders[model] = model_folder
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

    # get b-values for 60, 76, and 110 TE
    b_values = np.genfromtxt(f'/data/projects/followup-NOIV/students/kespaans/data/b_and_TE_vals/b_values.bval') # file is copy of .../followup-NOIV/data/proc/NOIV_10538/DWI/dwi_final_notdenoised.bval
    TEs = np.genfromtxt(f'/data/projects/followup-NOIV/students/kespaans/data/b_and_TE_vals/TE_values.te') # file is copy of .../followup-NOIV/data/proc/NOIV_10538/DWI/dwi_final_notdenoised.te

    bvalues_all = np.array(b_values)
    TEs = np.array(TEs) / 1000

    # bvalues close to 0 are set to 0 otherwise IVIM net does not work
    bvalues_all[bvalues_all < 0.01] = 0

    # create dictionaries to store b values for each TE
    b_values_dict = {}
    TEs_dict = {}

    # initialize dictionaries for storing NRMSE values for each parameter and for storing data for the distribution plots
    all_params = ['D', 'f', 'Dp', 'T2p', 'T2t']
    fixed_params = ['D', 'f', 'Dp', 'T2t']  # for the T2_IVIM_NET_seg_fix model
    nrmse_data = {model: {snr: {param: [] for param in (fixed_params if "T2_IVIM_NET_seg_fix" in model else all_params)} for snr in SNR_levels} for model in models}
    predictions_around_true = {param: {model: [] for model in models} for param in all_params}
    all_model_predictions = {param: {model: [] for model in models} for param in all_params}

    for snr_level in SNR_levels:

        print("***SIMULATING DATA FOR TRAINING***\n")
        # simulate data  for training
        data_train, D_train, f_train, Dp_train, T2p_train, T2t_train = sim_signal(snr_level, bvalues_all, TEs, sims=1000000, Dmin=0.0005, Dmax=0.002,
                                                                                  fmin=0.1, fmax=0.5, Dsmin=0, Dsmax=0.2, fp2min=0, fp2max=0.3, Ds2min=0.2, Ds2max=0.5, T2pmin=0.02,
                                                                                T2pmax=0.15, T2tmin=0.04, T2tmax=0.12, rician=False, state=123, model='T2')

        # example curves -- delete later when running the experiment
        T2_sim_example_plot(bvalues_all, TEs, data_train, D_train, f_train, Dp_train, T2p_train, T2t_train, 1000, colors=['blue', 'orange', 'purple'])

        print("***SIMULATING DATA FOR TESTING***\n")
        # simulate data for testing
        data_test, D_test, f_test, Dp_test, T2p_test, T2t_test = sim_signal(snr_level, bvalues_all, TEs, sims=10000, Dmin=0.0005, Dmax=0.002,
                                                                                  fmin=0.1, fmax=0.5, Dsmin=0, Dsmax=0.2, fp2min=0, fp2max=0.3, Ds2min=0.2, Ds2max=0.5, T2pmin=0.02,
                                                                                T2pmax=0.15, T2tmin=0.04, T2tmax=0.12, rician=False, state=123, model='T2')

        for n in range(n_sims):

            for model in models:
                print(f"***RUNNING MODEL {model}, SNR LEVEL {snr_level}, SIMULATION {n}***")

                # get according hyperparameters for the model used
                if model == "T2_IVIM_NET_orig":
                    arg = hp_orig()
                    arg = deep.checkarg(arg)
                    if cons_max is not None:
                        arg.net_pars.cons_max = cons_max
                    if loss is not None:
                        arg.train_pars.loss_fun = loss
                elif model == "T2_IVIM_NET_seg":
                    arg = hp_seg()
                    arg = deep.checkarg(arg)
                    if cons_max is not None:
                        arg.net_pars.cons_max = cons_max
                    if loss is not None:
                        arg.train_pars.loss_fun = loss
                elif model == "T2_IVIM_NET_seg_fix_100":
                    arg = hp_seg_fix_100()
                    arg = deep.checkarg(arg)
                    if cons_max is not None:
                        if len(cons_max) == 6:
                            cons_max_fixed = cons_max[:-2] + cons_max[-1:] # exclude T2p boundary
                            arg.net_pars.cons_max = cons_max_fixed
                            print(f'arg.net_pars.cons_max:', arg.net_pars.cons_max)
                    if loss is not None:
                        arg.train_pars.loss_fun = loss
                elif model == "T2_IVIM_NET_seg_fix_150":
                    arg = hp_seg_fix_150()
                    arg = deep.checkarg(arg)
                    if cons_max is not None:
                        if len(cons_max) == 6:
                            cons_max_fixed = cons_max[:-2] + cons_max[-1:] # exclude T2p boundary
                        arg.net_pars.cons_max = cons_max_fixed
                    if loss is not None:
                        arg.train_pars.loss_fun = loss
                else:
                    print("PLEASE SPECIFY MODEL\n")


                if skip == '0':
                    print(f"***TRAINING {model}***\n")
                    # train T2-IVIM-NET model
                    train_IVIM_sim(model, data_train, bvalues_all, TEs, arg, folder_experiment)
                else:
                    print(f"***TRAINING IS SKIPPED***\n")

                print(f"***TESTING {model}***\n")
                # make predictions on test data
                paramsNN = test_IVIM_sim(model, data_test, bvalues_all, TEs, arg, folder_experiment)
                print(len(paramsNN))

                print(f"***COMPUTING NRMSE FOR {model}, SIM {n}, SNR={snr_level}***\n")
                # Determine the parameters to use based on the model
                current_params = fixed_params if "T2_IVIM_NET_seg_fix" in model else all_params

                NRMSE = calculate_nrmse_params(D_test, f_test, Dp_test, T2t_test, paramsNN, T2p=None) if "T2_IVIM_NET_seg_fix" in model else calculate_nrmse_params(D_test, f_test, Dp_test, T2t_test, paramsNN, T2p_test)
                for param_idx, param in enumerate(current_params):
                    nrmse_data[model][snr_level][param].append(NRMSE[param_idx])

                if n == 0 and snr_level == 0:
                    print(f"***CREATING DISTRIBUTION MAPS FOR VISUALIZATION***")

                    # first the distribution maps for test data
                    if "T2_IVIM_NET_seg_fix" in model:
                        for i, param in enumerate(fixed_params):
                            all_model_predictions[param][model] = paramsNN[i]
                    else:
                        for i, param in enumerate(all_params):
                            all_model_predictions[param][model] = paramsNN[i]

                    print(all_model_predictions)
                    if model == "T2_IVIM_NET_seg_fix_150": # plot map after last model
                        parameter_distribution_plot_updated(all_model_predictions, 0, folder_fig)


                    # then distribution plots around some true value
                    true_values = {'D': 0.0008, 'f': 0.3, 'Dp': 0.003, 'T2t': 0.1, 'T2p': 0.06}

                    # simulate data to see the distributions around some true value
                    data_D, _, _, _, _, _ = sim_signal(snr_level, bvalues_all, TEs, sims=1000, Dmin=true_values['D'],
                                                               Dmax=true_values['D'], fmin=0.0, fmax=0.7, Dsmin=0, Dsmax=0.05,
                                                               fp2min=0, fp2max=0.3, Ds2min=0.2, Ds2max=0.5, T2pmin=0.02,
                                                               T2pmax=0.15, T2tmin=0.04, T2tmax=0.12, rician=False,
                                                               state=123, model='T2')

                    data_f, _, _, _, _, _ = sim_signal(snr_level, bvalues_all, TEs, sims=1000, Dmin=0,
                                                               Dmax=5.0 / 1000, fmin=true_values['f'], fmax=true_values['f'], Dsmin=0, Dsmax=0.05,
                                                               fp2min=0, fp2max=0.3, Ds2min=0.2, Ds2max=0.5, T2pmin=0.02,
                                                               T2pmax=0.15, T2tmin=0.04, T2tmax=0.12, rician=False,
                                                               state=123, model='T2')

                    data_Dp, _, _, _, _, _ = sim_signal(snr_level, bvalues_all, TEs, sims=1000, Dmin=0,
                                                               Dmax=5.0 / 1000, fmin=0.0, fmax=0.7, Dsmin=true_values['Dp'], Dsmax=true_values['Dp'],
                                                               fp2min=0, fp2max=0.3, Ds2min=0.2, Ds2max=0.5, T2pmin=0.02,
                                                               T2pmax=0.15, T2tmin=0.04, T2tmax=0.12, rician=False,
                                                               state=123, model='T2')

                    if "T2_IVIM_NET_seg_fix" not in model:
                        data_T2p, _, _, _, _, _ = sim_signal(snr_level, bvalues_all, TEs, sims=1000, Dmin=0,
                                                                   Dmax=5.0 / 1000, fmin=0.0, fmax=0.7, Dsmin=0, Dsmax=0.05,
                                                                   fp2min=0, fp2max=0.3, Ds2min=0.2, Ds2max=0.5, T2pmin=true_values['T2p'],
                                                                   T2pmax=true_values['T2p'], T2tmin=0.04, T2tmax=0.12, rician=False,
                                                                   state=123, model='T2')

                    data_T2t, _, _, _, _, _ = sim_signal(snr_level, bvalues_all, TEs, sims=1000, Dmin=0,
                                                               Dmax=5.0 / 1000, fmin=0.0, fmax=0.7, Dsmin=0, Dsmax=0.05,
                                                               fp2min=0, fp2max=0.3, Ds2min=0.2, Ds2max=0.5, T2pmin=0.02,
                                                               T2pmax=0.15, T2tmin=true_values['T2t'], T2tmax=true_values['T2t'], rician=False,
                                                               state=123, model='T2')

                    if "T2_IVIM_NET_seg_fix" in model:
                        for i, data in enumerate([data_D, data_f, data_Dp, data_T2t]):
                            predictions = test_IVIM_sim(model, data, bvalues_all, TEs, arg, folder_experiment)
                            predictions_around_true[fixed_params[i]][model] = predictions[i]

                    else:
                        for i, data in enumerate([data_D, data_f, data_Dp, data_T2t, data_T2p]):
                            predictions = test_IVIM_sim(model, data, bvalues_all, TEs, arg, folder_experiment)
                            predictions_around_true[all_params[i]][model] = predictions[i]

                    if model == "T2_IVIM_NET_seg_fix_150": # when all data of 4 models is collected, plot the distribution plots
                        distribution_plot(true_values, predictions_around_true, folder_fig)

    data_rows = []
    for model, snr_data in nrmse_data.items():
        for snr, params_data in snr_data.items():
            for param, nrmse_values in params_data.items():
                for sim_index, nrmse in enumerate(nrmse_values):
                    data_rows.append([model, snr, param, sim_index + 1, nrmse])

    df = pd.DataFrame(data_rows, columns=['Model', 'SNR', 'Parameter', 'Simulation', 'NRMSE'])

    csv_filename = f"{folder_experiment}/nrmse_data.csv"
    df.to_csv(csv_filename, index=False)

if __name__ == "__main__":
    experiment = sys.argv[1]
    n_sims = int(sys.argv[2])
    SNR = sys.argv[3]
    skip = sys.argv[4]

    run_experiment_sim(experiment, n_sims, SNR, skip)





        

