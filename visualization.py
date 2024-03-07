"""
19-01-2024
K.E. Spaans

This is code that contains functions to visualize model behaviour (parameter correlation map, parameter
distribution plots, etc.).
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import random as rd
from matplotlib.ticker import FormatStrFormatter


def plot_correlation_matrix(signal, D, f, Dp, T2t, bvalues, bvalue, TEs, TE, model, T2p=None):
    """
    Calculates the correlation matrix for the parameters and signal and saves these
    correlations in a heatmap. Additionally it returns the correlation values.
    """

    signal_filtered = signal[:, ((bvalues==bvalue) & (TEs==TE))][:, 0]

    if T2p is not None:
        data = {
            'D': D.squeeze(),
            'f': f.squeeze(),
            'Dp': Dp.squeeze(),
            'T2p': T2p.squeeze(),
            'T2t': T2t.squeeze(),
            'signal': signal_filtered
        }

    else:
        data = {
            'D': D.squeeze(),
            'f': f.squeeze(),
            'Dp': Dp.squeeze(),
            'T2t': T2t.squeeze(),
            'signal': signal_filtered
        }

    df = pd.DataFrame(data)

    corr_matrix = df.corr()  # Compute the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title(f'Correlation Matrix at b={bvalue}, TE={TE*1000}')
    plt.savefig(f"corr_{model}")

    return corr_matrix


def parameter_distribution_plot(model, snr_level, folder_fig, D_pred, Fp_pred, Dp_pred, T2t_pred, cpu=False, text=None, T2p_pred=None):
    """
    Plots the distribution of the given parameteres.
    """

    plt.subplots(2, 2, figsize=(11, 9))

    # D
    plt.subplot(2, 2, 1)
    if cpu:
        D_pred_plot = D_pred.cpu().data[:].numpy().flatten()
    else:
        D_pred_plot = D_pred
    plt.hist(D_pred_plot[~np.isnan(D_pred_plot)], bins=100, alpha=0.5, color='green', label='D_pred')
    # plt.xlim(-0.0005, 0.003)
    plt.legend()

    # Dp
    plt.subplot(2, 2, 2)
    if cpu:
        Dp_pred_plot = Dp_pred.cpu().data[:].numpy().flatten()
    else:
        Dp_pred_plot = Dp_pred
    plt.hist(Dp_pred_plot[~np.isnan(Dp_pred_plot)], bins=100, alpha=0.5, color='purple', label='Dp_pred')
    # plt.xlim(-0.0005, 0.05)
    plt.legend()

    # T2p & T2t
    plt.subplot(2, 2, 3)
    if cpu:
        if T2p_pred is not None:
            T2p_pred_plot = T2p_pred.cpu().data[:].numpy().flatten()
            T2t_pred_plot = T2t_pred.cpu().data[:].numpy().flatten()
        else:
            T2t_pred_plot = T2t_pred.cpu().data[:].numpy().flatten()
    else:
        if T2p_pred is not None:
            T2p_pred_plot = T2p_pred
            T2t_pred_plot = T2t_pred
        else:
            T2t_pred_plot = T2t_pred

    if T2p_pred is not None:
        plt.hist(T2p_pred_plot[~np.isnan(T2p_pred_plot)], bins=100, alpha=0.5, color='orange', label='T2p_pred')
        plt.hist(T2t_pred_plot[~np.isnan(T2t_pred_plot)], bins=100, alpha=0.5, color='red', label='T2t_pred')
    else:
        plt.hist(T2t_pred_plot[~np.isnan(T2t_pred_plot)], bins=100, alpha=0.5, color='red', label='T2t_pred')

    # plt.xlim(-0.0005, 0.15)
    plt.legend()

    # F
    plt.subplot(2, 2, 4)
    if cpu:
        Fp_pred_plt = Fp_pred.cpu().data[:].numpy().flatten()
    else:
        Fp_pred_plt = Fp_pred
    plt.hist(Fp_pred_plt[~np.isnan(Fp_pred_plt)], bins=100, alpha=0.5, color='blue', label='Fp_pred (Fp/(f0+Fp))')
    # plt.xlim(-0.005, 1)

    if text is not None:
        plt.suptitle(f'Distribution of all Parameters, {text}')
    else:
        plt.suptitle(f'Distribution of all Parameters')

    plt.legend()
    plt.savefig(f"{folder_fig}/distribution_{model}_snr{snr_level}.png")


def parameter_distribution_plot_updated(all_model_predictions, snr_level, folder_fig):
    """
    Plots the distribution of the given parameters for all models.
    """

    plt.figure(figsize=(15, 9))

    labels = ['$D$', '$f$', '$D*$', '$T_{2t}$', '$T_{2p}$']

    for i, param in enumerate(['D', 'f', 'Dp', 'T2t', 'T2p']):
        ax = plt.subplot(2, 3, i + 1)

        for model, predictions in all_model_predictions[param].items():
            if param == 'T2p' and "T2_IVIM_NET_seg_fix" in model:
                continue

            if param == 'f':
                sns.kdeplot(predictions*100, label=f'{model}')
            elif param == 'T2p' or param == 'T2t':
                sns.kdeplot(predictions*1000, label=f'{model}')
            else:
                sns.kdeplot(predictions, label=f'{model}')

        plt.title(f'{labels[i]}')
        plt.xlabel(f'$mm/s^2$') if (param == 'D') or (param == 'Dp') else plt.xlabel(f'$ms$')
        if param == 'f':
            plt.xlabel(f'%')

        if param == 'Dp':
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        # plt.legend()

    plt.suptitle(f'Distribution of T$_2$-IVIM Parameters')
    plt.tight_layout()
    plt.savefig(f"{folder_fig}/distribution.png")
    plt.close()

def parameter_distribution_plot_ivimnet(all_model_predictions, snr_level, folder_fig):
    """
    Plots the distribution of the given parameters for all models.
    """

    plt.figure(figsize=(15, 9))

    for i, param in enumerate(['D', 'f', 'Dp']):
        plt.subplot(2, 3, i + 1)

        for model, predictions in all_model_predictions[param].items():
            if param == 'T2p' and "T2_IVIM_NET_seg_fix" in model:
                continue

            sns.kdeplot(predictions, label=f'{model}')

        plt.title(f'{param}')
        plt.xlabel(f'{param} Value')
        plt.legend()

    plt.suptitle(f'Distribution of T$_2$-IVIM Parameters')
    plt.tight_layout()
    plt.savefig(f"{folder_fig}/distribution.png")
    plt.close()



def T2_sim_example_plot(bvalues, TEs, data_sim, D, f, Dp, T2p, T2t, sims, colors=['purple', 'blue', 'orange']):
    """
    Plots 4 example curves based on simulated T2-IVIM data. Also shows the parameter values for the curves. 
    """

    inds = np.lexsort((TEs, bvalues))
    data_sim = data_sim[:, inds]
    bvalues = bvalues[inds]
    TEs = TEs[inds]

    # show 4 random curves
    show = [rd.randint(1, sims), rd.randint(1, sims), rd.randint(1, sims), rd.randint(1, sims)]

    plt.subplots(2, 2, figsize=(11, 18))

    for j, idx in enumerate(show):
        plt.subplot(2, 2, j + 1)

        for i, te_value in enumerate(np.unique(TEs)):
            te_index = np.where(TEs == te_value)[0]

            plt.plot(bvalues[te_index], data_sim[idx, te_index], color=colors[i], label=f'signal TE={te_value * 1000}')
            plt.ylim((0, 1.4))
            plt.xlabel('b-value (s/mm2)')
            plt.ylabel(f'Normalized signal')
            plt.title(f'D={D[idx][0]:.3f}, f={f[idx][0]:.2f}, Dp={Dp[idx][0]:.3f},\nT2p={T2p[idx][0]:.2f}, T2t={T2t[idx][0]:.2f}', fontsize=9)

    plt.legend()
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def distribution_plot(true_values, predictions_around_true, folder_fig):
    """
    Plots the distribution of predicted parameter values from different models around their actual true values.

    Parameters:
    true_values: Dict with true values of parameters e.g., {'D': 0.0008, 'f': 0.3, ...}
    all_model_predictions: Dict with parameter keys and dict values containing model predictions
    folder_fig: Folder path to save the plots (optional)
    """

    labels = ['$D$', '$f$', '$D*$', '$T_{2t}$', '$T_{2p}$']
    i = 0

    for param, true_val in true_values.items():
        plt.figure()

        for model, predictions in predictions_around_true[param].items():
            # skip if model does not predict this parameter
            if param == 'T2p' and "T2_IVIM_NET_seg_fix" in model:
                continue
            if param == 'f':
                sns.kdeplot(predictions*100)
                if "seg_fix_150" in model:
                    true_val = true_val * 100
            elif param == 'T2p' or param == 'T2t':
                sns.kdeplot(predictions*1000)
                if "seg_fix_150" in model:
                    true_val = true_val * 1000
            else:
                sns.kdeplot(predictions)

        plt.axvline(x=true_val, color='r', linestyle='--', label=f'True value: {true_val}')
        plt.title(f'{labels[i]}')
        i += 1
        plt.xlabel(f'$mm/s^2$') if (param == 'D') or (param == 'Dp') else plt.xlabel(f'$ms$')
        if param == 'f':
            plt.xlabel(f'%')
        plt.legend()

        plt.savefig(f"{folder_fig}/distribution_{param}.png")

