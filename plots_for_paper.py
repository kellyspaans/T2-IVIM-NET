"""
K.E. Spaans 2024
This code is used for creating the plots that will be used in the paper.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def boxplot_nrmse_signals(csv_file, output_path):

    data = pd.read_csv(csv_file)

    # first column is the simulation number and others are model names
    melted_data = data.melt(id_vars=['sim'], var_name='Model', value_name='NRMSE')

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y='NRMSE', data=melted_data)

    plt.title('NRMSE Scores')
    plt.xlabel('Model')
    plt.ylabel('NRMSE')

    plt.savefig(f"{output_path}/boxplots.png")


csv_file = '/data/projects/followup-NOIV/students/kespaans/data/Experiments/T2/in_vivo_L1/nrmse_signals_in_vivo.csv'
output_path = '/data/projects/followup-NOIV/students/kespaans/data/thesis_figures/'

# boxplot_nrmse_signals(csv_file, output_path)


import pandas as pd
import matplotlib.pyplot as plt


def plot_nrmse_params(csv_file, output_path):

    data = pd.read_csv(csv_file)

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data['SNR'] = data['SNR'].astype(int)
    data['SNR'] = data['SNR'].replace(0, 140)
    data['NRMSE'] = data['NRMSE'].astype(float)

    parameters = data['Parameter'].unique()

    labels = ['$D$', '$f$', '$D*$', '$T_{2p}$', '$T_{2t}$']

    for i, parameter in enumerate(parameters):
        param_data = data[data['Parameter'] == parameter]

        median_nrmse = param_data.groupby(['Model', 'SNR'])['NRMSE'].median().unstack(0)
        sd_nrmse = param_data.groupby(['Model', 'SNR'])['NRMSE'].std().unstack(0)

        plt.figure(figsize=(8, 6))

        for model in median_nrmse.columns:
            snr_levels = median_nrmse.index
            median_values = median_nrmse[model]
            sd_values = sd_nrmse[model]

            plt.plot(snr_levels, median_values, marker='o', label=model)
            plt.fill_between(snr_levels, median_values - sd_values, median_values + sd_values, alpha=0.2)

            plt.title(f'{labels[i]}', size=15)
            plt.xlabel('SNR', size=15)
            plt.ylabel('NRMSE', size=15)
            plt.xticks([20, 60, 100, 140], size=15)
            plt.yticks(size=12)
            plt.gca().set_xticklabels([20, 60, 100, '$\infty$'])
            plt.gca().get_xticklabels()[-1].set_fontsize(20)
            # plt.legend()

        plt.savefig(f"{output_path}/nrmse_params_{parameter}.png")


csv_file = '/data/projects/followup-NOIV/students/kespaans/data/Experiments/T2/final_sim_20sims/nrmse_data_updated.csv'

plot_nrmse_params(csv_file, output_path)


