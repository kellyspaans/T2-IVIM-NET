"""
K.E. Spaans 2024
This script containts code used for statistical analysis.
"""

import pandas as pd
from scipy import stats

# data load
data = pd.read_csv('/data/projects/followup-NOIV/students/kespaans/data/Experiments/T2/in_vivo_L1 (copy 1)/nrmse_signals_in_vivo.csv', index_col=0)
data = data.melt(var_name='Model', value_name='NRMSE')

# perform the Shapiro-Wilk test for each model
grouped = data.groupby('Model')
for model, values in grouped:
    stat, p = stats.shapiro(values['NRMSE'])
    print(f'{model}: Statistics={stat:.3f}, p={p:.6f}')

groups = [group['NRMSE'].values for name, group in data.groupby('Model')]

# Levene's test for homogeneity of variances
stat, p = stats.levene(*groups)
print(f"Levene's Test for Homogeneity of Variances: Statistics={stat:.3f}, p={p:.3f}")

# either ANOVA or Kruskal-Wallis is used based on the p-values of the above tests
if p > 0.05:
    print("\nPerforming ANOVA...")
    f_stat, p_val = stats.f_oneway(*groups)
    print(f'ANOVA Test: F-Statistic={f_stat:.3f}, p={p_val:.3f}')
else:
    print("\nPerforming Kruskal-Wallis Test...")
    h_stat, p_val = stats.kruskal(*groups)
    print(f'Kruskal-Wallis Test: H-Statistic={h_stat:.3f}, p={p_val}')

# Mann-Whitney U test
for model in range(len(groups)):
    if model == 0:
        continue
    else:
        u_stat, p_val = stats.mannwhitneyu(groups[0], groups[model])
        print(f'\nMann-Whitney U Test between "T2_IVIM_NET_orig" and group:{model}: U-Statistic={u_stat}, p={p_val:.30f}')

