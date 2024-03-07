"""
K.E. Spaans 2024
This code is used to test various hyperparameters.
"""

from simulation_T2_experiment import run_experiment_sim

# Define the different hyperparameter combinations
cons_max_values = [[0.005, 0.7, 0.3, 2.5, 0.15, 0.12], [0.005, 0.7, 0.2, 2.0, 0.15, 0.12], [0.003, 1.0, 0.05, 1.0, 0.15, 0.12],
                   [0.005, 0.7, 0.3, 2.5, 0.15, 0.12], [0.005, 0.7, 0.2, 2.0, 0.15, 0.12], [0.003, 1.0, 0.05, 1.0, 0.15, 0.12]] # Dt, Fp, Ds, f0, T2p, T2t
loss_fun_values = ['rms', 'rms', 'rms', 'L1', 'L1', 'L1']
experiments = ["exp7", "exp8", "exp9", "exp10", "exp11", "exp12"]

# fixed arguments
n_sims = 1
SNR = "0"
skip = "0"

for cons_max, loss, exp in zip(cons_max_values, loss_fun_values, experiments):
    run_experiment_sim(exp, n_sims, SNR, skip, cons_max=cons_max, loss=loss)
