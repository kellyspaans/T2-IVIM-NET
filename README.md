# Estimating T2-Intra-Voxel Incoherent Motion (IVIM) parameters in stroke patients two years post-onset using a T2-corrected extension of IVIM-NET.

14-02-2024
By Kelly Spaans 

## Introduction

This project comprises scripts for T2-IVIM parameter estimation based on Diffusion-Weighted Imaging (DWI) scans. The code is built upon code of Laurens Prast (2023) and Oliver Gurney-Champion (2020) (https://github.com/oliverchampion/IVIMNET). The underlying architecture of IVIM-NET involves unsupervised physics-informed deep neural networks. In the current extension, known as T2-IVIM-NET, T2 relaxation time correction was introduced by incorporating echo times as input variables. 

## Features

Scripts include code for both a simulated and an in vivo experiment in which T2-IVIM is trained and evaluated. Furthermore, visualization and statistics scripts are available. 

Additionally, statistical tests can be run with the **statistics.py** code, and plots can be made with **plots_for_paper.py**.


## Getting Started

### Prerequisites

The following packages are needed before running the script:
* numpy
* torch
* tqdm
* matplotlib
* scipy
* joblib
* nibabel

## Usage

To train and test the T2-IVIM-NET model, two scripts can be used for either simulated or in vivo data:

**simulation_T2_experiment.py** contains code for training/testing the model on simulated data.
**in_vivo_T2_experiment.py** contains code for training/testing the model on in vivo data. Note that this script can only be used within the Amsterdam UMC environment due to patient information. 


## Acknowledgments

Special thanks to Oliver Gurney-Champion and Laurens Prast for their foundational code. Their work laid the groundwork for this project.

Finally, another thanks to Amsterdam UMC for generously providing access to patient data to realize this project.

## Contact

If there are any questions or suggestions regarding these scripts, please do not hesitate to contact me at kellyspaans11@gmail.com.



