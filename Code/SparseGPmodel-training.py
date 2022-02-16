# -*- coding: utf-8 -*-
"""
Created on Tue sep 14 

@author: nfraehr
"""

# %% Import libraries
import gpflow

import numpy as np
from sklearn.preprocessing import StandardScaler
import time

from EOF_Functions import CreateEOFobj, CreatePseudoECs, NorthsRule
from Datamanagement_Functions import tuple2array

start_time = time.monotonic()
# %% Import data
data = np.load('Managed_Data/Events_data/Binary_Data_Train_Events.npz', allow_pickle=True)

HF_events_bin = tuple(data['HF_events_bin'])
LF_int_events_bin = tuple(data['LFint_events_bin'])
Time = tuple(data['Time'])
Area = data['Area']

ckpt_loaddata_time = time.monotonic()


# %% Functions
# Training and storing SPGP model
def sparse_gaussian_model_training(highfidelity_ecs_train, lowfidelity_ecs_train,
                                   n_input_ecs=1, n_output_ecs=1, inducing_point_fraction=0.02):
    # Check if number of inducing variables is specified
    n_inducing_points = round(len(highfidelity_ecs_train) * inducing_point_fraction)

    # Create scaling objects
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # Training data
    n_input_ecs = np.array(range(n_input_ecs))
    x_train = np.c_[lowfidelity_ecs_train[:, n_input_ecs]]
    x_train_sc = scaler_x.fit_transform(x_train)

    # Loop for each mode that needs to be predicted
    for iECs in range(n_output_ecs):
        # y training data
        y_train = np.c_[highfidelity_ecs_train[:, iECs]]

        # Scaling of data to mean of 0 and variance 1
        y_train_sc = scaler_y.fit_transform(y_train)

        # Create array with initial inducing points
        dim = x_train_sc.shape[1]
        inducing_points = np.c_[np.linspace(x_train_sc[:, 0].min(), x_train_sc[:, 0].max(), n_inducing_points)]
        for j in range(1, dim):
            inducing_points = np.c_[inducing_points,
                                    np.linspace(x_train_sc[:, j].min(), x_train_sc[:, j].max(), n_inducing_points)]

        # Find initial lengthscale
        ini_length = np.mean(abs(x_train_sc))

        # Create kernel
        k_exp = gpflow.kernels.Exponential(variance=1, lengthscales=ini_length)

        # Create model
        spgp_model = gpflow.models.SGPR(data=(x_train_sc, y_train_sc), kernel=k_exp, inducing_variable=inducing_points)


        # Choosing optimiser
        opt = gpflow.optimizers.Scipy()

        # Fixing hyperparameters
        gpflow.set_trainable(spgp_model.kernel.variance, False)
        gpflow.set_trainable(spgp_model.kernel.lengthscales, False)
        gpflow.set_trainable(spgp_model.likelihood.variance, False)

        # Optimise model
        opt.minimize(spgp_model.training_loss,
                     spgp_model.trainable_variables,
                     method='L-BFGS-B',
                     options=dict(maxiter=100))

        # Fixing inducing points and optimising hyperparameters
        gpflow.set_trainable(spgp_model.kernel.variance, True)
        gpflow.set_trainable(spgp_model.kernel.lengthscales, True)
        gpflow.set_trainable(spgp_model.likelihood.variance, True)
        gpflow.set_trainable(spgp_model.inducing_variable.Z, False)

        opt.minimize(spgp_model.training_loss,
                     spgp_model.trainable_variables,
                     method='L-BFGS-B',
                     options=dict(maxiter=100))

        # Save model
        params = gpflow.utilities.parameter_dict(spgp_model)
        np.savez('Managed_Data/SPGP_class_models/SPGP_cls_trained_ECs_%s.npz' % iECs,
                 param=params,
                 x_train=x_train_sc, y_train=y_train_sc, M=n_inducing_points, scaler_x=scaler_x, scaler_y=scaler_y)

        # Print how far the process is
        print('ECs %i / %i is done \n' % (iECs + 1, n_output_ecs))


ckpt_loadfunction_time = time.monotonic()
# %% Define training data for EOF analysis
# HF data
HF_train = tuple2array(HF_events_bin)

# LF int data
LF_int_train = tuple2array(LF_int_events_bin)

# Classify data as dry, always flooded, or temporary flooded.
# EOF analysis is only performed on the temporary flooded cells.
# Condition for being wet or dry after converting to binary
dry_threshold = 0.5
AD_idx = np.where(HF_train.max(axis=0) < dry_threshold)[0]  # Dry cells AD
wet_idx = np.where(HF_train.max(axis=0) > dry_threshold)[0]  # Wet cells (AF+TF)
AF_idx = np.where(HF_train.min(axis=0) > dry_threshold)[0]  # Always flooded AF
TF_idx = np.where((HF_train.min(axis=0) < dry_threshold) &
                  (HF_train.max(axis=0) > dry_threshold))[0]  # Temporary flooded TF

# Save categories
np.savez('Managed_Data/SPGP_class_models/Categories_Training_data.npz',
         AD_idx=AD_idx, wet_idx=wet_idx, AF_idx=AF_idx, TF_idx=TF_idx)
# %% Perform EOF analysis on floodplain data
# Training data for floodplain
HF_train_FP = HF_train[:, TF_idx]
LF_int_train_FP = LF_int_train[:, TF_idx]

# Area for floodplain cells
Area_FP = Area[TF_idx]  # HF cell area for floodplain cells

# Perform EOF analysis and create EOF object
HF_EOFobj, HF_mean_train = CreateEOFobj(HF_train_FP, Area_FP)

# Get EOF modes for HF data
HF_EOFs = HF_EOFobj.components_

# Find number of significant modes
n_ECs = NorthsRule(HF_EOFobj, len(HF_train_FP))
ECs_variance_explained = np.cumsum(HF_EOFobj.explained_variance_ratio_)[-1]
Eigenvalues = HF_EOFobj.explained_variance_

# Save EOF data
np.savez('Managed_Data/SPGP_class_models/EOF_data.npz',
         HF_EOFs=HF_EOFs, HF_mean_train=HF_mean_train, Area_FP=Area_FP, n_ECs=n_ECs)

ckpt_saveEOF_time = time.monotonic()

# %% Get ECs for training
# Pseudo ECs and real ECs are the same for HF training data, as the EOF is performed on that.
HF_ECs_train = CreatePseudoECs(HF_train_FP, HF_EOFs[0:n_ECs, :], Area_FP)

# Pseudo ECs is used, so all ECs are made from the same modes, and thereby comparable
LF_int_pECs_train = CreatePseudoECs(LF_int_train_FP, HF_EOFs[0:n_ECs, :], Area_FP)


ckpt_createtrainingdata_time = time.monotonic()

# %% Sparse Gaussian Process regression model to predict HF ECs from LF pseudo ECs
sparse_gaussian_model_training(HF_ECs_train, LF_int_pECs_train,
                               n_input_ecs=n_ECs, n_output_ecs=n_ECs, inducing_point_fraction=0.02)


ckpt_trainSPGPend_time = time.monotonic()

# %% Timing
with open('trainingtime.txt', 'w') as f:
    Loadbinarydata_time = ckpt_loaddata_time - start_time
    f.write('Time for loading binary data: %i s \n' % Loadbinarydata_time)

    LoadFunctions_time = ckpt_loadfunction_time - ckpt_loaddata_time
    f.write('Time for loading functions: %i s \n' % LoadFunctions_time)

    SaveEOFobj_time = ckpt_saveEOF_time - ckpt_loadfunction_time
    f.write('Creating and saving time for EOF obj: %i s \n' % SaveEOFobj_time)

    CreateTrainingdata_time = ckpt_createtrainingdata_time - ckpt_saveEOF_time
    f.write('Time for EOF analysis and creating training data: %i s \n' % CreateTrainingdata_time)

    trainSPGP_time = ckpt_trainSPGPend_time - ckpt_createtrainingdata_time
    f.write('Training time for SPGP model: %i s \n' % trainSPGP_time)

    Total_time = ckpt_trainSPGPend_time - start_time
    f.write('Total time: %i s \n' % Total_time)

    f.close()


