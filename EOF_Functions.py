# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:49:03 2021

@author: nfraehr
"""

# %% Libraries
import numpy as np
from sklearn.decomposition import PCA


# %% Functions
# Perform EOF analysis on data
def CreateEOFobj(data, Weights):
    # Perform EOF analysis
    data_mean = np.mean(data, axis=0)  # Get mean of each timeseries
    data_center = data - data_mean  # Center data manually
    data_center_weighted = data_center * Weights  # weighting of data
    data_center_weighted = np.float32(data_center_weighted)
    EOFobj = PCA(n_components=100)
    EOFobj.fit(data_center_weighted)

    return EOFobj, data_mean


# Function for creating pseudo ECs
def CreatePseudoECs(data, EOFs, Weights):
    data_mean = np.mean(data, axis=0)  # Get mean of timeseries
    data_center = data - data_mean  # Center data

    data_center_weighted = data_center * Weights  # weighting of data
    # Generate pseudo ECs
    pseudo_ECs = np.dot(data_center_weighted, EOFs.T)

    return pseudo_ECs


# Reconstructing HF data using HF EOFs and  ECs
def ReconstructDataFromEOFandECs(EOFs, ECs, Mean, Weights):
    # Check if EOF and EC has equally many modes
    if EOFs.shape[0] != ECs.shape[1]:
        return print('EOF and ECs does not have equal number of modes')

    # Project ECs on EOFs
    recon = np.dot(ECs, EOFs)

    # Remove weight
    recon = recon / Weights

    recon = np.add(recon, Mean)
    # Return reconstructed values
    return recon


# Determining the number of EOFs needed according to Norths rul
def NorthsRule(EOFobj, n_samples):
    Eigenvalues = EOFobj.explained_variance_
    dEigen = abs(np.diff(Eigenvalues))  # Calculated difference between eigenvalues
    dError = np.sqrt(2 / n_samples) * Eigenvalues  # Calculates error for each eigenvalue
    dError = dError[:-1]  # removes last index of error array
    CheckErrorBoundary = np.where(
        (dEigen > dError) == False)  # Find first index of difference between eigenvalues being smaller than error.
    n_EofSignificant = CheckErrorBoundary[0][0]  # Gets index number

    return n_EofSignificant
