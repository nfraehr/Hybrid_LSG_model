# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 13:16:26 2021

@author: nfraehr
"""

# %% Libraries
import numpy as np


# %% Functions

# Convert tuple of events to array
def tuple2array(data_tuple):
    for idx in range(len(data_tuple)):
        if idx == 0:
            data_array = data_tuple[idx]
        else:
            data_array = np.vstack([data_array, data_tuple[idx]])

    return data_array
