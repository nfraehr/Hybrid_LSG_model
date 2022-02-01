# -*- coding: utf-8 -*-
"""
Import .dfsu data files from Mike 21FM and export for later use.

@author: nfraehr
"""

# %% Import libraries

from mikeio import Dfsu
import numpy as np
from os import walk
from sklearn.neighbors import BallTree
import time
import csv

start_time = time.monotonic()
# %% Get list of files to import
Path = "../Raw_Data/"

# HF model data
Path_HF = Path + 'HF/'
File_HF = next(walk(Path_HF), (None, None, []))[2]  # [] if no file

# LF model data
Path_LF = Path + 'LF/'
File_LF = next(walk(Path_LF), (None, None, []))[2]  # [] if no file

# %% Save events names to text file

with open('MIKE21_files.csv', mode='w') as f:
    csvwriter = csv.writer(f)
    for idx in range(len(File_HF)):
        csvwriter.writerow([idx, File_HF[idx], File_LF[idx]])

    f.close()


# %% Import data
# Open and read Mike21 data
def get_dfsu_data(filepath):
    # Open dfsu file
    dfs = Dfsu(filepath)
    # Read file
    ds = dfs.read()
    # Get Water depth
    data_wd = ds['Total water depth']

    # Get time
    time = ds.time

    # Remove burn-in period (10 days)
    data_wd = data_wd[40:, :]
    data_wd = np.float32(data_wd)  # Convert to float32

    time = time[40:]

    return data_wd, time


# Read HF data
HF_new_event, HF_new_event_time = get_dfsu_data(Path_HF + File_HF[0])
HF_events = (HF_new_event,)
HF_evt_time = (HF_new_event_time,)
for idx in range(1, len(File_HF)):
    print('HF %i / %i imported' % (idx, len(File_HF)))
    FilePath_HF = Path_HF + File_HF[idx]
    HF_new_event, HF_new_event_time = get_dfsu_data(FilePath_HF)

    HF_events = HF_events + (HF_new_event,)
    HF_evt_time = HF_evt_time + (HF_new_event_time,)
print('HF %i / %i imported' % (idx + 1, len(File_HF)))

# Read LF data
LF_new_event, LF_new_event_time = get_dfsu_data(Path_LF + File_LF[0])
# Check if arrays have different number of timesteps and shorten LF data if so
if len(HF_events[0]) < len(LF_new_event):
    LF_new_event = LF_new_event[0:len(HF_events[0]), :]

LF_events = (LF_new_event,)
LF_evt_time = (LF_new_event_time,)

for idx in range(1, len(File_LF)):
    print('LF %i / %i imported' % (idx, len(File_LF)))
    FilePath_LF = Path_LF + File_LF[idx]
    LF_new_event, LF_new_event_time = get_dfsu_data(FilePath_LF)

    # Check if arrays have different number of timesteps and shorten LF data if so
    if len(HF_events[idx]) < len(LF_new_event):
        LF_new_event = LF_new_event[0:len(HF_events[idx]), :]

    LF_events = LF_events + (LF_new_event,)
    LF_evt_time = LF_evt_time + (LF_new_event_time,)
print('LF %i / %i imported' % (idx + 1, len(File_LF)))


# %% Get Mike21 model data
def get_Mike21_model_data(filepath):
    # Open dfsu file
    dfs = Dfsu(filepath)

    # Element based data
    el_ids = dfs.element_ids  # Element id
    ec = dfs.element_coordinates  # Element coordinates

    # Node based data
    node_ids = dfs.node_ids  # Node id
    nc = dfs.node_coordinates  # Node coordinates

    # Element table
    elem_table = dfs.element_table

    # Get area of elements
    area = dfs.get_element_area()

    return ec, el_ids, nc, node_ids, elem_table, area


# Load data from file
HF_ec, HF_el_ids, HF_nc, HF_node_ids, HF_elem_table, HF_area = get_Mike21_model_data(Path_HF + File_HF[0])
LF_ec, LF_el_ids, LF_nc, LF_node_ids, LF_elem_table, LF_area = get_Mike21_model_data(Path_LF + File_LF[0])


# %% Interpolate LF data to HF grid

def get_nearest(src_points, candidates, k_neighbors=1):
    """
    Find nearest neighbors for all source points from a set of candidate points
    https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
    """

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=30, metric='euclidean')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return closest, closest_dist


def interpolate_lowfidelity_data(highfidelity_ec, lowfidelity_ec, lowfidelity_data):
    # Get index of nearest HF cell for each LF cell
    lf_2_hf_index, lf_2_hf_distance = get_nearest(highfidelity_ec[:, 0:2], lowfidelity_ec[:, 0:2])
    # Write LF data to nearest HF cell
    lf_interpolated = lowfidelity_data[:, lf_2_hf_index]

    return lf_interpolated


# Interpolate LF data for each event individually
LF_int_events = (interpolate_lowfidelity_data(HF_ec, LF_ec, LF_events[0]),)
for idx in range(1, len(File_LF)):
    # Interpolate data and save to tuple
    LF_int_events = LF_int_events + (interpolate_lowfidelity_data(HF_ec, LF_ec, LF_events[idx]),)


# %% Binary representation

# Convert data to binary. 1 for flooded, 0 for dry. If water depth above 0, then flooded.
def convert2binary(data):
    threshold_flood = 0.03  # Threshold for flooding
    data_bin = np.where(data >= threshold_flood, 1, 0)
    return data_bin


# Convert HF, LF and interpolated LF data to binary values. Each event individually
HF_events_bin = (convert2binary(HF_events[0]),)  # HF data
LF_events_bin = (convert2binary(LF_events[0]),)  # LF data
LF_int_events_bin = (convert2binary(LF_int_events[0]),)  # LF interpolated data
for idx in range(1, len(HF_events)):
    HF_events_bin = HF_events_bin + (convert2binary(HF_events[idx]),)
    LF_events_bin = LF_events_bin + (convert2binary(LF_events[idx]),)
    LF_int_events_bin = LF_int_events_bin + (convert2binary(LF_int_events[idx]),)

#%% Divide into training and validation data
# Training and validation events. Index is according to list of filenames in variable File_HF
train_events = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25]
val_events = [0, 2, 8]

# Training events
HF_train_events_bin = ()
LF_train_events_bin = ()
LF_int_train_events_bin = ()
HF_train_evt_time = ()
for idx in train_events:
    HF_train_events_bin = HF_train_events_bin + (HF_events_bin[idx],)
    LF_train_events_bin = LF_train_events_bin + (LF_events_bin[idx],)
    LF_int_train_events_bin = LF_int_train_events_bin + (LF_int_events_bin[idx],)
    HF_train_evt_time = HF_train_evt_time + (HF_evt_time[idx],)

# Validation events
HF_val_events_bin = ()
LF_val_events_bin = ()
LF_int_val_events_bin = ()
HF_val_evt_time = ()
for idx in val_events:
    HF_val_events_bin = HF_val_events_bin + (HF_events_bin[idx],)
    LF_val_events_bin = LF_val_events_bin + (LF_events_bin[idx],)
    LF_int_val_events_bin = LF_int_val_events_bin + (LF_int_events_bin[idx],)
    HF_val_evt_time = HF_val_evt_time + (HF_evt_time[idx],)

# %% Save binary data arrays
np.savez('../Managed_Data/Events_data/Binary_Data_Train_Events.npz', HF_events_bin=HF_train_events_bin,
         LFint_events_bin=LF_int_train_events_bin, Time=HF_train_evt_time, Area=HF_area)

np.savez('../Managed_Data/Events_data/Binary_Data_Val_Events.npz', HF_events_bin=HF_val_events_bin,
         LFint_events_bin=LF_int_val_events_bin, Time=HF_val_evt_time, Area=HF_area)

#Save LF data to use for comparison
np.savez('../Managed_Data/Events_data/Binary_Data_Train_Events_LFmodel.npz', LF_events_bin=LF_train_events_bin,
         Time=HF_train_evt_time, Area=LF_area) #timing for LF and HF events are the same.

np.savez('../Managed_Data/Events_data/Binary_Data_Val_Events_LFmodel.npz', LF_events_bin=LF_val_events_bin,
         Time=HF_val_evt_time, Area=LF_area) #timing for LF and HF events are the same.

# %%
end_time = time.monotonic()

with open('importtime.txt', 'w') as f:
    import_data_time = end_time - start_time
    f.write('Time for importing and converting data data: %i s \n' % import_data_time)

    f.close()
