# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:05:53 2021

@author: nfraehr
"""

# %% Import libraries
import gpflow
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.patches as mpatches
from mikeio import Dfsu
import numpy as np
import pandas as pd
import time

from EOF_Functions import CreatePseudoECs, ReconstructDataFromEOFandECs
from Datamanagement_Functions import tuple2array

# %% Import data
data = np.load('Managed_Data/Events_data/Binary_Data_Val_Events.npz', allow_pickle=True)

HF_events_bin = tuple(data['HF_events_bin'])
LF_int_events_bin = tuple(data['LFint_events_bin'])
Time = tuple(data['Time'])
Area = data['Area']

#LF data for comparison
data_LF = np.load('Managed_Data/Events_data/Binary_Data_Val_Events_LFmodel.npz', allow_pickle=True)
LF_events_bin = tuple(data_LF['LF_events_bin'])
LF_Area = data_LF['Area']

val_events = [1, 3, 6] #
# %% Functions
# Load SPGP classification models
def load_sparse_gaussian_models(n_output_ecs=1):
    k_exp_list = []
    spgp_model_list = []
    scaler_x_list = []
    scaler_y_list = []

    for iECs in range(n_output_ecs):
        # Load data
        fullpath = 'Managed_Data/SPGP_class_models/SPGP_cls_trained_ECs_%s.npz' % iECs

        data = np.load(fullpath, allow_pickle=True)
        m_param = data['param'].item()
        x_train = data['x_train']
        y_train = data['y_train']
        n_inducing_points = data['M']
        scaler_x = data['scaler_x'][()]
        scaler_y = data['scaler_y'][()]

        # Scaler lists
        scaler_x_list.append(scaler_x)
        scaler_y_list.append(scaler_y)

        # Find initial lengthscale
        # ini_length = np.array([abs(x_train).max(axis=0)])
        ini_length = np.mean(abs(x_train))

        # Appending kernel to list
        # k_exp_list.append(gpflow.kernels.Exponential(variance=1, lengthscales=ini_length))
        k_exp_list.append(gpflow.kernels.Exponential(variance=1, lengthscales=ini_length))

        # Create inducing variables
        dim = x_train.shape[1]
        inducing_points = np.c_[np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), n_inducing_points)]
        for j in range(1, dim):
            inducing_points = np.c_[inducing_points,
                                    np.linspace(x_train[:, j].min(), x_train[:, j].max(), n_inducing_points)]

        # create multi-output inducing variables from Z
        iv = gpflow.inducing_variables.InducingPoints(inducing_points)

        # Appending SPGP model to list
        spgp_model_list.append(gpflow.models.SGPR(data=(x_train, y_train),
                                                  kernel=k_exp_list[iECs], inducing_variable=iv))

        # Initialise model
        gpflow.utilities.multiple_assign(spgp_model_list[iECs], m_param)

    return spgp_model_list, scaler_x_list, scaler_y_list


# Predicting ECs using the SPGP models
def spgp_model_predict(lowfidelity_ecs_val, n_input_ecs=1, n_output_ecs=1):
    # Load model
    spgp_list, sc_x, sc_y = load_sparse_gaussian_models(n_output_ecs=n_output_ecs)

    # Validation data
    n_input_ecs = np.array(range(n_input_ecs))

    x_val = np.c_[lowfidelity_ecs_val[:, n_input_ecs]]

    for iECs in range(n_output_ecs):
        # Load model corresponding to the ECs
        spgp_model = spgp_list[iECs]

        # Scaling data
        x_val = sc_x[iECs].transform(x_val)

        # Predictions
        y_pred_mean, y_pred_var = spgp_model.predict_y(x_val)
        y_pred_mean = y_pred_mean.numpy()
        y_pred_var = y_pred_var.numpy()

        # Confidence interval
        y__pred_conf = 1.96 * np.sqrt(y_pred_var)
        y_lb = y_pred_mean - y__pred_conf
        y_ub = y_pred_mean + y__pred_conf

        # Scaled back to real values
        y_pred_mean = sc_y[iECs].inverse_transform(y_pred_mean)
        y_lb = sc_y[iECs].inverse_transform(y_lb)
        y_ub = sc_y[iECs].inverse_transform(y_ub)

        # Save predicted ECs to array
        if iECs == 0:
            ecs_pred = y_pred_mean
            ecs_pred_lb = y_lb
            ecs_pred_ub = y_ub
        else:
            ecs_pred = np.c_[ecs_pred, y_pred_mean]
            ecs_pred_lb = np.c_[ecs_pred_lb, y_lb]
            ecs_pred_ub = np.c_[ecs_pred_ub, y_ub]

    return ecs_pred, ecs_pred_lb, ecs_pred_ub


# Reconstruct all data
def reconstruct_binary_highres_data(highfidelity_val_data, eof_spatialpattern, highfidelity_mean_train,
                                    ecs_temporalfunction, floodplain_index, river_index, weights, dry_threshold=0.5):
    # Reconstruct floodplain data
    recon_floodplain = ReconstructDataFromEOFandECs(eof_spatialpattern[0:ecs_temporalfunction.shape[1], :],
                                                    ecs_temporalfunction, highfidelity_mean_train, weights)
    # convert to binary
    recon_floodplain_bin = np.where(recon_floodplain > dry_threshold, 1, 0)

    # Reconstruct all cells
    recon_bin_all = np.zeros(highfidelity_val_data.shape)
    recon_bin_all[:, floodplain_index] = recon_floodplain_bin
    recon_bin_all[:, river_index] = 1

    return recon_bin_all


# Convert validation data to tuple list for each validation event
def valdata2eventtuple(val_data_array, val_time):
    # Start timestep of first event
    event_start_timestep = 0
    for ival in range(len(val_time)):
        # Find end of event
        event_stop_timestep = event_start_timestep + len(val_time[ival])
        # Save events to tuple
        if ival == 0:
            val_data_events = (val_data_array[event_start_timestep:event_stop_timestep, :],)
        else:
            val_data_events = val_data_events + (val_data_array[event_start_timestep:event_stop_timestep, :],)

        # Find start of next event
        event_start_timestep = event_stop_timestep

    return val_data_events


# Calculate POD and RFA for one timestep
def pod_rfa_1tstep(true_labels, pred_labels):
    true_pos_all = np.sum(np.where((pred_labels > 0) & (true_labels > 0), 1, 0))
    true_neg_all = np.sum(np.where((pred_labels == 0) & (true_labels == 0), 1, 0))
    false_pos_all = np.sum(np.where((pred_labels > 0) & (true_labels == 0), 1, 0))
    false_neg_all = np.sum(np.where((pred_labels == 0) & (true_labels > 0), 1, 0))
    # ConfusionMatrix_sum = True_pos + True_neg + False_pos + False_neg

    precision_all = true_pos_all / (true_pos_all + false_pos_all)
    recall_all = true_pos_all / (true_pos_all + false_neg_all)
    fscore_all = (2 * precision_all * recall_all) / (precision_all + recall_all)

    # Rate of false alarm and probability of detection
    pod_all = recall_all
    rfa_all = (false_pos_all / (true_pos_all + false_pos_all))

    return pod_all, rfa_all


# Calculate POD and RFA for each timestep in array
def pod_fra_all_tstep(true_labels, pred_labels):
    tn = np.zeros(len(true_labels))
    fp = tn.copy()
    fn = tn.copy()
    tp = tn.copy()
    precision = tn.copy()
    recall = tn.copy()
    fscore = tn.copy()
    pod = tn.copy()
    rfa = tn.copy()

    for idx in range(len(true_labels)):
        tp[idx] = np.sum(np.where((pred_labels[idx, :] > 0) & (true_labels[idx, :] > 0), 1, 0))
        tn[idx] = np.sum(np.where((pred_labels[idx, :] == 0) & (true_labels[idx, :] == 0), 1, 0))
        fp[idx] = np.sum(np.where((pred_labels[idx, :] > 0) & (true_labels[idx, :] == 0), 1, 0))
        fn[idx] = np.sum(np.where((pred_labels[idx, :] == 0) & (true_labels[idx, :] > 0), 1, 0))

        precision[idx] = tp[idx] / (tp[idx] + fp[idx])
        recall[idx] = tp[idx] / (tp[idx] + fn[idx])
        fscore[idx] = (2 * precision[idx] * recall[idx]) / (precision[idx] + recall[idx])

        # Rate of false alarm and probability of detection
        pod[idx] = recall[idx]
        rfa[idx] = fp[idx] / (tp[idx] + fp[idx])

    return pod, rfa


# Function for determining if inundation is detected correctly or not.
def inundation_detection(true_labels, pred_labels):
    # find correctly detected
    detected_flood = np.where((pred_labels > 0) & (true_labels > 0))[0]
    detected_dry = np.where((pred_labels == 0) & (true_labels == 0))[0]

    # Find false alarm -> predicted flooded, but is not flooded
    false_alarm = np.where((pred_labels > 0) & (true_labels == 0))[0]
    # Find misses -> predicted dry, but is flooded
    misses = np.where((pred_labels == 0) & (true_labels > 0))[0]

    # Assemble data to matrix with all cells
    inundation_detection = np.zeros(len(true_labels))
    inundation_detection[detected_dry] = -1
    inundation_detection[detected_flood] = 0
    inundation_detection[false_alarm] = 1
    inundation_detection[misses] = 2

    return inundation_detection


# %% Import EOF data
EOF_data = np.load('Managed_Data/SPGP_class_models/EOF_data.npz', allow_pickle=True)

HF_EOFs = EOF_data['HF_EOFs']
HF_mean_train = EOF_data['HF_mean_train']
n_ECs = EOF_data['n_ECs']
Area_FP = EOF_data['Area_FP']

# %% Load category data
Cat_data = np.load('Managed_Data/SPGP_class_models/Categories_Training_data.npz',
                   allow_pickle=True)

AD_idx = Cat_data['AD_idx']  # Always dry
wet_idx = Cat_data['wet_idx']  # Wet cells (river+floodplain)
AF_idx = Cat_data['AF_idx']  # Always flooded
TF_idx = Cat_data['TF_idx']  # Temporary flooded

# %% Validation data
start_time = time.monotonic()

# Validation data
HF_val = tuple2array(HF_events_bin)
LF_int_val = tuple2array(LF_int_events_bin)

# Validation data for temporary flooded cells
HF_val_FP = HF_val[:, TF_idx]
LF_int_val_FP = LF_int_val[:, TF_idx]

HF_ECs_val = CreatePseudoECs(HF_val_FP, HF_EOFs[0:n_ECs, :], Area_FP)
LF_int_pECs_val = CreatePseudoECs(LF_int_val_FP, HF_EOFs[0:n_ECs, :], Area_FP)

Val_data_endtime = time.monotonic()
# %% Predictions
ECs_pred, ECs_pred_LB, ECs_pred_UB = spgp_model_predict(LF_int_pECs_val,
                                                        n_input_ecs=n_ECs,
                                                        n_output_ecs=n_ECs)
Prediction_endtime = time.monotonic()

# %% Reconstruct HF data
# Reconstruct LF converted data
LFint2HF_recon = reconstruct_binary_highres_data(HF_val, HF_EOFs, HF_mean_train, ECs_pred, TF_idx,
                                                 AF_idx, Area_FP, dry_threshold=0.5)

# Reconstruct using true ECs from the HF data for comparison
HF_recon = reconstruct_binary_highres_data(HF_val, HF_EOFs, HF_mean_train, HF_ECs_val[:, 0:n_ECs], TF_idx, AF_idx,
                                           Area_FP,
                                           dry_threshold=0.5)

# Divide validation data into each event
# Reconstructed data
LFint2HF_recon_events = valdata2eventtuple(LFint2HF_recon, Time)

HF_recon_events = valdata2eventtuple(HF_recon, Time)

Reconstruction_endtime = time.monotonic()

# %% Timing
with open('predictiontime.txt', 'w') as f:
    Validation_data_time = Val_data_endtime - start_time
    f.write('Time for generating validation data: %i s \n' % Validation_data_time)

    Prediction_time = Prediction_endtime - Val_data_endtime
    f.write('Prediction time for SPGP model: %i s \n' % Prediction_time)

    Reconstruction_time = Reconstruction_endtime - Prediction_endtime
    f.write('Reconstruction time: %i s \n' % Reconstruction_time)

    Total_time = Reconstruction_endtime - start_time
    f.write('Total time: %i s \n' % Total_time)

    f.close()

# %% Plot ECs
fig = plt.figure(figsize=(12, 10))
for i in range(1, 4):
    ax1 = plt.subplot(3, 1, i)
    k = i - 1
    ax1.title.set_text('Mode %i' % i)
    ax1.plot(HF_ECs_val[:, k], 'k-', label='High-fidelity ECs')
    ax1.plot(LF_int_pECs_val[:, k], 'r-', label='Low-fidelity ECs')

    ax1.plot(ECs_pred[:, k], 'b--', label='LF-GP model')
    ax1.fill_between(range(len(ECs_pred)), ECs_pred_LB[:, k], ECs_pred_UB[:, k],
                     label='GP conf. 95%', alpha=0.75, edgecolor='gray', facecolor='cyan')

    ax1.set_ylabel('ECs [-]')
    ax1.grid()

ax1.legend(title='Legend')

fig.tight_layout()

fig.savefig('Managed_Data/Classification_Figures/ECs_val_events.png',
            bbox_inches="tight", dpi=300)
plt.show()

# %% Calculate flooded area
RMSE = np.zeros(len(val_events)) #Root mean square error
Rel_RMSE = np.zeros(len(val_events)) #Relative RMSE
Rel_Peak_value_error = np.zeros(len(val_events)) #relative Peak Value Estimate
Rel_Peak_timing_error_peakperiod = np.zeros(len(val_events)) #relative average peak time error compared to the peak period
Rel_Peak_timing_error_floodperiod = np.zeros(len(val_events)) #relative average peak time error compared to the rising limb of the flood event

fig = plt.figure(figsize=(9, 6))
date_form = DateFormatter("%m-%Y")

for ival in range(len(val_events)):
    # True flooded area HF model
    HF_val_Aflood = np.sum(HF_events_bin[ival] * Area, axis=1)
    # LF without doing anything
    LF_Aflood_all = np.sum(LF_events_bin[ival] * LF_Area, axis=1)

    # HF data reconstruction using predicted LF ECs
    LFint2HF_recon_Aflood = np.sum(LFint2HF_recon_events[ival] * Area, axis=1)

    # Reconstructed using true ECs
    HF_recon_Aflood = np.sum(HF_recon_events[ival] * Area, axis=1)

    # Plot flooded area for all validation data
    index_subplot = ival + 1
    ax1 = plt.subplot(2, 2, index_subplot)
    ax1.title.set_text('Event %i'%val_events[ival])
    ax1.plot(Time[ival], HF_val_Aflood / 1e6, 'k-', label='High-fidelity simulation')
    ax1.plot(Time[ival], LF_Aflood_all / 1e6, 'r-', label='Low-fidelity simulation')
    ax1.plot(Time[ival], HF_recon_Aflood / 1e6, 'k--', label='Prediction with true ECs')

    ax1.plot(Time[ival], LFint2HF_recon_Aflood / 1e6, 'b-', linewidth=1.0, label='LF-GP simulation')

    ax1.set_ylabel('Inundation extent [km²]')
    ax1.grid()
    ax1.xaxis.set_major_formatter(date_form)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)

    # Inundation extent - evaluation
    n_tsteps_event = len(HF_val_Aflood)  # Total timesteps in event
    n_tsteps_5pc = round(0.05 * n_tsteps_event)  # Time steps for 5 percent

    # RMSE
    RMSE[ival] = np.sqrt((np.square(LFint2HF_recon_Aflood - HF_val_Aflood)).mean(axis=0))
    Rel_RMSE[ival] = RMSE[ival] / (HF_val_Aflood.mean(axis=0))

    # Peak values
    Peak_tsteps_SF = np.sort(np.argsort(-LFint2HF_recon_Aflood)[0:n_tsteps_5pc])
    Peak_tsteps_HF = np.sort(np.argsort(-HF_val_Aflood)[0:n_tsteps_5pc])

    Rel_Peak_value_error[ival] = (LFint2HF_recon_Aflood[Peak_tsteps_SF].mean() - HF_val_Aflood[Peak_tsteps_HF].mean()) / \
                                 HF_val_Aflood[Peak_tsteps_HF].mean()

    tstep_min_10pc = np.min(np.where(HF_val_Aflood > np.min(HF_val_Aflood) * 1.1)[0])
    peak_period_HF = Peak_tsteps_HF.max() - Peak_tsteps_HF.min()
    Rel_Peak_timing_error_peakperiod[ival] = (Peak_tsteps_SF.mean() - Peak_tsteps_HF.mean()) / \
                                  (peak_period_HF)
    Rel_Peak_timing_error_floodperiod[ival] = (Peak_tsteps_SF.mean() - Peak_tsteps_HF.mean()) / \
                                  (Peak_tsteps_HF.mean() - tstep_min_10pc)


fig.tight_layout()
leg = ax1.legend(title='Legend', loc='center left', bbox_to_anchor=(1.35, 0.5), frameon=False)
leg._legend_box.align = "left"

fig.savefig('Managed_Data/Classification_Figures/Floodedarea_valevents.png',
            bbox_inches="tight", dpi=300)
plt.show()

# %%Precision, recall and F-score for whole area

# Create array where each row is an event and each columns is a cell
LFint2HF_rec_events_max = np.zeros([len(val_events), HF_val.shape[1]], dtype=int)

HF_val_events_max = LFint2HF_rec_events_max.copy()

# Array to store POD and RFA data
LFint2HF_POD = np.zeros(len(val_events))
LFint2HF_RFA = np.zeros(len(val_events))

# event_start_timestep = 0
for ival in range(len(val_events)):
    # Get maximum extend of event for reconstruction
    LFint2HF_rec_events_max[ival, :] = LFint2HF_recon_events[ival].max(axis=0)

    # Get maximum extend of event
    HF_val_events_max[ival, :] = HF_events_bin[ival].max(axis=0)

    # POD and RFA
    LFint2HF_POD[ival], LFint2HF_RFA[ival] = pod_rfa_1tstep(HF_val_events_max[ival, :],
                                                            LFint2HF_rec_events_max[ival, :])

# %% Check POD and RFA for each timestep
POD_min = 1
RFA_max = 0

fig = plt.figure(figsize=(9, 6))

for ival in range(len(val_events)):
    LFint2HF_POD_alltstp, LFint2HF_RFA_alltstp = pod_fra_all_tstep(HF_events_bin[ival],
                                                                   LFint2HF_recon_events[ival])
    # Find lowest POD value and highest RFA
    if LFint2HF_POD_alltstp.min() < POD_min:
        POD_min = LFint2HF_POD_alltstp.min()

    if LFint2HF_RFA_alltstp.max() > RFA_max:
        RFA_max = LFint2HF_RFA_alltstp.max()

    index_subplot = ival + 1
    ax1 = plt.subplot(2, 2, index_subplot)
    ax1.title.set_text('Event %i' % val_events[ival])
    ax1.plot(Time[ival], LFint2HF_POD_alltstp, 'b-', label='Probability of detection')

    ax1.set_ylim([0, 1])
    ax1.set_ylabel('Detection')

    ax1.plot(Time[ival], LFint2HF_RFA_alltstp, 'r--', label='Rate of false alarm')

    ax1.grid()
    ax1.xaxis.set_major_formatter(date_form)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)

fig.tight_layout()
leg = ax1.legend(title='Legend', loc='center left', bbox_to_anchor=(1.35, 0.5), frameon=False)
leg._legend_box.align = "left"

fig.savefig('Managed_Data/Classification_Figures/POD_RFA_valevents.png',
            bbox_inches="tight", dpi=300)
plt.show()
# %% Examining location of highest errors

# Load Mike21 data for plotting
dfs_HF = Dfsu("Raw_Data/HF/Chow_HF_20110701_20111015.dfsu")  # dfsu object


color_map = matplotlib.colors.ListedColormap(['dodgerblue', 'yellow', 'red'])
cm = 1/2.54
fig = plt.figure(figsize=(15*cm, 12*cm))

for ival in range(len(val_events)):
    # Find locations where model differs from HF model
    LFint2HF_InunDetect = inundation_detection(HF_val_events_max[ival, :],
                                               LFint2HF_rec_events_max[ival, :])

    # Plot data exclusive Aflood
    plotelements = \
        np.where(
            (LFint2HF_InunDetect == 0) | (LFint2HF_InunDetect == 1) | (LFint2HF_InunDetect == 2))[
            0]

    index_subplot = ival + 1
    ax1 = plt.subplot(2, 2, index_subplot)
    ax1.title.set_text('Event %i' % val_events[ival])
    hb = dfs_HF.plot(LFint2HF_InunDetect[plotelements], elements=plotelements, plot_type='patch',
                     show_mesh=False, cmap=color_map, vmin=-0.5, vmax=2.5, ax=ax1)
    # remove the x and y ticks
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)

    # remove colorbar
    cbar = hb.collections[0].colorbar
    cbar.remove()



fig.tight_layout()

detected_patch = mpatches.Patch(color='dodgerblue', label='Detected')
FalseAlarm_patch = mpatches.Patch(color='yellow', label='False Alarm')
Misses_patch = mpatches.Patch(color='red', label='Miss')
leg = ax1.legend(title='Legend', handles=[detected_patch, FalseAlarm_patch, Misses_patch], loc='center left', bbox_to_anchor=(1.35, 0.5), frameon=False)
leg._legend_box.align = "left"

plt.savefig('Managed_Data/Classification_Figures/Detectec_falsealarm_misses.png',
            bbox_inches="tight", dpi=300)
plt.show()

#%% Plot the inundation for the LF, HF and LF-GP model at 3 different timesteps of event 1 for comparison
# Load Mike21 data for plotting
dfs_LF = Dfsu("Raw_Data/LF/Chow_LF_20110701_20111015.dfsu")  # dfsu object


# Timesteps to plot for comparison
plot_times_str = ['15/12/2010', '15/02/2011', '15/05/2011']
plot_times = pd.to_datetime(plot_times_str, format='%d/%m/%Y')

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15*cm, 11*cm))
color_map = matplotlib.colors.ListedColormap(['dodgerblue'])

for t_idx in range(len(plot_times)):
    # Find index of closest timestep to the timestep for plotting
    t_nearest = np.argmin(abs(plot_times[t_idx] - Time[0]))

    #LF model
    plotelements = np.where(LF_events_bin[0][t_nearest] == 1)[0]  # Elements to plot
    #Plot data
    hb = dfs_LF.plot(LF_events_bin[0][t_nearest][plotelements], elements=plotelements, plot_type='patch',
                     show_mesh=False, cmap=color_map, vmin=0.5, vmax=1.5, ax=axs[t_idx, 0])
    # remove the x and y ticks
    # axs[t_idx, 0].axis('off')
    axs[t_idx, 0].axes.xaxis.set_ticks([])
    axs[t_idx, 0].axes.yaxis.set_ticks([])
    # remove colorbar
    cbar = hb.collections[0].colorbar
    cbar.remove()

    #LF-GP model
    plotelements = np.where(LFint2HF_recon_events[0][t_nearest] == 1)[0]  # Elements to plot
    #Plot data
    hb = dfs_HF.plot(LFint2HF_recon_events[0][t_nearest][plotelements], elements=plotelements, plot_type='patch',
                     show_mesh=False, cmap=color_map, vmin=0.5, vmax=1.5, ax=axs[t_idx, 1])
    # remove the x and y ticks
    # axs[t_idx, 0].axis('off')
    axs[t_idx, 1].axes.xaxis.set_ticks([])
    axs[t_idx, 1].axes.yaxis.set_ticks([])
    # remove colorbar
    cbar = hb.collections[0].colorbar
    cbar.remove()

    #HF model
    plotelements = np.where(HF_events_bin[0][t_nearest] == 1)[0]  # Elements to plot
    #Plot data
    hb = dfs_HF.plot(HF_events_bin[0][t_nearest][plotelements], elements=plotelements, plot_type='patch',
                     show_mesh=False, cmap=color_map, vmin=0.5, vmax=1.5, ax=axs[t_idx, 2])
    # remove the x and y ticks
    # axs[t_idx, 0].axis('off')
    axs[t_idx, 2].axes.xaxis.set_ticks([])
    axs[t_idx, 2].axes.yaxis.set_ticks([])
    # remove colorbar
    cbar = hb.collections[0].colorbar
    cbar.remove()

    # Insert title
    if t_idx == 0:
        axs[t_idx, 0].title.set_text('Low-fidelity model')
        axs[t_idx, 1].title.set_text('LF-GP model')
        axs[t_idx, 2].title.set_text('High-fidelity model')

    # Insert text showing the plotted time
    axs[t_idx, 0].set_ylabel(plot_times_str[t_idx], rotation=0, fontsize=10)
    axs[t_idx, 0].yaxis.set_label_coords(-.3, .5)

fig.tight_layout()
plt.savefig('Managed_Data/Classification_Figures/Inundation_evolution_event1.png',
            bbox_inches="tight", dpi=300)
plt.show()
