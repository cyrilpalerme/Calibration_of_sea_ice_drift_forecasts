import matplotlib
matplotlib.use('Agg')
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
import pandas as pd
from pandas import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
###########################################################
# Parameters
###########################################################
#
# Paths
path_output = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Tuning/Figures/'
path_IB_buoys = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Tuning/Results_without_dayofyear/Direction/Variable_importances_buoys/'
path_IB_SAR = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Tuning/Results_without_dayofyear/Direction/Variable_importances_SAR/'
path_MA_buoys = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Tuning/Results_without_dayofyear/Speed/Variable_importances_buoys/'
path_MA_SAR = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Tuning/Results_without_dayofyear/Speed/Variable_importances_SAR/'
#
# Figure parameters
sizefont_performances = 15
sizefont_feature_importances = 22
sizefont_legend_performances = 20
sizefont_legend_feature_importances = 25
colorscale = plt.cm.Paired
#colorscale = plt.cm.gnuplot2
width = 1
#############################################################################################################
# Random forest parameters
#############################################################################################################
bootstrap = True
max_depth = None
max_features = 3
min_samples_leaf = 1
min_samples_split = 2
n_estimators = 200
n_jobs = 1
#
rf_param_str = 'nestimators_' + str(n_estimators) + '_maxfeatures_' + str(max_features) +  '_bootstrap_' + str(bootstrap) + '_maxdepth_' + str(max_depth) + \
               '_minsamplessplit_' + str(min_samples_split) + '_minsamplesleaf_' + str(min_samples_leaf)
#
date_min_buoys = '20130606'
date_min_SAR = '20180104'
date_max = '20200528'
date_str_buoys = date_min_buoys + '-' + date_max
date_str_SAR = date_min_SAR + '-' + date_max
#
RF_variables = ['ECMWF_wd10m', 'ECMWF_ws10m', 'OSISAF_SIC', 'T4_drift_initial_bearing', 'T4_drift_magnitude', 'T4_fice', 'T4_hice', 'distance_to_land', 'xc', 'yc']
RF_variables_label = ['ECMWF wind direction', 'ECMWF wind speed', 'OSISAF sea ice concentration', 'TOPAZ4 drift direction', 'TOPAZ4 drift speed', 'TOPAZ4 sea ice concentration', 'TOPAZ4 sea ice thickness', 'Distance to land', 'x coordinate', 'y coordinate']
#############################################################################################################
# Load data
#############################################################################################################
file_IB_buoys = path_IB_buoys + 'IB_' + date_str_buoys + '_' + rf_param_str + '.dat'
file_IB_SAR = path_IB_SAR + 'IB_' + date_str_SAR + '_' + rf_param_str + '.dat'
file_MA_buoys = path_MA_buoys + 'MA_' + date_str_buoys + '_' + rf_param_str + '.dat'
file_MA_SAR = path_MA_SAR + 'MA_' + date_str_SAR + '_' + rf_param_str + '.dat'
#
df_IB_buoys = pd.read_csv(file_IB_buoys, delimiter = '\t')
df_IB_SAR = pd.read_csv(file_IB_SAR, delimiter = '\t')
df_MA_buoys = pd.read_csv(file_MA_buoys, delimiter = '\t')
df_MA_SAR = pd.read_csv(file_MA_SAR, delimiter = '\t')
#
df_IB_buoys = df_IB_buoys.dropna(how = 'all', axis = 1)
df_IB_SAR = df_IB_SAR.dropna(how = 'all', axis = 1)
df_MA_buoys = df_MA_buoys.dropna(how = 'all', axis = 1)
df_MA_SAR = df_MA_SAR.dropna(how = 'all', axis = 1)
#
Var_importances_IB_buoys = np.full((10, len(RF_variables)), np.nan)
Var_importances_IB_SAR = np.full((10, len(RF_variables)), np.nan)
Var_importances_MA_buoys = np.full((10, len(RF_variables)), np.nan)
Var_importances_MA_SAR = np.full((10, len(RF_variables)), np.nan)
#
for v in range(0, len(RF_variables)):
	var = RF_variables[v]
	Var_importances_IB_buoys[:, v] = df_IB_buoys[var]
	Var_importances_IB_SAR[:, v] = df_IB_SAR[var]
	Var_importances_MA_buoys[:, v] = df_MA_buoys[var]
	Var_importances_MA_SAR[:, v] = df_MA_SAR[var]
#############################################################################################################
# Figure
#############################################################################################################
plt.figure()
plt.rc('xtick', labelsize = sizefont_feature_importances)
plt.rc('ytick', labelsize = sizefont_feature_importances)
fig, ax1 = plt.subplots(2, 3, figsize = (30, 20), facecolor = 'w', edgecolor = 'k')
fig.subplots_adjust(hspace = 0.25, wspace = 0.2)
#####
ax = plt.subplot(231)
colors = colorscale(np.linspace(0, 1, len(RF_variables)))
patches = []
for var in range(0, len(RF_variables)):
        bottom_value = np.sum(Var_importances_IB_buoys[:, 0:var], axis = 1)
        p = plt.bar(np.arange(10) + 1, Var_importances_IB_buoys[:, var], width, bottom = bottom_value, color = colors[var], edgecolor = 'none')
        patches.append(mpatches.Patch(color = colors[var], label = RF_variables_label[var]))
#
ax.set_title('Direction \n models trained with buoy observations', fontsize = sizefont_feature_importances, fontweight = 'bold')
ax.set_ylabel('Relative importance of predictor variables', fontsize = sizefont_feature_importances)
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont_feature_importances)
ax.set_xlim([0.5, 10.5])
ax.set_ylim([0, 1.00])
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_yticks(np.arange(0, 1.1, step = 0.1))
ax.text(-0.13, -0.1, 'a)', fontsize = sizefont_feature_importances * 1.2, color = 'k', transform = ax.transAxes)
#####
ax = plt.subplot(232)
colors = colorscale(np.linspace(0, 1, len(RF_variables)))
patches = []
for var in range(0, len(RF_variables)):
        bottom_value = np.sum(Var_importances_MA_buoys[:, 0:var], axis = 1)
        p = plt.bar(np.arange(10) + 1, Var_importances_MA_buoys[:, var], width, bottom = bottom_value, color = colors[var], edgecolor = 'none')
        patches.append(mpatches.Patch(color = colors[var], label = RF_variables_label[var]))
#
ax.set_title('Speed \n models trained with buoy observations', fontsize = sizefont_feature_importances, fontweight = 'bold')
ax.set_ylabel('Relative importance of predictor variables', fontsize = sizefont_feature_importances)
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont_feature_importances)
ax.set_xlim([0.5, 10.5])
ax.set_ylim([0, 1.00])
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_yticks(np.arange(0, 1.1, step = 0.1))
ax.text(-0.13, -0.1, 'c)', fontsize = sizefont_feature_importances * 1.2, color = 'k', transform = ax.transAxes)
#####
ax = plt.subplot(234)
colors = colorscale(np.linspace(0, 1, len(RF_variables)))
patches = []
for var in range(0, len(RF_variables)):
        bottom_value = np.sum(Var_importances_IB_SAR[:, 0:var], axis = 1)
        p = plt.bar(np.arange(10) + 1, Var_importances_IB_SAR[:, var], width, bottom = bottom_value, color = colors[var], edgecolor = 'none')
        patches.append(mpatches.Patch(color = colors[var], label = RF_variables_label[var]))
#
ax.set_title('Direction \n models trained with SAR observations', fontsize = sizefont_feature_importances, fontweight = 'bold')
ax.set_ylabel('Relative importance of predictor variables', fontsize = sizefont_feature_importances)
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont_feature_importances)
ax.set_xlim([0.5, 10.5])
ax.set_ylim([0, 1.00])
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_yticks(np.arange(0, 1.1, step = 0.1))
ax.text(-0.13, -0.1, 'b)', fontsize = sizefont_feature_importances * 1.2, color = 'k', transform = ax.transAxes)
#####
ax = plt.subplot(235)
colors = colorscale(np.linspace(0, 1, len(RF_variables)))
patches = []
for var in range(0, len(RF_variables)):
        bottom_value = np.sum(Var_importances_MA_SAR[:, 0:var], axis = 1)
        p = plt.bar(np.arange(10) + 1, Var_importances_MA_SAR[:, var], width, bottom = bottom_value, color = colors[var], edgecolor = 'none')
        patches.append(mpatches.Patch(color = colors[var], label = RF_variables_label[var]))
#
ax.set_title('Speed \n models trained with SAR observations', fontsize = sizefont_feature_importances, fontweight = 'bold')
ax.set_ylabel('Relative importance of predictor variables', fontsize = sizefont_feature_importances)
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont_feature_importances)
ax.set_xlim([0.5, 10.5])
ax.set_ylim([0, 1.00])
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_yticks(np.arange(0, 1.1, step = 0.1))
ax.text(-0.13, -0.1, 'd)', fontsize = sizefont_feature_importances * 1.2, color = 'k', transform = ax.transAxes)
#####
ax = plt.subplot(233)
ax.legend(handles = patches, fontsize = sizefont_legend_feature_importances, bbox_to_anchor = (0.9, 1.03))
ax.axis('off')
#####
ax = plt.subplot(236)
ax.axis('off')
#####
plt.savefig(path_output + 'Predictor_importances_scikitlearn_impurity_decrease.png', bbox_inches = 'tight')
plt.clf()
