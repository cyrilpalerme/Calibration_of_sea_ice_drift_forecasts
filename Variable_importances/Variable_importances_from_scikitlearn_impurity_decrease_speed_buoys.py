import matplotlib
matplotlib.use('Agg')
import os
from netCDF4 import Dataset
import numpy as np
import glob
from datetime import datetime, timedelta
import time
import sys
import pickle
import pandas as pd
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
#############################################################################################################
# Paths 
#############################################################################################################
path_data = '/lustre/storeB/project/copernicus/svalnav/April_2021/RF_training_data/Trained_with_buoys/201306_202005/'
path_output = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Tuning/Results_without_dayofyear/Speed/Variable_importances_buoys/'
#############################################################################################################
# Input parameters
#############################################################################################################
#
# Random forest parameters
#
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
model = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth, \
        min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, bootstrap = bootstrap, n_jobs = n_jobs)
#
# Datasets train - test
#
date_min_train = '20130606'
date_max_train = '20200528'
#
# Lead time
#
for lead_time_start in range(0, 10):
	lead_time_end = lead_time_start + 1
	leadtime_str = str(lead_time_start) + '-' + str(lead_time_end)
	print(leadtime_str)
	#############################################################################################################
	# Load data
	#############################################################################################################
	###
	### Dataset for training
	###
	file_input_train = 'Buoys_T4grid_daily_' + date_min_train + '-' + date_max_train + '_' + str(lead_time_start) + '-' + str(lead_time_end) + '_days.dat'
	#
	df_RF_train = pd.read_csv(path_data + file_input_train, delimiter = '\t')
	df_RF_train = df_RF_train.dropna(how='all', axis=1)  # Remove unnamed columns containing nan
	#
	Buoys_drift_magnitude = np.array(df_RF_train['Buoys_drift_magnitude'])
	Buoys_drift_initial_bearing = np.array(df_RF_train['Buoys_drift_initial_bearing'])
	Selected_buoys = np.logical_and(Buoys_drift_magnitude < 100 * 1000, Buoys_drift_initial_bearing != 0)
	df_RF_train = df_RF_train[Selected_buoys == True]
	#
	Target_train = np.array(df_RF_train['Buoys_drift_magnitude'])
	df_feat_train = df_RF_train.drop('Buoys_drift_magnitude', axis = 1)
	df_feat_train = df_feat_train.drop('Buoys_drift_initial_bearing', axis = 1)
	df_feat_train = df_feat_train.drop('Start_date', axis = 1)
	#
	Features_train = df_feat_train.sort_index(axis = 1)
	Feature_train_names = list(Features_train.columns)
	Features_train = np.array(Features_train)
	#####################################################################################################
	# Random forests
	#####################################################################################################
	model.fit(Features_train, Target_train)
	importances = list(model.feature_importances_) # Get numerical feature importances
	feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(Feature_train_names, importances)]
	feature_importances_sorted = sorted(feature_importances)        
	feature_importances_rank = sorted(feature_importances, key = lambda x: x[1], reverse = True)
	#####################################################################################################
	# Saving
	#####################################################################################################
	str_legend = 'lead_time' + '\t'
	str_results = leadtime_str + '\t'
	for fi in range(0, len(feature_importances_sorted)):
		str_legend = str_legend + feature_importances_sorted[fi][0] + '\t'
		str_results = str_results + str(feature_importances_sorted[fi][1]) + '\t'
	#
	file_output = 'MA_' + str(date_min_train) + '-' + str(date_max_train) + '_' + rf_param_str + '.dat'
	Output = open(path_output + file_output, 'a')
	#	
	if lead_time_start == 0:
		Output.write(str_legend + '\n')
	#
	Output.write(str_results + '\n')
	Output.close()
