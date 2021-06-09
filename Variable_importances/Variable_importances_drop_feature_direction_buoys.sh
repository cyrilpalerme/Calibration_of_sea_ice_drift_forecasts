#$ -S /bin/bash
#$ -l h_rt=05:00:00
#$ -q research-el7.q
#$ -l h_vmem=10G
#$ -t 1-10
#$ -o /lustre/storeB/users/cyrilp/Svalnav/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/cyrilp/Svalnav/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -cwd

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python/3.7.2

cat > "/lustre/storeB/users/cyrilp/Svalnav/data_processing_files/PROG/Drop_IB_buoys_2013_2020_""$SGE_TASK_ID"".py" << EOF

################################################
import matplotlib
matplotlib.use('Agg')
import os
from netCDF4 import Dataset
import numpy as np
import glob
from datetime import datetime, timedelta
import cmath
import time
import sys
import csv
import random
import pandas as pd
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
function_path='/lustre/storeB/users/cyrilp/Python_functions/'
sys.path.insert(0, function_path)
from julian_day import *
###
start_time = time.time()
#############################################################################################################
# Constants
#############################################################################################################
path_data = '/lustre/storeB/project/copernicus/svalnav/April_2021/RF_training_data/Trained_with_buoys/201306_202005/'
path_output = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Tuning/Results_without_dayofyear/Direction/Drop_feature_training_period_buoys/'
#
date_min = 20130606 # It must be a Thursday
date_max = 20200528
#
lead_time_start = $SGE_TASK_ID - 1
lead_time_end = lead_time_start + 1
leadtime_str = str(lead_time_start) + '-' + str(lead_time_end)
#
Test_size = 0.2
#
# Random forest parameters
#
bootstrap = True
max_depth = None
n_estimators = 200
min_samples_leaf = 1
min_samples_split = 2
n_jobs = 1
max_features = 3
random_state = np.arange(50) + 100
#
rf_param_str = 'nestimators_' + str(n_estimators) + '_maxfeatures_' + str(max_features) +  '_bootstrap_' + str(bootstrap) + '_maxdepth_' + str(max_depth) + \
	       '_minsamplessplit_' + str(min_samples_split) + '_minsamplesleaf_' + str(min_samples_leaf)
#
feat_list = ['ECMWF_wd10m', 'ECMWF_ws10m', 'distance_to_land', 'OSISAF_SIC', 'T4_drift_initial_bearing', 'T4_drift_magnitude', 'T4_fice', 'T4_hice', 'xc', 'yc', 'all']
#############################################################################################################
# Date selection
#############################################################################################################
for dat in range(0, 10):
	#
	model = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth, min_samples_split = min_samples_split,\
	min_samples_leaf = min_samples_leaf, bootstrap = bootstrap, n_jobs = n_jobs, random_state = random_state[dat])
	#
	st_date = datetime.datetime.strptime(str(date_min), '%Y%m%d')
	start_dates = []
	target_dates_start = []
	target_dates_end = []
	ini_dates = []
	while st_date <= datetime.datetime.strptime(str(date_max), '%Y%m%d'):
		start_dates.append(st_date.strftime('%Y%m%d'))
		target_dates_start.append((st_date + timedelta(days = lead_time_start)).strftime('%Y%m%d%H%M%S'))
		target_dates_end.append((st_date + timedelta(days = lead_time_start + 1)).strftime('%Y%m%d%H%M%S'))
		ini_dates.append((st_date - timedelta(days = 1)).strftime('%Y%m%d'))
		st_date = st_date + timedelta(days = 7)
	#
	size = int(Test_size * len(start_dates))
	idx_test = random.sample(range(len(start_dates)), size)
	idx_train = np.arange(len(start_dates))
	idx_train = [x for x in idx_train if x not in idx_test]
	#
	start_dates_test = np.array(start_dates)[idx_test]
	start_dates_train = np.array(start_dates)[idx_train]
	#############################################################################################################
	# Load data and select training and testing datasets
	#############################################################################################################
	filename_data = 'Buoys_T4grid_daily_' + str(date_min) + '-' + str(date_max) + '_' + str(lead_time_start) + '-' + str(lead_time_end) + '_days.dat'
	df = pd.read_csv(path_data + filename_data, delimiter = '\t')
	df = df.dropna(how='all', axis=1)  # Remove unnamed columns containing nan
	#
	Buoys_drift_magnitude = np.array(df['Buoys_drift_magnitude'])
	Buoys_drift_initial_bearing = np.array(df['Buoys_drift_initial_bearing'])
	Selected_buoys = np.logical_and(np.logical_and(Buoys_drift_magnitude > 100, Buoys_drift_magnitude < 100 * 1000), Buoys_drift_initial_bearing != 0)
	df = df[Selected_buoys == True]
	#
	print('Select train and test data', time.time() - start_time)
	del idx_test
	del idx_train
	#
	for sdt in range(0, len(start_dates_test)):
		if sdt == 0:
			idx_test = np.where(df['Start_date'] == float(start_dates_test[sdt]))[0]
		else:
			idx_test = np.hstack((idx_test, np.where(df['Start_date'] == float(start_dates_test[sdt]))[0]))
	#
	idx_bool = np.full(np.shape(df['Start_date']), False)
	idx_bool[idx_test] = True 
	#
	df_RF_test = df[idx_bool == True]
	df_RF_train = df[idx_bool == False]
	#
	T4_fice_test = df_RF_test['T4_fice']
	df_RF_test = df_RF_test[T4_fice_test >= 0.1]
	##########
	for drop_feat in range(0, len(feat_list)):
		#
		Target_train = np.array(df_RF_train['Buoys_drift_initial_bearing'])
		Reference_train = np.array(df_RF_train['T4_drift_initial_bearing'])
		df_feat_train = df_RF_train.drop('Buoys_drift_magnitude', axis = 1)
		df_feat_train = df_feat_train.drop('Buoys_drift_initial_bearing', axis = 1)
		if feat_list[drop_feat] != 'all':
			df_feat_train = df_feat_train.drop(feat_list[drop_feat], axis = 1)
		df_feat_train = df_feat_train.drop('Start_date', axis = 1)
		Features_train = df_feat_train.sort_index(axis = 1)
		Feature_train_names = list(Features_train.columns)
		Features_train = np.array(Features_train)	
		#
		Target_test = np.array(df_RF_test['Buoys_drift_initial_bearing'])
		Reference_test = np.array(df_RF_test['T4_drift_initial_bearing'])
		df_feat_test = df_RF_test.drop('Buoys_drift_magnitude', axis = 1)
		df_feat_test = df_feat_test.drop('Buoys_drift_initial_bearing', axis = 1)
		if feat_list[drop_feat] != 'all':
			df_feat_test = df_feat_test.drop(feat_list[drop_feat], axis = 1)
		df_feat_test = df_feat_test.drop('Start_date', axis = 1)
		Features_test = df_feat_test.sort_index(axis = 1)
		Feature_test_names = list(Features_test.columns)
		Features_test = np.array(Features_test)	
		#############################################################################################################
		# RF models
		#############################################################################################################
		model.fit(Features_train, Target_train)
		#####################################################################################################
		# Tests performances
		#####################################################################################################
		# Make prediction on the test set
		Test_forecasts_alltrees_deg = np.array([tree.predict(Features_test) for tree in model.estimators_])
		Test_forecasts_alltrees_rad = Test_forecasts_alltrees_deg * cmath.pi / 180
		#
		Test_forecasts_alltrees_com = np.full(np.shape(Test_forecasts_alltrees_rad), np.nan, dtype = complex)
		for i in range(0, np.shape(Test_forecasts_alltrees_rad)[0]):
			for j in range(0, np.shape(Test_forecasts_alltrees_rad)[1]):
				Test_forecasts_alltrees_com[i,j] = cmath.rect(1, Test_forecasts_alltrees_rad[i,j])
		Test_forecasts_mean_com = np.mean(Test_forecasts_alltrees_com, axis = 0)
		#
		Test_forecasts_mean_deg = np.full(len(Test_forecasts_mean_com), np.nan)
		for i in range(0, len(Test_forecasts_mean_com)):
			Test_forecasts_mean_deg[i] = cmath.phase(Test_forecasts_mean_com[i]) * 180 / cmath.pi
		Test_forecasts_mean_deg[Test_forecasts_mean_deg < 0] = 360 + Test_forecasts_mean_deg[Test_forecasts_mean_deg < 0]
		Test_forecasts = np.copy(Test_forecasts_mean_deg)		
		#
		Reference_test[Reference_test < 0] = np.nan
		Target_test[Target_test < 0] = np.nan
		#
		reference_errors = abs(Reference_test - Target_test)
		forecast_errors = abs(Test_forecasts - Target_test)
		reference_errors[reference_errors > 180] = 360 - reference_errors[reference_errors > 180]
		forecast_errors[forecast_errors > 180] = 360 - forecast_errors[forecast_errors > 180]
		Diff_errors = (reference_errors - forecast_errors) >= 0
		Forecast_improvement = np.sum(Diff_errors)*1.0 / len(Diff_errors)
		#
		Mean_reference_error = str(round(np.nanmean(reference_errors), 2))
		Mean_forecast_error = str(round(np.nanmean(forecast_errors), 2))
		#
		Nb_forecasts = str(len(Target_train) + len(Target_test))
		#####################################################################################################
		# Saving
		#####################################################################################################
		str_results =  str(lead_time_start) + '\t' + str(lead_time_end) + '\t' + Nb_forecasts + '\t' + \
			       Mean_reference_error + '\t' + Mean_forecast_error + '\t' + str(round(Forecast_improvement * 100, 2)) + '\t'
		#
		file_output = 'IB_' + str(date_min) + '-' + str(date_max) + '_ts_' + str(Test_size) + '_' + rf_param_str + '_' + leadtime_str + '_dropfeat_' + feat_list[drop_feat] + '.dat'
		Output = open(path_output + file_output,'a')
		#
		if dat == 0:
			str_legend = 'Lead_time_start' + '\t' + 'Lead_time_end' + '\t' + 'Number_of_forecasts' + '\t' + \
				     'Mean_reference_error' + '\t' + 'Mean_forecast_error' + '\t' + 'Fraction_of_forecasts_improved' + '\t'
			Output.write(str_legend + '\n')
		#
		Output.write(str_results + '\n')
		Output.close()
#####################################################################################################
EOF
python3 "/lustre/storeB/users/cyrilp/Svalnav/data_processing_files/PROG/Drop_IB_buoys_2013_2020_""$SGE_TASK_ID"".py"
