#$ -S /bin/bash
#$ -l h_rt=00:20:00
#$ -q research-el7.q
#$ -l h_vmem=10G
#$ -t 1-10
#$ -o /lustre/storeB/users/cyrilp/Svalnav/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/cyrilp/Svalnav/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -cwd

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python/3.7.2

cat > "/lustre/storeB/users/cyrilp/Svalnav/data_processing_files/PROG/make_datasets_buoys_""$SGE_TASK_ID"".py" << EOF

################################################
import matplotlib
matplotlib.use('Agg')
import os
from netCDF4 import Dataset
import numpy as np
import glob
from datetime import datetime, timedelta
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
#
start_time = time.time()
#############################################################################################################
# Paths 
#############################################################################################################
path_output = '/lustre/storeB/project/copernicus/svalnav/April_2021/RF_training_data/Trained_with_buoys/201206_202005/'
path_data = {}
path_data['Buoys'] = '/lustre/storeB/project/copernicus/svalnav/Data/IABP/Data_T4grid_SIC10/'
path_data['OSISAF'] = '/lustre/storeB/project/nwp/SALIENSEAS/Ice_drift_calibration_Pan_Arctic/Data/OSISAF_SIC_on_T4grid/'
path_data['T4'] = '/lustre/storeB/project/nwp/SALIENSEAS/Ice_drift_calibration_Pan_Arctic/Data/TOPAZ4/'
path_data['ECMWF'] = '/lustre/storeB/project/nwp/SALIENSEAS/Ice_drift_calibration_Pan_Arctic/Data/ECMWF_10m_wind_on_T4grid/'
file_region_mask = '/lustre/storeB/users/cyrilp/Data/NSIDC_regional_mask/sio_2016_mask_TOPAZ4_grid.nc'
#
filename = {}
filename['Buoys'] = 'Buoys_'
filename['OSISAF'] = 'OSISAF_SIC_distance_ice_edge_'
filename['T4'] = 'TOPAZ4_daily_'
filename['ECMWF'] = 'ECMWF_wind_forecasts_'
#############################################################################################################
# Mask Canadian Archipelago
#############################################################################################################
nc_region_mask = Dataset(file_region_mask, 'r')
mask = nc_region_mask.variables['mask'][:,:]
mask_Canadian_Archipelago = np.zeros(np.shape(mask))
mask_Canadian_Archipelago[mask == 14] = 1
#############################################################################################################
# Input parameters
#############################################################################################################
# Lead time forecasts (days)
lead_time_start = $SGE_TASK_ID - 1
lead_time_end = lead_time_start + 1
# date min and date max in yyyymmdd
date_min = 20120607 # It must be a Thursday
date_max = 20200528
# Target variable (variable that we want to predict), syntax: source_variable
target_var_list = ['Buoys_drift_magnitude', 'Buoys_drift_initial_bearing']
# Feature variables (variables that are used for the prediction), syntax: source_variable 
feature_var_dimensions = ['xc', 'yc', 'Start_date', 'distance_to_land']
feature_var_initial_conditions = ['OSISAF_SIC']
feature_var_forecasts = ['ECMWF_ws10m', 'ECMWF_wd10m', 'T4_fice', 'T4_hice', 'T4_drift_magnitude', 'T4_drift_initial_bearing']
target_feat_str = target_var_list + feature_var_dimensions + feature_var_initial_conditions + feature_var_forecasts
target_feat_str.sort()
print('............................................................................................')
print('Lead time: ' + str(lead_time_start)+' days')
print('Period from ' + str(date_min)+' to '+str(date_max))
print('Target variables: ' + str(target_var_list))
print('Feature variables dimensions: ' + str(feature_var_dimensions))
print('Feature variables initial conditions: ' + str(feature_var_initial_conditions))
print('Feature variables forecasts: ' + str(feature_var_forecasts))
###
st_date = datetime.datetime.strptime(str(date_min), '%Y%m%d')
start_dates = []
target_dates = []
ini_dates = []
while st_date <= datetime.datetime.strptime(str(date_max), '%Y%m%d'):
	start_dates.append(st_date.strftime('%Y%m%d'))
	target_dates.append((st_date + timedelta(days = lead_time_start)).strftime('%Y%m%d'))
	ini_dates.append((st_date - timedelta(days = 1)).strftime('%Y%m%d'))
	st_date = st_date + timedelta(days = 7)
###
file_output = path_output + 'Buoys_T4grid_daily_' + str(date_min) + '-' + str(date_max) + '_' + str(lead_time_start) + '-' + str(lead_time_end) + '_days' + '.dat'
###
file_T4_distance_to_land = '/lustre/storeB/project/copernicus/svalnav/Data/TOPAZ4/TOPAZ4_land_sea_mask.nc'
nc_distance_to_land = Dataset(file_T4_distance_to_land, 'r')
distance_to_land_T4 = nc_distance_to_land.variables['distance_to_land'][:,:]
distance_to_land_T4[mask_Canadian_Archipelago == 1] = np.nan
##############################################################################################################
## Test dataset
## Target variable
##############################################################################################################
for sd in range(0, len(start_dates)):
	try:
		Target_feat_var = {}
		td_test_str = target_dates[sd][0:4] + '/' + target_dates[sd][4:6] + '/'
		file_target_var = path_data['Buoys'] + td_test_str + filename['Buoys'] + target_dates[sd] + '.nc'
		exist_file = []
		exist_file.append(os.path.isfile(file_target_var)) 
		if exist_file[len(exist_file) - 1] == True:
			#
			nc_target = Dataset(file_target_var, 'r')
			start_date_var = int(start_dates[sd][4:6])	
			#
			for tv in range(0, len(target_var_list)):
				target_var = target_var_list[tv]
				var_ta_str = target_var[target_var.find('_') + 1 : len(target_var)]
				target_variable_tv = nc_target.variables[var_ta_str][:,:]
				Target_feat_var[target_var] = np.ndarray.flatten(target_variable_tv)
		#####################################################################################
		# Feature variables
		#####################################################################################
		for fic in range(0, len(feature_var_initial_conditions)):
			feature_var = feature_var_initial_conditions[fic]
			source_fe_str = feature_var[0:feature_var.find('_')]
			var_fe_str = feature_var[feature_var.find('_') + 1 : len(feature_var)]
			file_feature_var = path_data[source_fe_str] + ini_dates[sd][0:4] + '/' + ini_dates[sd][4:6] + '/' + filename[source_fe_str] + ini_dates[sd] + '.nc'
			exist_file.append(os.path.isfile(file_feature_var))
			if exist_file[len(exist_file) - 1] == True:
				nc_feature = Dataset(file_feature_var, 'r')				
				Target_feat_var[feature_var] = np.ndarray.flatten(nc_feature.variables[var_fe_str][:,:])		
				#
		xcoord = nc_feature.variables['xc'][:]
		ycoord = nc_feature.variables['yc'][:]
		xxc, yyc = np.meshgrid(xcoord, ycoord)
		Target_feat_var['xc'] = np.ndarray.flatten(xxc)
		Target_feat_var['yc'] = np.ndarray.flatten(yyc)
		Target_feat_var['distance_to_land'] = np.ndarray.flatten(distance_to_land_T4)
		#
		for ffo in range(0, len(feature_var_forecasts)):
			feature_var = feature_var_forecasts[ffo]
			source_fe_str = feature_var[0:feature_var.find('_')]
			var_fe_str = feature_var[feature_var.find('_') + 1 : len(feature_var)]
			file_feature_var = path_data[source_fe_str] + filename[source_fe_str] + start_dates[sd] + '.nc'
			exist_file.append(os.path.isfile(file_feature_var))
			if exist_file[len(exist_file) - 1] == True:
				nc_feature = Dataset(file_feature_var, 'r')
				Target_feat_var[feature_var] = np.ndarray.flatten(nc_feature.variables[var_fe_str][lead_time_start,:,:])							
		#
		Target_feat_var['Start_date'] = int(start_dates[sd])
		#####################################################################################
		# Select grid points without nan and saving
		#####################################################################################
		if (np.all(exist_file) == True):
			df_tar_feat = pd.DataFrame(Target_feat_var)
			idx_any_nan = pd.DataFrame.any(np.isnan(df_tar_feat), axis = 1)
			idx_any_0 = np.logical_or(df_tar_feat['Buoys_drift_initial_bearing'] == 0, df_tar_feat['Buoys_drift_magnitude'] == 0)
			idx_all = np.logical_and(idx_any_nan == False, idx_any_0 == False)
			#
			if np.sum(idx_all == True) > 0:
				df_tar_feat_select = df_tar_feat[idx_all == True]
				#
				if sd == 0:
					with open(file_output, 'a') as f:
						df_tar_feat_select.to_csv(f, sep = '\t', header = True, index = False, index_label = None)
					f.close()
				else:
					with open(file_output, 'a') as f:
						df_tar_feat_select.to_csv(f, sep = '\t', header = False, index = False, index_label = None)
					f.close()
	except:
		pass
##############################################################################################################
EOF

python3 "/lustre/storeB/users/cyrilp/Svalnav/data_processing_files/PROG/make_datasets_buoys_""$SGE_TASK_ID"".py"
