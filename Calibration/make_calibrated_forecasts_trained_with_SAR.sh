#$ -S /bin/bash
#$ -l h_rt=01:00:00
#$ -q research-el7.q 
#$ -l h_vmem=10G
#$ -t 1-1
#$ -o /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -cwd

module load Python/3.7.2

echo "Got $NSLOTS slots for job $SGE_TASK_ID."
###################################################################################################
datemin=20210531
#
n1=$(($SGE_TASK_ID - 1))
n2=$(($SGE_TASK_ID - 2))
#
date_1d=$(date -d "$datemin + $n1 days" +%Y%m%d)
date_2d=$(date -d "$datemin + $n2 days" +%Y%m%d)
#
path_ECMWF_on_T4grid="/lustre/storeB/project/copernicus/svalnav/Data_operational_ice_drift_forecasts/Pan_Arctic/ECMWF_10m_wind_on_T4grid/"
path_OSISAF_on_T4grid="/lustre/storeB/project/copernicus/svalnav/Data_operational_ice_drift_forecasts/Pan_Arctic/OSISAF_SIC_on_T4grid/"
path_T4_direction_and_speed="/lustre/storeB/project/copernicus/svalnav/Data_operational_ice_drift_forecasts/Pan_Arctic/TOPAZ4/"
#
cat > "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/make_calibrated_forecasts_PA_reforecasts_"$SGE_TASK_ID".py" << EOF
##################################################
import os
import numpy as np
import numpy.ma as ma
import pickle
import sys
import cmath
import pandas as pd
from math import *
from netCDF4 import Dataset
from pyproj import Proj, transform
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
function_path='/lustre/storeB/users/cyrilp/Python_functions/'
sys.path.insert(0, function_path)
from do_kdtree import *
from destination_coordinates import *
##################################################
# Constants
##################################################
path_models = '/lustre/storeB/project/copernicus/svalnav/April_2021/RF_models/Trained_with_SAR/'
#
file_OSISAF = '$path_OSISAF_on_T4grid' + 'OSISAF_SIC_distance_ice_edge_' + '$date_2d' + '.nc'
file_ECMWF = '$path_ECMWF_on_T4grid' + 'ECMWF_wind_forecasts_' + '$date_1d' + '.nc'
file_T4 = '$path_T4_direction_and_speed' + 'TOPAZ4_daily_' + '$date_1d' + '.nc'
file_LSM = '/lustre/storeB/project/copernicus/svalnav/Data/TOPAZ4/TOPAZ4_land_sea_mask.nc'
#
# Projections
#
proj_EPSG4326 = Proj(init = 'epsg:4326')
proj_T4 = Proj('+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere')
#
# RF parameters
#
date_min = '20180104'
date_max = '20200528'
n_estimators = 200
max_features = 3
bootstrap = True
max_depth = None
min_samples_leaf = 1
min_samples_split = 2
n_jobs = 1
#
date_str = date_min + '-' + date_max
rf_str = 'nestimators_' + str(n_estimators) + '_maxfeatures_' + str(max_features) +  '_bootstrap_' + str(bootstrap) + '_maxdepth_' + str(max_depth) + 	 '_minsamplessplit_' + str(min_samples_split) + '_minsamplesleaf_' + str(min_samples_leaf)
#
# TOPAZ4 grid
#
x_T4grid = np.linspace(-38, 38, 609) * 100 * 1000
y_T4grid = np.linspace(-55, 55, 881) * 100 * 1000
xx_T4grid, yy_T4grid = np.meshgrid(x_T4grid, y_T4grid)
xx_yy_T4grid = np.dstack([xx_T4grid.ravel(), yy_T4grid.ravel()])[0]
lon_T4grid, lat_T4grid = transform(proj_T4, proj_EPSG4326, xx_T4grid, yy_T4grid)
#################################################
# Functions
#################################################
# Find nearest neighbor point index
def do_kdtree(combined_x_y_arrays,points):
	mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
	dist, indexes = mytree.query(points)
	return indexes
###
def destination_coordinates(lat1, lon1, initial_bearing, distance): # distance in meters
	RT = 6371.0 # km, but distance must be in meters
	d_div_r = (distance * 0.001) / RT
	lat1 = math.radians(lat1)
	lon1 = math.radians(lon1)
	bearing = math.radians(initial_bearing)
	lat2 = asin(sin(lat1) * cos(d_div_r) + cos(lat1) * sin(d_div_r) * cos(bearing))
	lon2 = lon1 + atan2(sin(bearing) * sin(d_div_r) * cos(lat1), cos(d_div_r) - sin(lat1) * sin(lat2))
	lat2 = math.degrees(lat2)
	lon2 = math.degrees(lon2)
	return(lat2, lon2)
##################################################
# RF variables
##################################################
T4_variables = ['T4_drift_magnitude', 'T4_drift_initial_bearing', 'T4_fice', 'T4_hice']
OSISAF_variables = ['OSISAF_SIC']
ECMWF_variables = ['ECMWF_ws10m', 'ECMWF_wd10m']
Geolocation_variables = ['xc', 'yc', 'distance_to_land']
feature_variables_list = T4_variables + OSISAF_variables + ECMWF_variables + Geolocation_variables
feature_variables_list.sort()
xc_pos = feature_variables_list.index('xc')
yc_pos = feature_variables_list.index('yc')
##################################################
# Load data
##################################################
Feat_variables = {}
#
nc_LSM = Dataset(file_LSM, 'r')
nc_T4 = Dataset(file_T4, 'r')
nc_OSISAF = Dataset(file_OSISAF, 'r')
nc_ECMWF = Dataset(file_ECMWF, 'r')
#
time_T4 = nc_T4.variables['time'][:]
Feat_variables['xc'] = np.ndarray.flatten(xx_T4grid)
Feat_variables['yc'] = np.ndarray.flatten(yy_T4grid)
Feat_variables['distance_to_land'] = np.ndarray.flatten(nc_LSM.variables['distance_to_land'][:,:])
#
initial_bearing_calib = np.full((len(time_T4), len(y_T4grid), len(x_T4grid)), np.nan)
magnitude_calib = np.full((len(time_T4), len(y_T4grid), len(x_T4grid)), np.nan)
#
for lt in range(0, len(time_T4)):
	target_dates_start = (datetime.strptime('$date_1d', '%Y%m%d') + timedelta(days = lt)).strftime('%Y%m%d%H%M%S')
	target_dates_end = (datetime.strptime('$date_1d', '%Y%m%d') + timedelta(days = lt + 1)).strftime('%Y%m%d%H%M%S')
	#########
	for v in range(0, len(T4_variables)):
		source_var = T4_variables[v]
		source =  source_var[0:source_var.find('_')]
		var = source_var.replace(source + '_', '')
		Feat_variables[source_var] = np.ndarray.flatten(nc_T4.variables[var][lt,:,:])
	#
	for v in range(0, len(OSISAF_variables)):
		source_var = OSISAF_variables[v]
		source =  source_var[0:source_var.find('_')]
		var = source_var.replace(source + '_', '')
		Feat_variables[source_var] = np.ndarray.flatten(nc_OSISAF.variables[var][:,:])
	#
	for v in range(0, len(ECMWF_variables)):
		source_var = ECMWF_variables[v]
		source =  source_var[0:source_var.find('_')]
		var = source_var.replace(source + '_', '')
		Feat_variables[source_var] = np.ndarray.flatten(nc_ECMWF.variables[var][lt,:,:])
	#
	for v in range(0, len(feature_variables_list)):
		source_var = feature_variables_list[v]
		if v == 0:
			Feature_variables = np.expand_dims(Feat_variables[source_var], axis = 1)
		else:
			Feature_variables = np.concatenate((Feature_variables, np.expand_dims(Feat_variables[source_var], axis = 1)), axis = 1)
	#########
	idx_any_nan = np.any(np.isnan(Feature_variables), axis = 1)
	Feature_variables = Feature_variables[idx_any_nan == False]
	xxc = Feature_variables[:, xc_pos]
	yyc = Feature_variables[:, yc_pos]
	xxc_yyc = np.dstack([xxc.ravel(), yyc.ravel()])[0]
	#############################################################################################################
	# Calibration
	#############################################################################################################
	lt_str = str(lt) + '-' + str(lt + 1) + '_days'
	file_model_IB = path_models + 'Direction_201801_202005_2_percents_without_dayofyear/RF_model_IB_' + rf_str + '_' +  date_str + '_' + lt_str + '.pkl'
	file_model_MA = path_models + 'Speed_201801_202005_2_percents_without_dayofyear/RF_model_MA_' + rf_str + '_' +  date_str + '_' + lt_str + '.pkl'
	#####
	with open(file_model_IB, 'rb') as file:
		model_initial_bearing = pickle.load(file)
	#
	calib_initial_bearing_alltrees_rad = np.array([tree.predict(Feature_variables) for tree in model_initial_bearing.estimators_]) * cmath.pi / 180
	calib_initial_bearing_alltrees_com = np.full(np.shape(calib_initial_bearing_alltrees_rad), np.nan, dtype = complex)
	for i in range(0, np.shape(calib_initial_bearing_alltrees_rad)[0]):
		for j in range(0, np.shape(calib_initial_bearing_alltrees_rad)[1]):
			calib_initial_bearing_alltrees_com[i,j] = cmath.rect(1, calib_initial_bearing_alltrees_rad[i,j])
	calib_initial_bearing_mean_com = np.mean(calib_initial_bearing_alltrees_com, axis = 0)
	#
	calib_initial_bearing = np.full(len(calib_initial_bearing_mean_com), np.nan)
	for i in range(0, len(calib_initial_bearing_mean_com)):
		calib_initial_bearing[i] = cmath.phase(calib_initial_bearing_mean_com[i]) * 180 / cmath.pi
	calib_initial_bearing[calib_initial_bearing < 0] = 360 + calib_initial_bearing[calib_initial_bearing < 0]
	del model_initial_bearing
	#####
	with open(file_model_MA, 'rb') as file:
		model_magnitude = pickle.load(file)
	calib_magnitude = model_magnitude.predict(Feature_variables)
	del model_magnitude
	#
	idx_pos = do_kdtree(xx_yy_T4grid, xxc_yyc)
	calibrated_initial_bearing = np.full(len(xx_T4grid.ravel()), np.nan)
	calibrated_magnitude = np.full(len(xx_T4grid.ravel()), np.nan)
	calibrated_initial_bearing[idx_pos] = calib_initial_bearing
	calibrated_magnitude[idx_pos] = calib_magnitude
	#
	initial_bearing_calib[lt,:,:] = calibrated_initial_bearing.reshape(len(y_T4grid), len(x_T4grid))
	magnitude_calib[lt,:,:] = calibrated_magnitude.reshape(len(y_T4grid), len(x_T4grid)) 
##################################################
# Land sea mask and sea ice cover
##################################################
fice_T4 = nc_T4.variables['fice'][:,:,:]
initial_bearing_calib[fice_T4 < 0.1] = np.nan
magnitude_calib[fice_T4 < 0.1] = np.nan
#
initial_bearing_calib = ma.masked_array(initial_bearing_calib, mask = np.isnan(fice_T4) == True)
magnitude_calib = ma.masked_array(magnitude_calib, mask = np.isnan(fice_T4) == True)
##################################################
## NetCDF file Pan Arctic
##################################################
output_filename = 'calibrated_ice_drift_forecasts_' + '$date_1d' + '.nc'
output_netcdf = Dataset(output_filename, 'w', format='NETCDF4')
#
x = output_netcdf.createDimension('x', len(x_T4grid))
y = output_netcdf.createDimension('y', len(y_T4grid))
time = output_netcdf.createDimension('time', len(time_T4))
#
x = output_netcdf.createVariable('x', 'd', ('x'))
y = output_netcdf.createVariable('y', 'd', ('y'))
time = output_netcdf.createVariable('time', 'd', ('time'))
latitude = output_netcdf.createVariable('latitude', 'd', ('y','x'))
longitude = output_netcdf.createVariable('longitude', 'd', ('y','x'))
drift_magnitude = output_netcdf.createVariable('drift_magnitude', 'd', ('time', 'y', 'x'))
drift_direction = output_netcdf.createVariable('drift_direction', 'd', ('time', 'y', 'x'))
#
x.units = '100 km'
x.standard_name = 'projection_x_coordinate'
y.units = '100 km'
y.standard_name = 'projection_y_coordinate'
time.units = 'hour since 1950-1-1T00:00:00Z'
time.standard_name = 'time'
latitude.units = 'degrees_north'
latitude.standard_name = 'latitude'
longitude.units = 'degrees_east'
longitude.standard_name = 'longitude'
drift_magnitude.units = 'm/day'
drift_magnitude.standard_name = 'drift_magnitude'
drift_direction.units = 'degree (compass)'
drift_direction.standard_name = 'drift_direction'
#
x[:] = x_T4grid * 0.001 * 0.01
y[:] = y_T4grid * 0.001 * 0.01
time[:] = time_T4
longitude[:,:] = lon_T4grid
latitude[:,:] = lat_T4grid
drift_magnitude[:,:,:] = magnitude_calib
drift_direction[:,:,:] = initial_bearing_calib 
#
output_netcdf.title = "Calibrated sea ice drift forecasts"
output_netcdf.abstract = "Sea ice drift forecasts obtained from statistical models at a spatial resolution of 12.5 km and with daily time steps."
output_netcdf.grid_projection = "+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. ellps=sphere"
output_netcdf.area = "Pan Arctic"
output_netcdf.institution = "Norwegian Meteorological Institute"
output_netcdf.PI_name = "Cyril Palerme"
output_netcdf.contact = "cyril.palerme@met.no"
output_netcdf.bulletin_type = "Forecast"
output_netcdf.forecast_range = "10 days"
output_netcdf.close()
#
os.system('mv ' + output_filename + ' /lustre/storeB/project/copernicus/svalnav/April_2021/Calibrated_forecasts/RF_trained_with_SAR_201801_202005_2_percents_without_dayofyear/')
###################################################
EOF
python3 "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/make_calibrated_forecasts_PA_reforecasts_"$SGE_TASK_ID".py"

