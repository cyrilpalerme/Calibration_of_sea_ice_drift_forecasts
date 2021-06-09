#$ -S /bin/bash
#$ -l h_rt=01:00:00
#$ -q research-el7.q
#$ -l h_vmem=3G
#$ -t 1-1
#$ -o /lustre/storeB/users/cyrilp/Svalnav/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/cyrilp/Svalnav/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -cwd

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

source /modules/centos7/conda/Feb2021/etc/profile.d/conda.sh
conda activate production

cat > "/lustre/storeB/users/cyrilp/Svalnav/data_processing_files/PROG/Maps_sampling_""$SGE_TASK_ID"".py" << EOF

################################################
import matplotlib
matplotlib.use('Agg')
import os
import glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.feature import NaturalEarthFeature, LAND, COASTLINE
###########################################################
# Constants
###########################################################
path_output = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Verification/Figures/Maps_observations/'
path_LSM = '/lustre/storeB/project/copernicus/svalnav/Data/TOPAZ4/'
path_buoys = '/lustre/storeB/project/copernicus/svalnav/Data/IABP/Data_T4grid_SIC10/'
path_buoy_train = '/lustre/storeB/project/copernicus/svalnav/April_2021/RF_training_data/Trained_with_buoys/201306_202005/'
path_T4_train = '/lustre/storeB/project/nwp/SALIENSEAS/Ice_drift_calibration_Pan_Arctic/Data/TOPAZ4/'
path_T4_eval = '/lustre/storeB/project/copernicus/svalnav/Data_operational_ice_drift_forecasts/Pan_Arctic/TOPAZ4/'
path_SAR = '/lustre/storeB/project/copernicus/svalnav/Data_operational_ice_drift_forecasts/Pan_Arctic/SAR_v2/'
path_SAR_train = '/lustre/storeB/project/copernicus/svalnav/April_2021/RF_training_data/Trained_with_SAR/201801_202005_2_percents/'
path_region_mask = '/lustre/storeB/users/cyrilp/Data/NSIDC_regional_mask/'
#
date_min_train_buoys = '20130606'
date_max_train_buoys = '20200528'
date_min_train_SAR = '20180104'
date_max_train_SAR = '20200528'
date_min_eval = '20200601'
date_max_eval = '20210531'
#
sizefont = 16
###
colorscale_N = 'gist_stern_r'
#
levels_N_buoys = [0, 1, 2, 5, 10, 20, 30]
levels_N_SAR_eval = [0, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450]
levels_N_SAR_train = [0, 1, 2, 4, 6, 8, 10]
#
norm_N_buoys = BoundaryNorm(levels_N_buoys, 256)
norm_N_SAR_eval = BoundaryNorm(levels_N_SAR_eval, 256)
norm_N_SAR_train = BoundaryNorm(levels_N_SAR_train, 256)
###########################################################
# Region mask
###########################################################
file_region_mask = path_region_mask + 'sio_2016_mask_TOPAZ4_grid.nc'
nc_region_mask = Dataset(file_region_mask, 'r')
mask = nc_region_mask.variables['mask'][:,:]
mask_Canadian_Archipelago = np.zeros(np.shape(mask))
mask_Canadian_Archipelago[mask == 14] = 1
###########################################################
# Buoys evaluation
###########################################################
file_LSM = path_LSM + 'TOPAZ4_land_sea_mask.nc'
nc_LSM = Dataset(file_LSM, 'r')
distance_to_land = nc_LSM.variables['distance_to_land'][:,:]
LSM_xc = nc_LSM.variables['x'][:] * 100 * 1000
LSM_yc = nc_LSM.variables['y'][:] * 100 * 1000
LSM_xc = LSM_xc.astype(int)
LSM_yc = LSM_yc.astype(int)
####
min_distance_to_land = 50 * 1000 # m
idx_distance_to_land = distance_to_land > min_distance_to_land
#
min_speed = 100     # m.day-1
max_speed = 100 * 1000   # m.day-1
str_speed = str(min_speed) + '-' + str(max_speed)
###
N_Buoys_eval = {}
#
first_date_eval = datetime(int(date_min_eval[0:4]), int(date_min_eval[4:6]), int(date_min_eval[6:8]))
last_date_eval = datetime(int(date_max_eval[0:4]), int(date_max_eval[4:6]), int(date_max_eval[6:8]))
delta_eval = last_date_eval - first_date_eval
#
for d in range(0, delta_eval.days + 1):
	start_date = (first_date_eval + timedelta(days = d)).strftime('%Y%m%d')
	file_T4 = path_T4_eval + 'TOPAZ4_daily_' + start_date + '.nc'
	if os.path.isfile(file_T4) == True:
		print('Buoys evaluation', start_date)
		for lt in range(0, 10):
			target_date = (first_date_eval + timedelta(days = d + lt)).strftime('%Y%m%d')
			file_buoys = path_buoys + target_date[0:4] + '/' + target_date[4:6] + '/Buoys_' + target_date + '.nc'
			if os.path.isfile(file_buoys) == True:
				nc_buoys = Dataset(file_buoys, 'r')
				lat_buoys = nc_buoys.variables['latitude'][:,:]
				lon_buoys = nc_buoys.variables['longitude'][:,:]
				Buoys_drift_magnitude = nc_buoys.variables['drift_magnitude'][:,:]
				Buoys_drift_direction = nc_buoys.variables['drift_initial_bearing'][:,:]
				#
				idx_speed = np.logical_and(Buoys_drift_magnitude > min_speed, Buoys_drift_magnitude < max_speed)
				idx_direction = Buoys_drift_direction == 0
				idx_region = mask_Canadian_Archipelago == 0
				idx_all = np.logical_and(np.logical_and(idx_direction == False, np.logical_and(idx_distance_to_land == True, idx_speed == True)), idx_region == True)
				#
				Buoys_drift_magnitude[idx_all == False] = np.nan
				Buoys_drift_direction[idx_all == False] = np.nan
				#
				N = np.zeros(np.shape(Buoys_drift_magnitude))
				N[np.isnan(Buoys_drift_magnitude) == False] = 1
				#
				if str(lt) in N_Buoys_eval:
					N_Buoys_eval[str(lt)] = N_Buoys_eval.get(str(lt)) + N
				else:
					N_Buoys_eval[str(lt)] = N
###########################################################
# Buoys training
###########################################################
N_buoy_train = np.zeros(np.shape(distance_to_land))
file_buoy_train = path_buoy_train + 'Buoys_T4grid_daily_20130606-20200528_0-1_days.dat'
df_buoy_train = pd.read_csv(file_buoy_train, delimiter = '\t')
df_buoy_train = df_buoy_train.dropna(how = 'all', axis = 1)  # Remove unnamed columns containing nan
df_buoy_train_x = df_buoy_train['xc']
df_buoy_train_y = df_buoy_train['yc']
#
for i in range(0, len(df_buoy_train_x)):
        pos_x = np.where(LSM_xc == df_buoy_train_x[i])
        pos_y = np.where(LSM_yc == df_buoy_train_y[i])
        N_buoy_train[pos_y, pos_x] = N_buoy_train[pos_y, pos_x] + 1
###########################################################
# SAR observations evaluation
###########################################################
N_SAR_eval = {}
latitude_max = 0
#
first_date = datetime(int(date_min_eval[0:4]), int(date_min_eval[4:6]), int(date_min_eval[6:8]))
last_date = datetime(int(date_max_eval[0:4]), int(date_max_eval[4:6]), int(date_max_eval[6:8]))
delta = last_date - first_date
for d in range(0, delta.days + 1):
	start_date = (first_date + timedelta(days = d)).strftime('%Y%m%d')
	file_T4 = path_T4_eval + 'TOPAZ4_daily_' + start_date + '.nc'
	print('SAR observations - evaluation', start_date)
	#
	if os.path.isfile(file_T4) == True:
		for lt in range(0, 10):
			target_date_start = (first_date + timedelta(days = d + lt)).strftime('%Y%m%d%H%M%S')
			target_date_end = (first_date + timedelta(days = d + lt + 1)).strftime('%Y%m%d%H%M%S')
			file_SAR = path_SAR + target_date_start[0:4] + '/' + target_date_start[4:6] + '/' + 'SAR_drift_' + target_date_start + '-' + target_date_end + '.nc'
			#
			if os.path.isfile(file_SAR) == True:
				nc_T4 = Dataset(file_T4, 'r')
				nc_SAR = Dataset(file_SAR, 'r')
				#
				if (d == 0 and lt == 0):
					latitude = nc_T4.variables['latitude'][:,:]
					longitude = nc_T4.variables['longitude'][:,:]
				#
				T4_drift_direction = nc_T4.variables['drift_initial_bearing'][lt,:,:]
				SAR_drift_direction = nc_SAR.variables['drift_initial_bearing'][:,:]
				#
				T4_direction_error = abs(T4_drift_direction - SAR_drift_direction)
				T4_direction_error[T4_direction_error > 180] = 360 - T4_direction_error[T4_direction_error > 180]
				#
				N = np.zeros(np.shape(T4_direction_error))
				N[np.isnan(T4_direction_error) == False] = 1
				N[mask_Canadian_Archipelago == 1] = 0
				#
				if str(lt) in N_SAR_eval:
					N_SAR_eval[str(lt)] = N_SAR_eval.get(str(lt)) + N
				else:
					N_SAR_eval[str(lt)] = N
###########################################################
# SAR observations training
###########################################################
N_SAR_train = np.zeros(np.shape(distance_to_land))
file_SAR_train = path_SAR_train + 'RF_variables_T4grid_daily_20180104-20200528_0-1_days.dat'
df_SAR_train = pd.read_csv(file_SAR_train, delimiter = '\t')
df_SAR_train = df_SAR_train.dropna(how = 'all', axis = 1)  # Remove unnamed columns containing nan
df_SAR_train_x = df_SAR_train['xc']
df_SAR_train_y = df_SAR_train['yc']
#
for i in range(0, len(df_SAR_train_x)):
	pos_x = np.where(LSM_xc == df_SAR_train_x[i])
	pos_y = np.where(LSM_yc == df_SAR_train_y[i])
	N_SAR_train[pos_y, pos_x] = N_SAR_train[pos_y, pos_x] + 1
###########################################################
# Maps
print('Mapping')
###########################################################
map_proj = cartopy.crs.NorthPolarStereo()
LAND_highres = cpf.NaturalEarthFeature('physical', 'land', '50m', edgecolor = 'face', facecolor = 'dimgrey', linewidth = .1)
map_extent = (-180, 180, 65, 90)
###################
fig = plt.figure()
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
fig, ax = plt.subplots(2, 2, figsize=(17, 20), facecolor = 'w', edgecolor = 'k')
cbar_ax = fig.add_axes([0.2, 0.6, 0.02, 0.2])
plt.subplots_adjust(hspace = 0.02, wspace = 0.02)
#####
ax1 = plt.subplot(2, 2, 1, projection = map_proj)
ax1.set_extent(map_extent, crs = cartopy.crs.PlateCarree())
ax1.add_feature(LAND_highres, zorder = 1)
ax1.set_title('Number of buoy observations per grid cell \n used for training the random forest algorithms', fontsize = sizefont, fontweight = 'bold')
cs = ax1.pcolormesh(longitude, latitude, N_buoy_train, transform = ccrs.PlateCarree(), norm = norm_N_buoys, cmap = colorscale_N, zorder = 0, shading = 'flat')
cbar = plt.colorbar(cs, ax = ax1, orientation = 'horizontal', pad = 0.03, ticks = levels_N_buoys, shrink = 0.8, extend = 'max')
cbar.ax.tick_params(labelsize = sizefont * 0.8)
ax1.text(0.0, -0.08, 'a)', fontsize = sizefont * 1.5, color='k', transform = ax1.transAxes)
#####
ax2 = plt.subplot(2, 2, 2, projection = map_proj)
ax2.set_extent(map_extent, crs = cartopy.crs.PlateCarree())
ax2.add_feature(LAND_highres, zorder = 1)
ax2.set_title('Number of buoy observations per grid cell \n used for evaluating the random forest algorithms', fontsize = sizefont, fontweight = 'bold')
cs = ax2.pcolormesh(longitude, latitude, N_Buoys_eval['0'], transform = ccrs.PlateCarree(), norm = norm_N_buoys, cmap = colorscale_N, zorder = 0, shading = 'flat')
cbar = plt.colorbar(cs, ax = ax2, orientation = 'horizontal', pad = 0.03, ticks = levels_N_buoys, shrink = 0.8, extend = 'max')
cbar.ax.tick_params(labelsize = sizefont * 0.8)
ax2.text(0.0, -0.08, 'b)', fontsize = sizefont * 1.5, color='k', transform = ax2.transAxes)
#####
ax3 = plt.subplot(2, 2, 3, projection = map_proj)
ax3.set_extent(map_extent, crs = cartopy.crs.PlateCarree())
ax3.add_feature(LAND_highres, zorder = 1)
ax3.set_title('Number of SAR observations per grid cell \n used for training the random forest algorithms', fontsize = sizefont, fontweight = 'bold')
cs = ax3.pcolormesh(longitude, latitude, N_SAR_train, transform = ccrs.PlateCarree(), norm = norm_N_SAR_train, cmap = colorscale_N, zorder = 0, shading = 'flat')
cbar = plt.colorbar(cs, ax = ax3, orientation = 'horizontal', pad = 0.03, ticks = levels_N_SAR_train, shrink = 0.8)
cbar.ax.tick_params(labelsize = sizefont * 0.8)
ax3.text(0.0, -0.08, 'c)', fontsize = sizefont * 1.5, color='k', transform = ax3.transAxes)
#####
ax4 = plt.subplot(2, 2, 4, projection = map_proj)
ax4.set_extent(map_extent, crs = cartopy.crs.PlateCarree())
ax4.add_feature(LAND_highres, zorder = 1)
ax4.set_title('Number of SAR observations per grid cell \n used for evaluating the random forest algorithms', fontsize = sizefont, fontweight = 'bold')
cs = ax4.pcolormesh(longitude, latitude, N_SAR_eval['0'], transform = ccrs.PlateCarree(), norm = norm_N_SAR_eval, cmap = colorscale_N, zorder = 0, shading = 'flat')
cbar = plt.colorbar(cs, ax = ax4, orientation = 'horizontal', pad = 0.03, ticks = levels_N_SAR_eval, shrink = 0.8)
cbar.ax.tick_params(labelsize = sizefont * 0.8)
ax4.text(0.0, -0.08, 'd)', fontsize = sizefont * 1.5, color='k', transform = ax4.transAxes)
#####
plt.savefig(path_output + 'Maps_N_Buoys_and_SAR_observations_resampled_2_percents.png', dpi = 300, bbox_inches = 'tight')
plt.close()
##############################################################################################################
EOF

python3 "/lustre/storeB/users/cyrilp/Svalnav/data_processing_files/PROG/Maps_sampling_""$SGE_TASK_ID"".py"
