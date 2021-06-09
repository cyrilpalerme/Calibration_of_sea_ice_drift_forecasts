#$ -S /bin/bash
#$ -l h_rt=02:00:00
#$ -q research-el7.q
#$ -l h_vmem=10G
#$ -t 1-4
#$ -o /lustre/storeB/users/cyrilp/Svalnav/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/cyrilp/Svalnav/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -cwd

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

source /modules/centos7/conda/Feb2021/etc/profile.d/conda.sh 
conda activate production

cat > "/lustre/storeB/users/cyrilp/Svalnav/data_processing_files/PROG/Maps_PA_SAR_distland_20_obs_""$SGE_TASK_ID"".py" << EOF

################################################
import matplotlib
matplotlib.use('Agg')
import os
import glob
import pickle
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
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
path_data = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Verification/Figures/Maps_fraction_of_forecasts_improved/Data/'
path_output = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Verification/Figures/Maps_fraction_of_forecasts_improved/'
path_T4 = '/lustre/storeB/project/copernicus/svalnav/Data_operational_ice_drift_forecasts/Pan_Arctic/TOPAZ4/'
path_SAR = '/lustre/storeB/project/copernicus/svalnav/Data_operational_ice_drift_forecasts/Pan_Arctic/SAR_v2/'
path_CF_buoys = '/lustre/storeB/project/copernicus/svalnav/April_2021/Calibrated_forecasts/RF_trained_with_buoys_201306_202005_without_dayofyear/'
path_CF_SAR = '/lustre/storeB/project/copernicus/svalnav/April_2021/Calibrated_forecasts/RF_trained_with_SAR_201801_202005_2_percents_without_dayofyear/'
path_region_mask = '/lustre/storeB/users/cyrilp/Data/NSIDC_regional_mask/'
#####
date_min = '20200601'
date_max = '20210531'
#
threshold_N = 20
#
sizefont = 13
sizefont_legend = 12
alpha = 0.1
#
date_test = str(date_min) + '-' + str(date_max)
###
levels_N = [0, 1, 5, 10, 20, 30, 50, 70, 100, 150, 200]
norm_N = BoundaryNorm(levels_N, 256)
colorscale_N = 'gist_stern_r'
#
colorscale_div = 'RdYlBu'
levels_frac_improved = np.linspace(25, 75, num = 11)
norm_frac_improved = BoundaryNorm(levels_frac_improved, 256)
#####
if $SGE_TASK_ID == 1:
	path_CF = path_CF_buoys
	file_figure = 'Maps_direction_fraction_improved_20_SAR_observations_RF_trained_buoys_' + date_test + '.png'
	figure_title = 'Calibrated forecasts (models trained with buoy observations)' + '\n' + 'Fraction of forecasts improved for the direction of sea-ice drift (%)'
	T4_var = 'drift_initial_bearing'
	SAR_var = 'drift_initial_bearing'
	CF_var = 'drift_direction'
#
elif $SGE_TASK_ID == 2:
	path_CF = path_CF_buoys
	file_figure = 'Maps_speed_fraction_improved_20_SAR_observations_RF_trained_buoys_' + date_test + '.png'
	figure_title = 'Calibrated forecasts (models trained with buoy observations)' + '\n' + 'Fraction of forecasts improved for the speed of sea-ice drift (%)'
	T4_var = 'drift_magnitude'
	SAR_var = 'drift_magnitude'
	CF_var = 'drift_magnitude'
#
elif $SGE_TASK_ID == 3:
	path_CF = path_CF_SAR
	file_figure = 'Maps_direction_fraction_improved_20_SAR_observations_RF_trained_SAR_' + date_test + '.png'
	figure_title = 'Calibrated forecasts (models trained with SAR observations)' + '\n' + 'Fraction of forecasts improved for the direction of sea-ice drift (%)'
	T4_var = 'drift_initial_bearing'
	SAR_var = 'drift_initial_bearing'
	CF_var = 'drift_direction'
#
elif $SGE_TASK_ID == 4:
	path_CF = path_CF_SAR
	file_figure = 'Maps_speed_fraction_improved_20_SAR_observations_RF_trained_SAR_' + date_test + '.png'
	figure_title = 'Calibrated forecasts (models trained with SAR observations)' + '\n' + 'Fraction of forecasts improved for the speed of sea-ice drift (%)'
	T4_var = 'drift_magnitude'
	SAR_var = 'drift_magnitude'
	CF_var = 'drift_magnitude'
###########################################################
# Region mask
###########################################################
file_region_mask = path_region_mask + 'sio_2016_mask_TOPAZ4_grid.nc'
nc_region_mask = Dataset(file_region_mask, 'r')
mask = nc_region_mask.variables['mask'][:,:]
mask_Canadian_Archipelago = np.zeros(np.shape(mask))
mask_Canadian_Archipelago[mask == 14] = 1
###########################################################
# Datasets
###########################################################
N_errors = {}
N_forecasts_improved = {}
#
first_date = datetime(int(date_min[0:4]), int(date_min[4:6]), int(date_min[6:8]))
last_date = datetime(int(date_max[0:4]), int(date_max[4:6]), int(date_max[6:8]))
delta = last_date - first_date
for d in range(delta.days + 1):
	start_date = (first_date + timedelta(days = d)).strftime('%Y%m%d')
	file_CF = path_CF + 'calibrated_ice_drift_forecasts_' + start_date + '.nc'
	file_T4 = path_T4 + 'TOPAZ4_daily_' + start_date + '.nc'
	print(start_date)
	#
	for lt in range(0, 10):
		target_date_start = (first_date + timedelta(days = d + lt)).strftime('%Y%m%d%H%M%S') 
		target_date_end = (first_date + timedelta(days = d + lt + 1)).strftime('%Y%m%d%H%M%S') 
		file_SAR = path_SAR + target_date_start[0:4] + '/' + target_date_start[4:6] + '/' + 'SAR_drift_' + target_date_start + '-' + target_date_end + '.nc'
		#
		if os.path.isfile(file_SAR) == True:
			nc_CF = Dataset(file_CF, 'r')
			nc_T4 = Dataset(file_T4, 'r')
			nc_SAR = Dataset(file_SAR, 'r')
			#
			if (d == 0 and lt == 0):
				latitude = nc_T4.variables['latitude'][:,:]
				longitude = nc_T4.variables['longitude'][:,:]
			#
			CF_drift = nc_CF.variables[CF_var][lt,:,:]
			T4_drift = nc_T4.variables[T4_var][lt,:,:]
			SAR_drift = nc_SAR.variables[SAR_var][:,:]
			#
			CF_drift[CF_drift.mask == True] = np.nan
			CF_drift[mask_Canadian_Archipelago == 1] = np.nan
			T4_drift[mask_Canadian_Archipelago == 1] = np.nan
			SAR_drift[mask_Canadian_Archipelago == 1] = np.nan
			####
			CF_error_file = abs(CF_drift - SAR_drift)
			T4_error_file = abs(T4_drift - SAR_drift)
			#
			if CF_var == 'drift_direction':
				CF_error_file[CF_error_file > 180] = 360 - CF_error_file[CF_error_file > 180]
				T4_error_file[T4_error_file > 180] = 360 - T4_error_file[T4_error_file > 180]
			###
			idx_nan = np.logical_or(np.isnan(CF_error_file) == True, np.isnan(T4_error_file) == True)
			CF_error_file[idx_nan == True] = np.nan
			T4_error_file[idx_nan == True] = np.nan
			###
			# Binary values
			#
			Forecasts_improved = np.zeros(np.shape(CF_error_file))
			Forecasts_improved[CF_error_file < T4_error_file] = 1
			#
			N_data = np.zeros(np.shape(CF_error_file))
			N_data[np.isnan(CF_error_file) == False] = 1
			###
			if str(lt) in N_errors:
				N_errors[str(lt)] = N_errors.get(str(lt)) + N_data
				N_forecasts_improved[str(lt)] = N_forecasts_improved.get(str(lt)) + Forecasts_improved
			else:
				N_errors[str(lt)] =  np.copy(N_data)
				N_forecasts_improved[str(lt)] = np.copy(Forecasts_improved)
###########################################################
# Results
###########################################################
N_CF = np.full((10, np.shape(CF_drift)[0], np.shape(CF_drift)[1]), np.nan)
Fraction_improved = np.full((10, np.shape(CF_drift)[0], np.shape(CF_drift)[1]), np.nan)
#
for lt in range(0, 10):
	N_CF[lt,:,:] = N_errors[str(lt)]
	Fraction_improved[lt,:,:] = N_forecasts_improved[str(lt)] / N_errors[str(lt)]
#####
idx_N = N_CF >= threshold_N
Fraction_improved[idx_N == False] = np.nan
#####
Fraction_spatial_frac_improved = np.full(10, np.nan)
#
for lt in range(0, 10):
	Fraction_spatial_frac_improved[lt] = np.sum(Fraction_improved[lt,:,:] >= 0.5) / np.sum(np.isnan(Fraction_improved[lt,:,:]) == False)
###########################################################
# Save data
###########################################################
Data = {}
Data['latitude'] = latitude
Data['longitude'] = longitude
Data['Fraction_improved'] = Fraction_improved
Data['Fraction_spatial_frac_improved'] = Fraction_spatial_frac_improved
#
file_data = file_figure.replace('.png', '.pkl')
with open(path_data + file_data, 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)
###########################################################
# Maps
print('Mapping')
###########################################################
#map_proj = cartopy.crs.NorthPolarStereo()
map_proj = cartopy.crs.Stereographic(central_latitude=0.0, central_longitude=0.0)
LAND_highres = cpf.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor = 'dimgrey', linewidth=.1)
map_extent = (-180, 180, 65, 90)
gs = gridspec.GridSpec(3, 4)
###########################################################
# N comparisons 
###########################################################
plt.figure()
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
fig, axs = plt.subplots(3, 4, figsize=(13, 10), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.2, wspace = 0.15)
fig.suptitle(figure_title, fontsize = sizefont * 1.5)
axs = axs.ravel()
#
Fraction_improved[Fraction_improved == 0] = np.nan
#
for lt in range(0, 12):
	if lt < 10:
		axs[lt] = plt.subplot(3, 4, lt + 1, projection = map_proj)
		axs[lt].set_extent(map_extent, crs = cartopy.crs.PlateCarree())
		axs[lt].add_feature(LAND_highres, zorder = 1)
		cs = axs[lt].pcolormesh(longitude, latitude, N_CF[lt,:,:], transform = ccrs.PlateCarree(), norm = norm_N, cmap = colorscale_N, zorder = 0, shading = 'flat')
		#
		if lt == 0:
			axs[lt].set_title('lead time: ' + str(lt + 1) + ' day', fontsize = sizefont)
		else:
			axs[lt].set_title('lead time: ' + str(lt + 1) + ' days', fontsize = sizefont)
	elif lt == 10:
		axs[lt] = plt.subplot(gs[lt // 4, lt % 4 : lt % 4 + 2])
		axs[lt].plot(np.arange(10) + 1, 100 * Fraction_spatial_frac_improved, 'bo--')
		axs[lt].set_xticks(np.arange(1, 11, step = 1))
		axs[lt].grid()
		axs[lt].set_title('N comparisons', fontsize = sizefont)
		axs[lt].set_xlabel('Forecast lead time (days)', fontsize = sizefont)
	else:
		axs[lt].patch.set_visible(False)
		axs[lt].axis('off')
###
cbar_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
cbar = fig.colorbar(cs, cax = cbar_ax, ticks = levels_N[1:-1], extend = 'both')
fig.subplots_adjust(left = None, bottom = None, right = 0.9, top = None, wspace = None, hspace = None)
#plt.savefig(path_output + file_figure.replace('20_SAR_observations', 'N_comparisons'), dpi = 200, bbox_inches='tight')
plt.close()
###########################################################
# Fraction of forecasts improved 
###########################################################
plt.figure()
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
fig, axs = plt.subplots(3, 4, figsize=(13, 10), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.2, wspace = 0.15)
fig.suptitle(figure_title, fontsize = sizefont * 1.5)
axs = axs.ravel()
#
Fraction_improved[Fraction_improved == 0] = np.nan
#
for lt in range(0, 12):
	if lt < 10:
		axs[lt] = plt.subplot(3, 4, lt + 1, projection = map_proj)
		axs[lt].set_extent(map_extent, crs = cartopy.crs.PlateCarree())
		axs[lt].add_feature(LAND_highres, zorder = 1)
		cs = axs[lt].pcolormesh(longitude, latitude, 100 * Fraction_improved[lt,:,:], transform = ccrs.PlateCarree(), norm = norm_frac_improved, cmap = colorscale_div, zorder = 0, shading = 'flat')
		#
		if lt == 0:
			axs[lt].set_title('lead time: ' + str(lt + 1) + ' day', fontsize = sizefont)
		else:
			axs[lt].set_title('lead time: ' + str(lt + 1) + ' days', fontsize = sizefont)
	elif lt == 10:
		axs[lt] = plt.subplot(gs[lt // 4, lt % 4 : lt % 4 + 2])
		axs[lt].plot(np.arange(10) + 1, 100 * Fraction_spatial_frac_improved, 'bo--')
		axs[lt].set_xticks(np.arange(1, 11, step = 1))
		axs[lt].grid()
		axs[lt].set_title('Fraction of the surface where the calibrated forecasts \n outperform the TOPAZ4 forecasts (%)', fontsize = sizefont)
		axs[lt].set_xlabel('Forecast lead time (days)', fontsize = sizefont)
	else:
		axs[lt].patch.set_visible(False)
		axs[lt].axis('off')
###
cbar_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
cbar = fig.colorbar(cs, cax = cbar_ax, ticks = levels_frac_improved[1:-1], extend = 'both')
fig.subplots_adjust(left = None, bottom = None, right = 0.9, top = None, wspace = None, hspace = None)
plt.savefig(path_output + file_figure, dpi = 200, bbox_inches='tight')
plt.close()
##############################################################################################################
EOF

python3 "/lustre/storeB/users/cyrilp/Svalnav/data_processing_files/PROG/Maps_PA_SAR_distland_20_obs_""$SGE_TASK_ID"".py"
