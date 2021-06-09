import matplotlib
matplotlib.use('Agg')
import os
import sys
import glob
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
from scipy.stats.stats import pearsonr
from math import sqrt
function_path='/lustre/storeB/users/cyrilp/Python_functions/'
sys.path.insert(0, function_path)
from circular_correlation_coefficient import *
###########################################################
# Constants
###########################################################
path_output = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Verification/Figures/SAR_vs_buoys/'
path_buoys = '/lustre/storeB/project/copernicus/svalnav/Data/IABP/Data_T4grid_SIC10/'
path_SAR = '/lustre/storeB/project/copernicus/svalnav/Data_operational_ice_drift_forecasts/Pan_Arctic/SAR_v2/'
path_LSM = '/lustre/storeB/project/copernicus/svalnav/Data/TOPAZ4/'
path_region_mask = '/lustre/storeB/users/cyrilp/Data/NSIDC_regional_mask/'
###########################################################
# Region mask
###########################################################
file_region_mask = path_region_mask + 'sio_2016_mask_TOPAZ4_grid.nc'
nc_region_mask = Dataset(file_region_mask, 'r')
mask = nc_region_mask.variables['mask'][:,:]
mask_Canadian_Archipelago = np.zeros(np.shape(mask))
mask_Canadian_Archipelago[mask == 14] = 1
#
levels_speed = np.arange(200)
levels_direction = np.arange(91)
norm_N_speed = BoundaryNorm(levels_speed, 256)
norm_N_direction = BoundaryNorm(levels_direction, 256)
###########################################################
# Datasets
###########################################################
file_LSM = path_LSM + 'TOPAZ4_land_sea_mask.nc'
nc_LSM = Dataset(file_LSM, 'r')
distance_to_land = np.ndarray.flatten(nc_LSM.variables['distance_to_land'][:,:])
##########
date_min_dataset = '20180101'
date_max_dataset = '20201231'
#
st_date = datetime.strptime(str(date_min_dataset), '%Y%m%d')
start_dates = []
end_dates = []
while st_date <= datetime.strptime(str(date_max_dataset), '%Y%m%d'):
	start_dates.append(st_date.strftime('%Y%m%d'))
	end_dates.append((st_date + timedelta(days = 1)).strftime('%Y%m%d'))
	st_date = st_date + timedelta(days = 1)
#
SAR_speed = []
SAR_direction = []
buoys_speed = []
buoys_direction = []
distance_to_land_all = []
#
for i in range(0, len(start_dates)):
	try:
		print(start_dates[i])
		year_month = start_dates[i][0:4] + '/' + start_dates[i][4:6] + '/'
		file_buoys = path_buoys + year_month +  'Buoys_' + start_dates[i] + '.nc'
		file_SAR = path_SAR + year_month + 'SAR_drift_' +  start_dates[i] + '000000-' + end_dates[i] + '000000.nc'
		#
		nc_buoys = Dataset(file_buoys, 'r')
		nc_SAR = Dataset(file_SAR, 'r')
		#
		drift_magnitude_buoys = nc_buoys.variables['drift_magnitude'][:,:]
		drift_initial_bearing_buoys = nc_buoys.variables['drift_initial_bearing'][:,:]
		drift_magnitude_SAR = nc_SAR.variables['drift_magnitude'][:,:]
		drift_initial_bearing_SAR = nc_SAR.variables['drift_initial_bearing'][:,:]
		#####
		drift_magnitude_buoys[drift_initial_bearing_buoys == 0] = np.nan
		drift_initial_bearing_buoys[drift_initial_bearing_buoys == 0] = np.nan
		#####
		drift_magnitude_buoys[mask_Canadian_Archipelago == 1] = np.nan
		drift_initial_bearing_buoys[mask_Canadian_Archipelago == 1] = np.nan
		drift_magnitude_SAR[mask_Canadian_Archipelago == 1] = np.nan
		drift_initial_bearing_SAR[mask_Canadian_Archipelago == 1] = np.nan
		#####
		drift_magnitude_buoys = np.ndarray.flatten(drift_magnitude_buoys)
		drift_initial_bearing_buoys = np.ndarray.flatten(drift_initial_bearing_buoys)
		drift_magnitude_SAR = np.ndarray.flatten(drift_magnitude_SAR)
		drift_initial_bearing_SAR = np.ndarray.flatten(drift_initial_bearing_SAR)
		#####
		diff_magnitude = drift_magnitude_SAR - drift_magnitude_buoys
		idx_nan = np.isnan(diff_magnitude)
		#
		SAR_speed = np.hstack((SAR_speed, drift_magnitude_SAR[idx_nan == False]))
		SAR_direction = np.hstack((SAR_direction, drift_initial_bearing_SAR[idx_nan == False]))
		#
		buoys_speed = np.hstack((buoys_speed, drift_magnitude_buoys[idx_nan == False]))
		buoys_direction = np.hstack((buoys_direction, drift_initial_bearing_buoys[idx_nan == False]))
		#
		distance_to_land_all = np.hstack((distance_to_land_all,  distance_to_land[idx_nan == False]))
	except:
		pass
###########################################################
# Data selection
###########################################################
idx_direction = buoys_direction == 0
#
min_distance_to_land = 50 * 1000 # m
idx_distance_to_land = distance_to_land_all > min_distance_to_land
#
min_speed = 100     # m.day-1
max_speed = 100 * 1000   # m.day-1
str_speed = str(min_speed) + '-' + str(max_speed)
idx_speed = np.logical_and(buoys_speed > min_speed, buoys_speed < max_speed)
#
idx_all = np.logical_and(idx_direction == False, np.logical_and(idx_distance_to_land == True, idx_speed == True))
#
SAR_speed = SAR_speed[idx_all == True] 
SAR_direction = SAR_direction[idx_all == True]
buoys_speed = buoys_speed[idx_all == True]
buoys_direction = buoys_direction[idx_all == True] 
###########################################################
# Statistics
###########################################################
diff_direction = SAR_direction - buoys_direction
diff_direction[diff_direction > 180] = diff_direction[diff_direction > 180] - 360
diff_direction[diff_direction < -180] = 360 + diff_direction[diff_direction < -180]
#####
N = str(len(SAR_speed))
N_int = int(N)
print('number of points: ', N)
#
ME_speed = '{:.2f}'.format(round(0.001 * np.mean(SAR_speed - buoys_speed), 2))
ME_direction = str(round(np.mean(diff_direction), 1))
print('mean error direction = ', np.mean(diff_direction))
#
Median_AE_speed = '{:.2f}'.format(round(0.001 * np.median(np.abs(SAR_speed - buoys_speed)), 2))
Median_AE_direction = str(round(np.median(np.abs(diff_direction)), 1))
#
MAE_speed = '{:.2f}'.format(round(0.001 * np.mean(np.abs(SAR_speed - buoys_speed)), 2))
MAE_direction = str(round(np.mean(np.abs(diff_direction)), 1))
#
RMSE_speed = '{:.2f}'.format(round(sqrt(np.sum((0.001 * SAR_speed - 0.001 * buoys_speed) ** 2) / N_int), 2))
RMSE_direction = str(round(sqrt(np.sum(diff_direction ** 2) / N_int), 1))
#
R_speed = '{:.2f}'.format(round(pearsonr(buoys_speed, SAR_speed)[0], 2))
R_direction = str(round(circular_correlation_coefficient(buoys_direction, SAR_direction), 2))
#
fit_speed = np.polyfit(buoys_speed, SAR_speed, 1)
fit_direction = np.polyfit(buoys_direction, SAR_direction, 1)
#
fit_fn_speed = np.poly1d(fit_speed)
fit_fn_direction = np.poly1d(fit_direction)
#
slope_speed = str(round(fit_speed[0], 2))
slope_direction = str(round(fit_direction[0], 2))
###########################################################
# Scatterplots
###########################################################
sizefont = 20
#########
plt.figure()
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
fig, ax1 = plt.subplots(1, 2, figsize=(20, 10), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.2, wspace = 0.2)
#
ax = plt.subplot(121)
l1 = ax.hist2d(buoys_direction, SAR_direction, (72, 72), cmap = plt.cm.gnuplot2_r, norm = norm_N_direction)
l2 = ax.plot([0, 360], [0, 360], 'k')
ax.set_title('Direction', fontsize = sizefont, fontweight = 'bold')
ax.set_xlabel('Buoys (degrees)', fontsize = sizefont)
ax.set_ylabel('SAR (degrees)', fontsize = sizefont)
ax.set_xlim([0, 360])
ax.set_ylim([0, 360])
ax.text(0.01,0.96, 'N = ' + N, fontsize = sizefont * 0.9, ha='left', transform = ax.transAxes)
ax.text(0.01,0.92, 'Rc = ' + R_direction, fontsize = sizefont * 0.9, ha='left', transform = ax.transAxes)
ax.text(0.01,0.88, 'RMSE = ' + RMSE_direction, fontsize = sizefont * 0.9, ha='left', transform = ax.transAxes)
ax.text(0.01,0.84, 'Mean error = ' + ME_direction, fontsize = sizefont * 0.9, ha='left', transform = ax.transAxes)
ax.text(0.01,0.80, 'Mean absolute error = ' + MAE_direction, fontsize = sizefont * 0.9, ha='left', transform = ax.transAxes)
ax.text(0.01,0.76, 'Median absolute error = ' + Median_AE_direction, fontsize = sizefont * 0.9, ha='left', transform = ax.transAxes)
ax.text(-0.08,-0.18, 'a)', fontsize = sizefont * 1.5, ha = 'left', transform = ax.transAxes)
cbar = fig.colorbar(l1[3], ax = ax, orientation = 'horizontal', pad = 0.1)
cbar.set_label('Number of observations', size = sizefont)
#
ax = plt.subplot(122)
l1 = ax.hist2d(buoys_speed * 0.001, SAR_speed * 0.001, bins = [60, 60], range = [[0, 30],[0, 30]], cmap = plt.cm.gnuplot2_r, norm = norm_N_speed)
l2 = ax.plot([0, 30], [0, 30], 'k')
ax.set_title('Speed', fontsize = sizefont, fontweight = 'bold')
ax.set_xlabel('Buoys (km / day) ', fontsize = sizefont)
ax.set_ylabel('SAR (km / day)', fontsize = sizefont)
ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.text(0.01,0.96, 'N = ' + N, fontsize = sizefont * 0.9, ha='left', transform = ax.transAxes)
ax.text(0.01,0.92, 'Rp = ' + R_speed, fontsize = sizefont * 0.9, ha='left', transform = ax.transAxes)
ax.text(0.01,0.88, 'RMSE = ' + RMSE_speed, fontsize = sizefont * 0.9, ha='left', transform = ax.transAxes)
ax.text(0.01,0.84, 'Mean error = ' + ME_speed, fontsize = sizefont * 0.9, ha='left', transform = ax.transAxes)
ax.text(0.01,0.80, 'Mean absolute error = ' + MAE_speed, fontsize = sizefont * 0.9, ha='left', transform = ax.transAxes)
ax.text(0.01,0.76, 'Median absolute error = ' + Median_AE_speed, fontsize = sizefont * 0.9, ha='left', transform = ax.transAxes)
ax.text(-0.08,-0.18, 'b)', fontsize = sizefont * 1.5, ha = 'left', transform = ax.transAxes)
cbar = fig.colorbar(l1[3], ax = ax, orientation = 'horizontal', pad = 0.1)
cbar.set_label('Number of observations', size = sizefont)
#
plt.savefig(path_output + 'Scatterplots_' + date_min_dataset + '_' + date_max_dataset + '.png', bbox_inches = 'tight')
plt.clf()
