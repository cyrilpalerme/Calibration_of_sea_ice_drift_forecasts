import matplotlib
matplotlib.use('Agg')
import os
import glob
import cmath
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
from scipy.stats.stats import pearsonr
from scipy.stats import wilcoxon
###########################################################
# Constants
###########################################################
path_output = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Verification/Figures/Buoys_as_reference/'
path_CF_buoys = '/lustre/storeB/project/copernicus/svalnav/April_2021/Calibrated_forecasts/RF_trained_with_buoys_201306_202005_without_dayofyear/'
path_CF_SAR = '/lustre/storeB/project/copernicus/svalnav/April_2021/Calibrated_forecasts/RF_trained_with_SAR_201801_202005_2_percents_without_dayofyear/'
path_T4 = '/lustre/storeB/project/copernicus/svalnav/Data_operational_ice_drift_forecasts/Pan_Arctic/TOPAZ4/'
path_Buoys = '/lustre/storeB/project/copernicus/svalnav/Data/IABP/Data_T4grid_SIC10/'
path_LSM = '/lustre/storeB/project/copernicus/svalnav/Data/TOPAZ4/'
path_region_mask = '/lustre/storeB/users/cyrilp/Data/NSIDC_regional_mask/'
#
date_min = '20200601'
date_max = '20210531'
#
sizefont = 17
sizefont_legend = 11
alpha = 0.1
#
sizemarker = 7
color_T4 = 'dimgrey'
color_calib_buoys = 'royalblue'
color_calib_SAR = 'crimson'
#
alpha = 0.5
#
date_test = str(date_min) + '-' + str(date_max)
#
threshold_pvalue = 0.05
###########################################################
# Circular correlation coefficient
###########################################################
def circular_correlation_coefficient(variable1, variable2):
	#
	var1_rad = variable1 * np.pi / 180
	var2_rad = variable2 * np.pi / 180
	#
	var1_com = np.full(np.shape(variable1), np.nan, dtype = complex)
	var2_com = np.full(np.shape(variable2), np.nan, dtype = complex)
	for i in range(0, len(variable1)):
		var1_com[i] = cmath.rect(1, var1_rad[i])
		var2_com[i] = cmath.rect(1, var2_rad[i])
	#
	var1_com_mean = np.mean(var1_com)
	var2_com_mean = np.mean(var2_com)
	#
	var1_rad_mean = cmath.phase(var1_com_mean)
	if var1_rad_mean < 0:
		var1_rad_mean = var1_rad_mean + 2 * np.pi
	#
	var2_rad_mean = cmath.phase(var2_com_mean)
	if var2_rad_mean < 0:
		var2_rad_mean = var2_rad_mean + 2 * np.pi
	#
	diff_var1 = var1_rad - var1_rad_mean
	diff_var2 = var2_rad - var2_rad_mean
	#
	Rcc = np.sum(np.sin(diff_var1) * np.sin(diff_var2)) / np.sqrt(np.sum(np.sin(diff_var1) ** 2) * np.sum(np.sin(diff_var2) ** 2))
	#
	return Rcc
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
file_LSM = path_LSM + 'TOPAZ4_land_sea_mask.nc'
nc_LSM = Dataset(file_LSM, 'r')
distance_to_land = nc_LSM.variables['distance_to_land'][:,:]
#
min_distance_to_land = 50 * 1000 # m
idx_distance_to_land = distance_to_land > min_distance_to_land
#
min_speed = 100     # m.day-1
max_speed = 100 * 1000   # m.day-1
str_speed = str(min_speed) + '-' + str(max_speed)
#########
Forecast_start_date = {}
#
Buoys_IB = {}
T4_IB = {}
CF_buoys_IB = {}
CF_SAR_IB = {}
#
T4_MA = {}
Buoys_MA = {}
CF_buoys_MA = {}
CF_SAR_MA = {}
#####
first_date = datetime(int(date_min[0:4]), int(date_min[4:6]), int(date_min[6:8]))
last_date = datetime(int(date_max[0:4]), int(date_max[4:6]), int(date_max[6:8]))
delta = last_date - first_date
for d in range(delta.days + 1):
	start_date = (first_date + timedelta(days = d)).strftime('%Y%m%d')
	file_CF_buoys = path_CF_buoys + 'calibrated_ice_drift_forecasts_' + start_date + '.nc'
	file_CF_SAR = path_CF_SAR + 'calibrated_ice_drift_forecasts_' + start_date + '.nc'
	file_T4 = path_T4 + 'TOPAZ4_daily_' + start_date + '.nc'
	print(start_date)
	#
	for lt in range(0, 10):
		target_date = (first_date + timedelta(days = d + lt)).strftime('%Y%m%d') 
		target_date_start = (first_date + timedelta(days = d + lt)).strftime('%Y%m%d%H%M%S')
		target_date_end = (first_date + timedelta(days = d + lt + 1)).strftime('%Y%m%d%H%M%S')
		#
		file_Buoys = path_Buoys + target_date[0:4] + '/' + target_date[4:6] + '/' + 'Buoys_' + target_date + '.nc'
		#
		if os.path.isfile(file_Buoys) == True:
			nc_CF_buoys = Dataset(file_CF_buoys, 'r')
			nc_CF_SAR = Dataset(file_CF_SAR, 'r')
			nc_T4 = Dataset(file_T4, 'r')
			nc_Buoys = Dataset(file_Buoys, 'r')
			#
			CF_buoys_drift_magnitude = nc_CF_buoys.variables['drift_magnitude'][lt,:,:]
			CF_buoys_drift_direction = nc_CF_buoys.variables['drift_direction'][lt,:,:]
			#
			CF_SAR_drift_magnitude = nc_CF_SAR.variables['drift_magnitude'][lt,:,:]
			CF_SAR_drift_direction = nc_CF_SAR.variables['drift_direction'][lt,:,:]
			###
			if lt == 0:
				T4_latitude = nc_T4.variables['latitude'][:,:]	
				T4_longitude = nc_T4.variables['longitude'][:,:]	
			#
			T4_drift_magnitude = nc_T4.variables['drift_magnitude'][lt,:,:]
			T4_drift_direction = nc_T4.variables['drift_initial_bearing'][lt,:,:]
			###
			Buoys_drift_magnitude = nc_Buoys.variables['drift_magnitude'][:,:]
			Buoys_drift_direction = nc_Buoys.variables['drift_initial_bearing'][:,:]
			###
			CF_buoys_drift_magnitude[CF_buoys_drift_magnitude.mask == True] = np.nan
			CF_buoys_drift_direction[CF_buoys_drift_direction.mask == True] = np.nan
			CF_SAR_drift_magnitude[CF_SAR_drift_magnitude.mask == True] = np.nan
			CF_SAR_drift_direction[CF_SAR_drift_direction.mask == True] = np.nan
			###
			idx_speed = np.logical_and(Buoys_drift_magnitude > min_speed, Buoys_drift_magnitude < max_speed)
			idx_direction = np.logical_or(Buoys_drift_direction == 0, np.isnan(Buoys_drift_direction) == True)
			idx_all = np.logical_and(mask_Canadian_Archipelago == 0, np.logical_and(idx_distance_to_land == True, np.logical_and(idx_direction == False, idx_speed == True)))
			Buoys_drift_magnitude[idx_all == False] = np.nan
			Buoys_drift_direction[idx_all == False] = np.nan
			###
			idx_nan_buoys = np.logical_or(np.isnan(Buoys_drift_magnitude) == True, np.isnan(Buoys_drift_direction) == True)
			idx_nan_T4 = np.logical_or(np.isnan(T4_drift_magnitude) == True, np.isnan(T4_drift_direction) == True)
			idx_nan_CF_buoys = np.logical_or(np.isnan(CF_buoys_drift_magnitude) == True, np.isnan(CF_buoys_drift_direction) == True)
			idx_nan_CF_SAR = np.logical_or(np.isnan(CF_SAR_drift_magnitude) == True, np.isnan(CF_SAR_drift_direction) == True)
			idx_nan_all = np.logical_or(np.logical_or(idx_nan_buoys == True, idx_nan_T4 == True), np.logical_or(idx_nan_CF_buoys == True, idx_nan_CF_SAR == True))
			#
			Buoys_drift_magnitude = np.ndarray.flatten(Buoys_drift_magnitude[idx_nan_all == False])
			Buoys_drift_direction = np.ndarray.flatten(Buoys_drift_direction[idx_nan_all == False])
			T4_drift_magnitude = np.ndarray.flatten(T4_drift_magnitude[idx_nan_all == False])
			T4_drift_direction = np.ndarray.flatten(T4_drift_direction[idx_nan_all == False])
			CF_buoys_drift_magnitude = np.ndarray.flatten(CF_buoys_drift_magnitude[idx_nan_all == False])
			CF_buoys_drift_direction = np.ndarray.flatten(CF_buoys_drift_direction[idx_nan_all == False])
			CF_SAR_drift_magnitude = np.ndarray.flatten(CF_SAR_drift_magnitude[idx_nan_all == False])
			CF_SAR_drift_direction = np.ndarray.flatten(CF_SAR_drift_direction[idx_nan_all == False])
			###
			#
			print(np.sum(Buoys_drift_magnitude > 0))
			###
			if str(lt) in Buoys_MA:
				Forecast_start_date[str(lt)] = np.hstack((Forecast_start_date.get(str(lt)), np.repeat(start_date, len(Buoys_drift_magnitude))))
				#
				Buoys_MA[str(lt)] = np.hstack((Buoys_MA.get(str(lt)), Buoys_drift_magnitude))
				T4_MA[str(lt)] = np.hstack((T4_MA.get(str(lt)), T4_drift_magnitude))
				CF_buoys_MA[str(lt)] = np.hstack((CF_buoys_MA.get(str(lt)), CF_buoys_drift_magnitude))
				CF_SAR_MA[str(lt)] = np.hstack((CF_SAR_MA.get(str(lt)), CF_SAR_drift_magnitude))
				#
				Buoys_IB[str(lt)] = np.hstack((Buoys_IB.get(str(lt)), Buoys_drift_direction))
				T4_IB[str(lt)] = np.hstack((T4_IB.get(str(lt)), T4_drift_direction))
				CF_buoys_IB[str(lt)] = np.hstack((CF_buoys_IB.get(str(lt)), CF_buoys_drift_direction))
				CF_SAR_IB[str(lt)] = np.hstack((CF_SAR_IB.get(str(lt)), CF_SAR_drift_direction))
			else:
				Forecast_start_date[str(lt)] = np.repeat(start_date, len(Buoys_drift_magnitude))
				#
				Buoys_MA[str(lt)] = np.copy(Buoys_drift_magnitude)
				T4_MA[str(lt)] = np.copy(T4_drift_magnitude)
				CF_buoys_MA[str(lt)] = np.copy(CF_buoys_drift_magnitude)
				CF_SAR_MA[str(lt)] = np.copy(CF_SAR_drift_magnitude)
				#
				Buoys_IB[str(lt)] = np.copy(Buoys_drift_direction)
				T4_IB[str(lt)] = np.copy(T4_drift_direction)
				CF_buoys_IB[str(lt)] = np.copy(CF_buoys_drift_direction)
				CF_SAR_IB[str(lt)] = np.copy(CF_SAR_drift_direction)
###########################################################
# Statistics during the evaluation period (EP)
###########################################################
N_errors = np.full(10, np.nan)
#
CF_buoys_MA_MAE = np.full(10, np.nan)
CF_buoys_IB_MAE = np.full(10, np.nan)
CF_SAR_MA_MAE = np.full(10, np.nan)
CF_SAR_IB_MAE = np.full(10, np.nan)
T4_MA_MAE = np.full(10, np.nan)
T4_IB_MAE = np.full(10, np.nan)
#
MA_fraction_improved_buoys = np.full(10, np.nan)
IB_fraction_improved_buoys = np.full(10, np.nan)
MA_fraction_improved_SAR = np.full(10, np.nan)
IB_fraction_improved_SAR = np.full(10, np.nan)
#
CF_buoys_MA_CC = np.full(10, np.nan)
CF_buoys_IB_CC = np.full(10, np.nan)
CF_SAR_MA_CC = np.full(10, np.nan)
CF_SAR_IB_CC = np.full(10, np.nan)
T4_MA_CC = np.full(10, np.nan)
T4_IB_CC = np.full(10, np.nan)
#
wilcoxon_CF_buoys_vs_T4_MA = np.full(10, np.nan)
wilcoxon_CF_buoys_vs_T4_IB = np.full(10, np.nan)
wilcoxon_CF_SAR_vs_T4_MA = np.full(10, np.nan)
wilcoxon_CF_SAR_vs_T4_IB = np.full(10, np.nan)
wilcoxon_CF_buoys_vs_CF_SAR_MA = np.full(10, np.nan)
wilcoxon_CF_buoys_vs_CF_SAR_IB = np.full(10, np.nan)
#
Pvalue_CF_buoys_vs_T4_MA = np.full(10, np.nan)
Pvalue_CF_buoys_vs_T4_IB = np.full(10, np.nan)
Pvalue_CF_SAR_vs_T4_MA = np.full(10, np.nan)
Pvalue_CF_SAR_vs_T4_IB = np.full(10, np.nan)
Pvalue_CF_buoys_vs_CF_SAR_MA = np.full(10, np.nan)
Pvalue_CF_buoys_vs_CF_SAR_IB = np.full(10, np.nan)
######
for lt in range(0, 10):
	#
	N_errors[lt] = np.sum(np.isnan(Buoys_MA[str(lt)]) == False)
	##########
	CF_buoys_MA_AE = abs(CF_buoys_MA[str(lt)] - Buoys_MA[str(lt)])
	CF_SAR_MA_AE = abs(CF_SAR_MA[str(lt)] - Buoys_MA[str(lt)])
	T4_MA_AE = abs(T4_MA[str(lt)] - Buoys_MA[str(lt)])
	#
	CF_buoys_IB_AE = abs(CF_buoys_IB[str(lt)] - Buoys_IB[str(lt)])
	CF_buoys_IB_AE[CF_buoys_IB_AE > 180] = 360 - CF_buoys_IB_AE[CF_buoys_IB_AE > 180]
	CF_SAR_IB_AE = abs(CF_SAR_IB[str(lt)] - Buoys_IB[str(lt)])
	CF_SAR_IB_AE[CF_SAR_IB_AE > 180] = 360 - CF_SAR_IB_AE[CF_SAR_IB_AE > 180]
	T4_IB_AE = abs(T4_IB[str(lt)] - Buoys_IB[str(lt)])
	T4_IB_AE[T4_IB_AE > 180] = 360 - T4_IB_AE[T4_IB_AE > 180]
	##########
	CF_buoys_MA_MAE[lt] = np.mean(CF_buoys_MA_AE)
	CF_SAR_MA_MAE[lt] = np.mean(CF_SAR_MA_AE)
	T4_MA_MAE[lt] = np.mean(T4_MA_AE)
	CF_buoys_IB_MAE[lt] = np.mean(CF_buoys_IB_AE)
	CF_SAR_IB_MAE[lt] = np.mean(CF_SAR_IB_AE)
	T4_IB_MAE[lt] = np.mean(T4_IB_AE)
	###
	MA_fraction_improved_buoys[lt] = np.nansum(T4_MA_AE - CF_buoys_MA_AE > 0) / np.sum(np.isnan(CF_buoys_MA_AE) == False)
	IB_fraction_improved_buoys[lt] = np.nansum(T4_IB_AE - CF_buoys_IB_AE > 0) / np.sum(np.isnan(CF_buoys_IB_AE) == False)
	MA_fraction_improved_SAR[lt] = np.nansum(T4_MA_AE - CF_SAR_MA_AE > 0) / np.sum(np.isnan(CF_SAR_MA_AE) == False)
	IB_fraction_improved_SAR[lt] = np.nansum(T4_IB_AE - CF_SAR_IB_AE > 0) / np.sum(np.isnan(CF_SAR_IB_AE) == False)
	#
	CF_buoys_MA_CC[lt] = pearsonr(CF_buoys_MA[str(lt)], Buoys_MA[str(lt)])[0]
	CF_buoys_IB_CC[lt] = circular_correlation_coefficient(CF_buoys_IB[str(lt)], Buoys_IB[str(lt)])
	CF_SAR_MA_CC[lt] = pearsonr(CF_SAR_MA[str(lt)], Buoys_MA[str(lt)])[0]
	CF_SAR_IB_CC[lt] = circular_correlation_coefficient(CF_SAR_IB[str(lt)], Buoys_IB[str(lt)])
	T4_MA_CC[lt] = pearsonr(T4_MA[str(lt)], Buoys_MA[str(lt)])[0]
	T4_IB_CC[lt] = circular_correlation_coefficient(T4_IB[str(lt)], Buoys_IB[str(lt)])
	###
	wilcoxon_CF_buoys_vs_T4_MA[lt], Pvalue_CF_buoys_vs_T4_MA[lt] = wilcoxon(CF_buoys_MA_AE, T4_MA_AE, zero_method='wilcox', correction=False, alternative='two-sided', mode='auto')
	wilcoxon_CF_SAR_vs_T4_MA[lt], Pvalue_CF_SAR_vs_T4_MA[lt] = wilcoxon(CF_SAR_MA_AE, T4_MA_AE, zero_method='wilcox', correction=False, alternative='two-sided', mode='auto')
	wilcoxon_CF_buoys_vs_T4_IB[lt], Pvalue_CF_buoys_vs_T4_IB[lt] = wilcoxon(CF_buoys_IB_AE, T4_IB_AE, zero_method='wilcox', correction=False, alternative='two-sided', mode='auto')
	wilcoxon_CF_SAR_vs_T4_IB[lt], Pvalue_CF_SAR_vs_T4_IB[lt] = wilcoxon(CF_SAR_IB_AE, T4_IB_AE, zero_method='wilcox', correction=False, alternative='two-sided', mode='auto')
	wilcoxon_CF_buoys_vs_CF_SAR_MA[lt], Pvalue_CF_buoys_vs_CF_SAR_MA[lt] = wilcoxon(CF_buoys_MA_AE, CF_SAR_MA_AE, zero_method='wilcox', correction=False, alternative='two-sided', mode='auto')
	wilcoxon_CF_buoys_vs_CF_SAR_IB[lt], Pvalue_CF_buoys_vs_CF_SAR_IB[lt] = wilcoxon(CF_buoys_IB_AE, CF_SAR_IB_AE, zero_method='wilcox', correction=False, alternative='two-sided', mode='auto')
	#
	###########################################################
	# Figures Error distribution
	###########################################################
	n_bins = 50
	#
	plt.figure()
	plt.rc('xtick', labelsize = 15)
	plt.rc('ytick', labelsize = 15)
	fig, ax1 = plt.subplots(3, 2, figsize=(15, 20), facecolor='w', edgecolor='k')
	fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
	#
	ax = plt.subplot(321)
	ax.hist(T4_IB_AE, bins = n_bins)
	ax.set_xlabel('T4_IB_AE', fontsize = sizefont)
	#
	ax = plt.subplot(322)
	ax.hist(T4_MA_AE, bins = n_bins)
	ax.set_xlabel('T4_MA_AE', fontsize = sizefont)
	#
	ax = plt.subplot(323)
	ax.hist(CF_buoys_IB_AE, bins = n_bins)
	ax.set_xlabel('CF_buoys_IB_AE', fontsize = sizefont)
	#
	ax = plt.subplot(324)
	ax.hist(CF_buoys_MA_AE, bins = n_bins)
	ax.set_xlabel('CF_buoys_MA_AE', fontsize = sizefont)
	#
	ax = plt.subplot(325)
	ax.hist(CF_SAR_IB_AE, bins = n_bins)
	ax.set_xlabel('CF_SAR_IB_AE', fontsize = sizefont)
	#
	ax = plt.subplot(326)
	ax.hist(CF_SAR_MA_AE, bins = n_bins)
	ax.set_xlabel('CF_SAR_MA_AE', fontsize = sizefont)
	#
	plt.savefig(path_output + 'Absolute_errors_histograms_' + date_test + '_lead_time_' + str(lt) + '_all_buoys.png', bbox_inches='tight')
	plt.clf()
###########################################################
# Statistics
###########################################################
Improvement_MAE_IB_buoys = 100 * (T4_IB_MAE - CF_buoys_IB_MAE) / T4_IB_MAE 
Improvement_MAE_MA_buoys = 100 * (T4_MA_MAE - CF_buoys_MA_MAE) / T4_MA_MAE 
Improvement_MAE_IB_SAR = 100 * (T4_IB_MAE - CF_SAR_IB_MAE) / T4_IB_MAE 
Improvement_MAE_MA_SAR = 100 * (T4_MA_MAE - CF_SAR_MA_MAE) / T4_MA_MAE 
###########################################################
# Print statistics
###########################################################
print('...................................................')
print('........... General statistics ...........')
print('...................................................')
print('Number of errors (min, max, mean)', np.min(N_errors), np.max(N_errors), np.mean(N_errors))
print('...................................................')
print('Improvement MAE direction buoys (%, min, max, mean): ', np.min(Improvement_MAE_IB_buoys), np.max(Improvement_MAE_IB_buoys), np.mean(Improvement_MAE_IB_buoys))
print('Improvement MAE direction SAR (%, min, max, mean): ', np.min(Improvement_MAE_IB_SAR), np.max(Improvement_MAE_IB_SAR), np.mean(Improvement_MAE_IB_SAR))
print('Improvement MAE speed buoys (%, min, max, mean): ', np.min(Improvement_MAE_MA_buoys), np.max(Improvement_MAE_MA_buoys), np.mean(Improvement_MAE_MA_buoys))
print('Improvement MAE speed SAR (%, min, max, mean): ', np.min(Improvement_MAE_MA_SAR), np.max(Improvement_MAE_MA_SAR), np.mean(Improvement_MAE_MA_SAR))
print('...................................................')
print('Fraction of forecasts improved direction buoys (%, min, max, mean): ', 100 * np.min(IB_fraction_improved_buoys), 100 * np.max(IB_fraction_improved_buoys), 100 * np.mean(IB_fraction_improved_buoys))
print('Fraction of forecasts improved direction SAR (%, min, max, mean): ', 100 * np.min(IB_fraction_improved_SAR), 100 * np.max(IB_fraction_improved_SAR), 100 * np.mean(IB_fraction_improved_SAR))
print('Fraction of forecasts improved speed buoys (%, min, max, mean): ', 100 * np.min(MA_fraction_improved_buoys), 100 * np.max(MA_fraction_improved_buoys), 100 * np.mean(MA_fraction_improved_buoys))
print('Fraction of forecasts improved speed SAR (%, min, max, mean): ', 100 * np.min(MA_fraction_improved_SAR), 100 * np.max(MA_fraction_improved_SAR), 100 * np.mean(MA_fraction_improved_SAR))
####################
# Significance
####################
print('...................................................')
print('........... Significance of the results ...........')
print('...................................................')
print('Pvalue_CF_buoys_vs_T4_MA', Pvalue_CF_buoys_vs_T4_MA < threshold_pvalue)
print('Pvalue_CF_SAR_vs_T4_MA', Pvalue_CF_SAR_vs_T4_MA < threshold_pvalue)
print('Pvalue_CF_buoys_vs_CF_SAR_MA', Pvalue_CF_buoys_vs_CF_SAR_MA < threshold_pvalue)
print('Pvalue_CF_buoys_vs_T4_IB', Pvalue_CF_buoys_vs_T4_IB < threshold_pvalue)
print('Pvalue_CF_SAR_vs_T4_IB', Pvalue_CF_SAR_vs_T4_IB < threshold_pvalue)
print('Pvalue_CF_buoys_vs_CF_SAR_IB', Pvalue_CF_buoys_vs_CF_SAR_IB < threshold_pvalue)
plt.figure()
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
fig, ax = plt.subplots(1, 1, figsize=(20, 15), facecolor='w', edgecolor='k')
l1 = ax.plot(np.arange(10) + 1, Pvalue_CF_buoys_vs_T4_IB, color = 'maroon', alpha = alpha, label = 'RF models predicting the direction trained with buoy observations vs TOPAZ4')
l2 = ax.plot(np.arange(10) + 1, Pvalue_CF_SAR_vs_T4_IB, color = 'red', alpha = alpha, label = 'RF models predicting the direction trained with SAR observations vs TOPAZ4') 
l3 = ax.plot(np.arange(10) + 1, Pvalue_CF_buoys_vs_CF_SAR_IB, color = 'orange', alpha = alpha, label = 'RF models predicting the direction trained with buoy vs SAR observations')
l4 = ax.plot(np.arange(10) + 1, Pvalue_CF_buoys_vs_T4_MA, color = 'blue', alpha = alpha, label = 'RF predicting the speed trained with buoy observations vs TOPAZ4') 
l5 = ax.plot(np.arange(10) + 1, Pvalue_CF_SAR_vs_T4_MA, color = 'cyan', alpha = alpha, label = 'RF predicting the speed trained with SAR observations vs TOPAZ4') 
l6 = ax.plot(np.arange(10) + 1, Pvalue_CF_buoys_vs_CF_SAR_MA, color = 'green', alpha = alpha, label = '2 RF models predicting the speed trained with buoy vs SAR observations')
l7 = ax.plot(np.arange(10) + 1, np.repeat(0.05, 10), color = 'gray', alpha = alpha, label = 'Statistical significance threshold')
lns = l1 + l2 + l3 + l4 + l5 + l6
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, fontsize = sizefont_legend, loc = 'best')
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont)
ax.set_ylabel('P-value', fontsize = sizefont)
plt.savefig(path_output + 'P_values_' + date_test + '_all_buoys.png', bbox_inches='tight')
plt.clf()
###########################################################
# Figures MAE and fraction of forecasts improved
###########################################################
plt.figure()
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
fig, ax1 = plt.subplots(2, 3, figsize=(20, 15), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
###################
ax = plt.subplot(241)
l1 = ax.plot(np.arange(10) + 1, T4_IB_MAE, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_T4, label = 'TOPAZ4')
l2 = ax.plot(np.arange(10) + 1, CF_buoys_IB_MAE, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_buoys, label = 'Calibrated forecasts (models \n trained with buoy observations)')
l3 = ax.plot(np.arange(10) + 1, CF_SAR_IB_MAE, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_SAR, label = 'Calibrated forecasts (models \n trained with SAR observations)')
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont)
ax.set_ylabel('Mean absolute error (degrees)', fontsize = sizefont)
ax.set_xlim([0.5, 10.5])
#ax.set_ylim([20, 67])
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_yticks(np.arange(20, 75, step = 5))
ax.set_title('Direction', fontsize = sizefont, fontweight = 'bold')
lns = l1 + l2 + l3
labs = [l.get_label() for l in lns]
#ax.legend(lns, labs, fontsize = sizefont_legend, loc='best')
plt.text(-0.12,-0.1, 'a)', fontsize = sizefont, ha='left', transform = ax.transAxes)
#####
ax = plt.subplot(242)
l1 = ax.plot(np.arange(10) + 1, 100 * (T4_IB_MAE - CF_buoys_IB_MAE) / T4_IB_MAE, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_buoys)
l2 = ax.plot(np.arange(10) + 1, 100 * (T4_IB_MAE - CF_SAR_IB_MAE) / T4_IB_MAE, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_SAR)
#l2 = ax.plot(np.arange(10) + 1, np.repeat(0, 10), 'k', alpha = 0.2)
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont)
ax.set_ylabel(r'$\mathrm{100 x \frac{TOPAZ4\ MAE\ -\ calibrated\ forecast \ MAE}{TOPAZ4\ MAE}}$', fontsize = sizefont * 1.1)
ax.set_xlim([0.5, 10.5])
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_yticks(np.arange(-5, 20, step = 2))
ax.set_title('Direction', fontsize = sizefont, fontweight = 'bold')
plt.text(-0.12,-0.1, 'c)', fontsize = sizefont, ha='left', transform = ax.transAxes)
#####
ax = plt.subplot(243)
l1 = ax.plot(np.arange(10) + 1, 100 * IB_fraction_improved_buoys, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_buoys, label = 'Fraction of forecasts trained with buoys improved')
l2 = ax.plot(np.arange(10) + 1, 100 * IB_fraction_improved_SAR, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_SAR, label = 'Fraction of forecasts trained with SAR improved')
#l2 = ax.plot(np.arange(10) + 0.5, np.repeat(50, 10), 'k', alpha = 0.2)
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont)
ax.set_ylabel('Fraction of forecasts improved (%)', fontsize = sizefont)
ax.set_xlim([0.5, 10.5])
ax.set_ylim([47, 64])
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_title('Direction', fontsize = sizefont, fontweight = 'bold')
plt.text(-0.12,-0.1, 'e)', fontsize = sizefont, ha='left', transform = ax.transAxes)
#####
ax = plt.subplot(244)
l1 = ax.plot(np.arange(10) + 1, T4_IB_CC, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_T4, label = 'TOPAZ4')
l2 = ax.plot(np.arange(10) + 1, CF_buoys_IB_CC, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_buoys, label = 'Calibrated forecasts trained \n with buoy observations')
l3 = ax.plot(np.arange(10) + 1, CF_SAR_IB_CC, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_SAR, label = 'Calibrated forecasts trained \n with SAR observations')
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont)
ax.set_ylabel('Circular correlation coefficient', fontsize = sizefont)
ax.set_xlim([0.5, 10.5])
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_title('Direction', fontsize = sizefont, fontweight = 'bold')
lns = l1 + l2 + l3
labs = [l.get_label() for l in lns]
ax.text(-0.1,-0.1, 'g)', fontsize = sizefont, ha = 'left', transform = ax.transAxes)
###################
ax = plt.subplot(245)
l1 = ax.plot(np.arange(10) + 1, T4_MA_MAE * 0.001, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_T4, label = 'TOPAZ4')
l2 = ax.plot(np.arange(10) + 1, CF_buoys_MA_MAE * 0.001, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_buoys, label = 'Calibrated forecasts trained with buoys')
l3 = ax.plot(np.arange(10) + 1, CF_SAR_MA_MAE * 0.001, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_SAR, label = 'Calibrated forecasts trained with SAR')
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont)
ax.set_ylabel('Mean absolute error (km / day)', fontsize = sizefont)
ax.set_xlim([0.5, 10.5])
#ax.set_ylim([2900, 5600])
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_title('Speed', fontsize = sizefont, fontweight = 'bold')
lns = l1 + l2 + l3
labs = [l.get_label() for l in lns]
#ax.legend(lns, labs, fontsize = sizefont_legend, loc='best')
plt.text(-0.12,-0.1, 'b)', fontsize = sizefont, ha = 'left', transform = ax.transAxes)
#####
ax = plt.subplot(246)
l1 = ax.plot(np.arange(10) + 1, 100 * (T4_MA_MAE - CF_buoys_MA_MAE) / T4_MA_MAE, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_buoys, label = 'Fraction of forecasts trained with buoys improved')
l2 = ax.plot(np.arange(10) + 1, 100 * (T4_MA_MAE - CF_SAR_MA_MAE) / T4_MA_MAE, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_SAR, label = 'Fraction of forecasts trained with SAR improved')
#l2 = ax.plot(np.arange(10) + 1, np.repeat(0, 10), 'k', alpha = 0.2)
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont)
ax.set_ylabel(r'$\mathrm{100 x \frac{TOPAZ4\ MAE\ -\ calibrated\ forecast \ MAE}{TOPAZ4\ MAE}}$', fontsize = sizefont * 1.1)
ax.set_xlim([0.5, 10.5])
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_yticks(np.arange(-3, 16, step = 2))
ax.set_title('Speed', fontsize = sizefont, fontweight = 'bold')
plt.text(-0.12,-0.1, 'd)', fontsize = sizefont, ha = 'left', transform = ax.transAxes)
#####
ax = plt.subplot(247)
l1 = ax.plot(np.arange(10) + 1, 100 * MA_fraction_improved_buoys, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_buoys, label = 'Fraction of forecasts trained with buoys improved')
l1 = ax.plot(np.arange(10) + 1, 100 * MA_fraction_improved_SAR, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_SAR, label = 'Fraction of forecasts trained with SAR improved')
#l2 = ax.plot(np.arange(10) + 1, np.repeat(50, 10), 'k', alpha = 0.2)
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont)
ax.set_ylabel('Fraction of forecasts improved (%)', fontsize = sizefont)
ax.set_xlim([0.5, 10.5])
ax.set_ylim([49.5, 59])
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_title('Speed', fontsize = sizefont, fontweight = 'bold')
plt.text(-0.12,-0.1, 'f)', fontsize = sizefont, ha = 'left', transform = ax.transAxes)
#####
ax = plt.subplot(248)
l1 = ax.plot(np.arange(10) + 1, T4_MA_CC, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_T4, label = 'TOPAZ4', zorder = 10)
l2 = ax.plot(np.arange(10) + 1, CF_buoys_MA_CC, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_buoys, label = 'RF trained with \n buoy observations', zorder = 11)
l3 = ax.plot(np.arange(10) + 1, CF_SAR_MA_CC, linestyle = '--', marker = 'o', markersize = sizemarker, color = color_calib_SAR, label = 'RF trained with \n SAR observations', zorder = 12)
ax.set_xlabel('Forecast lead time (days)', fontsize = sizefont)
ax.set_ylabel('Pearson correlation coefficient', fontsize = sizefont)
ax.set_xlim([0.5, 10.5])
ax.set_xticks(np.arange(1, 11, step = 1))
ax.set_title('Speed', fontsize = sizefont, fontweight = 'bold')
lns = l1 + l2 + l3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, fontsize = sizefont_legend, loc = 'best')
ax.text(-0.1,-0.11, 'h)', fontsize = sizefont, ha = 'left', transform = ax.transAxes)
#####
plt.savefig(path_output + 'Performances_' + date_test + '_all_buoys.png', bbox_inches='tight')
plt.clf()
