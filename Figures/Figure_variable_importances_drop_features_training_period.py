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
path_IB_buoys = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Tuning/Results_without_dayofyear/Direction/Drop_feature_training_period_buoys/'
path_MA_buoys = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Tuning/Results_without_dayofyear/Speed/Drop_feature_training_period_buoys/'
path_IB_SAR = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Tuning/Results_without_dayofyear/Direction/Drop_feature_training_period_SAR/'
path_MA_SAR = '/lustre/storeB/users/cyrilp/Svalnav/April_2021/Tuning/Results_without_dayofyear/Speed/Drop_feature_training_period_SAR/'
#
# Figures parameters
sizefont = 27
sizefont_legend = 25
sizefont_performances = 20
sizemarker = 20
alphamarker = 0.7
xmin = 0.5
xmax = 10.5
mean_lt = np.arange(10) + 0.5
#
# hyperameter 
n_estimators = 200
bootstrap = True
max_features = 3
min_samples_leaf = 1
min_samples_split = 2
max_depth = None
#
date_str_buoys = '20130606-20200528'
date_str_SAR = '20180104-20200528'
test_size = '0.2'
#
rf_param_str = 'nestimators_' + str(n_estimators) + '_maxfeatures_' + str(max_features) + '_bootstrap_' + str(bootstrap) + '_maxdepth_' + str(max_depth) + \
		'_minsamplessplit_' + str(min_samples_split) + '_minsamplesleaf_' + str(min_samples_leaf)
#
feat_list = ['ECMWF_wd10m', 'ECMWF_ws10m', 'T4_drift_initial_bearing', 'T4_drift_magnitude', 'T4_fice', 'T4_hice', 'OSISAF_SIC', 'distance_to_land', 'xc', 'yc', 'all']
feat_list_ticks = ['ECMWF wind direction', 'ECMWF wind speed', 'TOPAZ4 drift direction', 'TOPAZ4 drift speed', 'TOPAZ4 sea ice concentration', 'TOPAZ4 sea ice thickness', 'OSISAF sea ice concentration', 'Distance to land', 'x coordinate', 'y coordinate']
#
#colorscale = plt.cm.tab20b
#colorscale = plt.cm.plasma
colorscale = plt.cm.gnuplot
#colors = colorscale(np.linspace(0, 1, len(feat_list)))
colors = colorscale(np.linspace(0, 1, 11))
###########################################################
# Create datasets
###########################################################
Mean_forecast_error_IB_buoys = {}
Mean_frac_improved_IB_buoys = {}
Std_forecast_error_IB_buoys = {}
Std_frac_improved_IB_buoys = {}
#
Mean_forecast_error_MA_buoys = {}
Mean_frac_improved_MA_buoys = {}
Std_forecast_error_MA_buoys = {}
Std_frac_improved_MA_buoys = {}
###
Mean_forecast_error_IB_SAR = {}
Mean_frac_improved_IB_SAR = {}
Std_forecast_error_IB_SAR = {}
Std_frac_improved_IB_SAR = {}
#
Mean_forecast_error_MA_SAR = {}
Mean_frac_improved_MA_SAR = {}
Std_forecast_error_MA_SAR = {}
Std_frac_improved_MA_SAR = {}
#####
for lt in range(0, 10):
	lt_str = str(lt) + '-' + str(lt + 1)
	#
	file_IB_buoys_all = path_IB_buoys + 'IB_' + date_str_buoys + '_ts_' + test_size + '_' + rf_param_str + '_' + lt_str + '_dropfeat_all.dat'
	file_MA_buoys_all = path_MA_buoys + 'MA_' + date_str_buoys + '_ts_' + test_size + '_' + rf_param_str + '_' + lt_str + '_dropfeat_all.dat'
	df_IB_buoys_all = pd.read_csv(file_IB_buoys_all, delimiter = '\t')
	df_MA_buoys_all = pd.read_csv(file_MA_buoys_all, delimiter = '\t')
	#
	for drop_feat in range(0, len(feat_list)):
		dvar = feat_list[drop_feat]
		#
		file_IB_buoys_dropfeat = path_IB_buoys + 'IB_' + date_str_buoys + '_ts_' + test_size + '_' + rf_param_str + '_' + lt_str + '_dropfeat_' + dvar + '.dat'
		file_MA_buoys_dropfeat = path_MA_buoys + 'MA_' + date_str_buoys + '_ts_' + test_size + '_' + rf_param_str + '_' + lt_str + '_dropfeat_' + dvar + '.dat'
		df_IB_buoys_dropfeat = pd.read_csv(file_IB_buoys_dropfeat, delimiter = '\t')
		df_MA_buoys_dropfeat = pd.read_csv(file_MA_buoys_dropfeat, delimiter = '\t')
		#
		Diff_Mean_forecast_error_IB_buoys = np.mean(df_IB_buoys_dropfeat['Mean_forecast_error'] - df_IB_buoys_all['Mean_forecast_error'])
		Diff_Mean_frac_improved_IB_buoys = np.mean(df_IB_buoys_dropfeat['Fraction_of_forecasts_improved'] - df_IB_buoys_all['Fraction_of_forecasts_improved'])
		Diff_Std_forecast_error_IB_buoys = np.std(df_IB_buoys_dropfeat['Mean_forecast_error'] - df_IB_buoys_all['Mean_forecast_error'])
		Diff_Std_frac_improved_IB_buoys = np.std(df_IB_buoys_dropfeat['Fraction_of_forecasts_improved'] - df_IB_buoys_all['Fraction_of_forecasts_improved'])
		#
		Diff_Mean_forecast_error_MA_buoys = np.mean(df_MA_buoys_dropfeat['Mean_forecast_error'] - df_MA_buoys_all['Mean_forecast_error'])
		Diff_Mean_frac_improved_MA_buoys = np.mean(df_MA_buoys_dropfeat['Fraction_of_forecasts_improved'] - df_MA_buoys_all['Fraction_of_forecasts_improved'])
		Diff_Std_forecast_error_MA_buoys = np.std(df_MA_buoys_dropfeat['Mean_forecast_error'] - df_MA_buoys_all['Mean_forecast_error'])
		Diff_Std_frac_improved_MA_buoys = np.std(df_MA_buoys_dropfeat['Fraction_of_forecasts_improved'] - df_MA_buoys_all['Fraction_of_forecasts_improved'])
		#
		if lt == 0:
			Mean_forecast_error_IB_buoys[dvar] = Diff_Mean_forecast_error_IB_buoys
			Mean_frac_improved_IB_buoys[dvar] = Diff_Mean_frac_improved_IB_buoys
			Std_forecast_error_IB_buoys[dvar] = Diff_Std_forecast_error_IB_buoys
			Std_frac_improved_IB_buoys[dvar] = Diff_Std_frac_improved_IB_buoys
			#
			Mean_forecast_error_MA_buoys[dvar] = Diff_Mean_forecast_error_MA_buoys
			Mean_frac_improved_MA_buoys[dvar] = Diff_Mean_frac_improved_MA_buoys
			Std_forecast_error_MA_buoys[dvar] = Diff_Std_forecast_error_MA_buoys
			Std_frac_improved_MA_buoys[dvar] = Diff_Std_frac_improved_MA_buoys
		else:
			Mean_forecast_error_IB_buoys[dvar] = np.hstack((Mean_forecast_error_IB_buoys.get(dvar), Diff_Mean_forecast_error_IB_buoys))
			Mean_frac_improved_IB_buoys[dvar] = np.hstack((Mean_frac_improved_IB_buoys.get(dvar), Diff_Mean_frac_improved_IB_buoys))
			Std_forecast_error_IB_buoys[dvar] = np.hstack((Std_forecast_error_IB_buoys.get(dvar), Diff_Std_forecast_error_IB_buoys))
			Std_frac_improved_IB_buoys[dvar] = np.hstack((Std_frac_improved_IB_buoys.get(dvar), Diff_Std_frac_improved_IB_buoys))
			#
			Mean_forecast_error_MA_buoys[dvar] = np.hstack((Mean_forecast_error_MA_buoys.get(dvar), Diff_Mean_forecast_error_MA_buoys))
			Mean_frac_improved_MA_buoys[dvar] = np.hstack((Mean_frac_improved_MA_buoys.get(dvar), Diff_Mean_frac_improved_MA_buoys))
			Std_forecast_error_MA_buoys[dvar] = np.hstack((Std_forecast_error_MA_buoys.get(dvar), Diff_Std_forecast_error_MA_buoys))
			Std_frac_improved_MA_buoys[dvar] = np.hstack((Std_frac_improved_MA_buoys.get(dvar), Diff_Std_frac_improved_MA_buoys))
#####
for lt in range(0, 10):
	lt_str = str(lt) + '-' + str(lt + 1)
	#
	file_IB_SAR_all = path_IB_SAR + 'IB_' + date_str_SAR + '_ts_' + test_size + '_' + rf_param_str + '_' + lt_str + '_dropfeat_all.dat'
	file_MA_SAR_all = path_MA_SAR + 'MA_' + date_str_SAR + '_ts_' + test_size + '_' + rf_param_str + '_' + lt_str + '_dropfeat_all.dat'
	df_IB_SAR_all = pd.read_csv(file_IB_SAR_all, delimiter = '\t')
	df_MA_SAR_all = pd.read_csv(file_MA_SAR_all, delimiter = '\t')
	#
	for drop_feat in range(0, len(feat_list)):
		dvar = feat_list[drop_feat]
		#
		file_IB_SAR_dropfeat = path_IB_SAR + 'IB_' + date_str_SAR + '_ts_' + test_size + '_' + rf_param_str + '_' + lt_str + '_dropfeat_' + dvar + '.dat'
		file_MA_SAR_dropfeat = path_MA_SAR + 'MA_' + date_str_SAR + '_ts_' + test_size + '_' + rf_param_str + '_' + lt_str + '_dropfeat_' + dvar + '.dat'
		df_IB_SAR_dropfeat = pd.read_csv(file_IB_SAR_dropfeat, delimiter = '\t')
		df_MA_SAR_dropfeat = pd.read_csv(file_MA_SAR_dropfeat, delimiter = '\t')
		#
		Diff_Mean_forecast_error_IB_SAR = np.mean(df_IB_SAR_dropfeat['Mean_forecast_error'] - df_IB_SAR_all['Mean_forecast_error'])
		Diff_Mean_frac_improved_IB_SAR = np.mean(df_IB_SAR_dropfeat['Fraction_of_forecasts_improved'] - df_IB_SAR_all['Fraction_of_forecasts_improved'])
		Diff_Std_forecast_error_IB_SAR = np.std(df_IB_SAR_dropfeat['Mean_forecast_error'] - df_IB_SAR_all['Mean_forecast_error'])
		Diff_Std_frac_improved_IB_SAR = np.std(df_IB_SAR_dropfeat['Fraction_of_forecasts_improved'] - df_IB_SAR_all['Fraction_of_forecasts_improved'])
		#
		Diff_Mean_forecast_error_MA_SAR = np.mean(df_MA_SAR_dropfeat['Mean_forecast_error'] - df_MA_SAR_all['Mean_forecast_error'])
		Diff_Mean_frac_improved_MA_SAR = np.mean(df_MA_SAR_dropfeat['Fraction_of_forecasts_improved'] - df_MA_SAR_all['Fraction_of_forecasts_improved'])
		Diff_Std_forecast_error_MA_SAR = np.std(df_MA_SAR_dropfeat['Mean_forecast_error'] - df_MA_SAR_all['Mean_forecast_error'])
		Diff_Std_frac_improved_MA_SAR = np.std(df_MA_SAR_dropfeat['Fraction_of_forecasts_improved'] - df_MA_SAR_all['Fraction_of_forecasts_improved'])
		#
		if lt == 0:
			Mean_forecast_error_IB_SAR[dvar] = Diff_Mean_forecast_error_IB_SAR
			Mean_frac_improved_IB_SAR[dvar] = Diff_Mean_frac_improved_IB_SAR
			Std_forecast_error_IB_SAR[dvar] = Diff_Std_forecast_error_IB_SAR
			Std_frac_improved_IB_SAR[dvar] = Diff_Std_frac_improved_IB_SAR
			#
			Mean_forecast_error_MA_SAR[dvar] = Diff_Mean_forecast_error_MA_SAR
			Mean_frac_improved_MA_SAR[dvar] = Diff_Mean_frac_improved_MA_SAR
			Std_forecast_error_MA_SAR[dvar] = Diff_Std_forecast_error_MA_SAR
			Std_frac_improved_MA_SAR[dvar] = Diff_Std_frac_improved_MA_SAR
		else:
			Mean_forecast_error_IB_SAR[dvar] = np.hstack((Mean_forecast_error_IB_SAR.get(dvar), Diff_Mean_forecast_error_IB_SAR))
			Mean_frac_improved_IB_SAR[dvar] = np.hstack((Mean_frac_improved_IB_SAR.get(dvar), Diff_Mean_frac_improved_IB_SAR))
			Std_forecast_error_IB_SAR[dvar] = np.hstack((Std_forecast_error_IB_SAR.get(dvar), Diff_Std_forecast_error_IB_SAR))
			Std_frac_improved_IB_SAR[dvar] = np.hstack((Std_frac_improved_IB_SAR.get(dvar), Diff_Std_frac_improved_IB_SAR))
			#
			Mean_forecast_error_MA_SAR[dvar] = np.hstack((Mean_forecast_error_MA_SAR.get(dvar), Diff_Mean_forecast_error_MA_SAR))
			Mean_frac_improved_MA_SAR[dvar] = np.hstack((Mean_frac_improved_MA_SAR.get(dvar), Diff_Mean_frac_improved_MA_SAR))
			Std_forecast_error_MA_SAR[dvar] = np.hstack((Std_forecast_error_MA_SAR.get(dvar), Diff_Std_forecast_error_MA_SAR))
			Std_frac_improved_MA_SAR[dvar] = np.hstack((Std_frac_improved_MA_SAR.get(dvar), Diff_Std_frac_improved_MA_SAR))
#################################################
# Figure performances
#################################################
plt.figure()
plt.rc('xtick', labelsize = sizefont)
plt.rc('ytick', labelsize = sizefont)
fig, ax1 = plt.subplots(2, 2, figsize = (25, 25), facecolor = 'w', edgecolor = 'k')
fig.subplots_adjust(hspace = 0.5, wspace = 0.25)
#####
ax = plt.subplot(2,2,1)
for v in range(0, len(feat_list) - 1):
	dvar = feat_list[v]
	#
	for lt in range(0, 10):
		if v == 0:
			if lt == 0:
				label_str = str(lt + 1) + ' day'
			else:
				label_str = str(lt + 1) + ' days'
			#	
			l1 = ax.plot(v + 1, Mean_forecast_error_IB_buoys['all'][lt] - Mean_forecast_error_IB_buoys[dvar][lt], color = colors[lt], marker = 'X', markersize = sizemarker, alpha = alphamarker, linestyle = 'None', label = label_str)
		else:
			l1 = ax.plot(v + 1, Mean_forecast_error_IB_buoys['all'][lt] - Mean_forecast_error_IB_buoys[dvar][lt], color = colors[lt], marker = 'X', markersize = sizemarker, alpha = alphamarker)
ax.grid()
ax.legend(fontsize = sizefont_legend, loc = 'lower right', ncol = 2)
ax.set_title('Direction \n models trained with buoy observations', fontsize = 25, fontweight = 'bold')
ax.set_ylabel('Difference in mean absolute error \n (degrees)', fontsize = sizefont)
ax.set_ylim([-7.5, 0.5])
plt.xticks(np.arange(len(feat_list_ticks)) + 1, feat_list_ticks, rotation = 30, ha = 'right', fontsize = sizefont)
ax.text(-0.18, 0.00, 'a)', fontsize = sizefont * 1.2, color = 'k', transform=ax.transAxes)
#####
ax = plt.subplot(2,2,2)
for v in range(0, len(feat_list) - 1):
	dvar = feat_list[v]
	#
	for lt in range(0, 10):
		if v == 0:
			if lt == 0:
				label_str = 'lead time: ' + str(lt + 1) + ' day'
			else:
				label_str = 'lead time: ' + str(lt + 1) + ' days'
			#
			l1 = ax.plot(v + 1, Mean_forecast_error_MA_buoys['all'][lt] - Mean_forecast_error_MA_buoys[dvar][lt], color = colors[lt], marker = 'X', markersize = sizemarker, alpha = alphamarker, label = label_str)
		else:
			l1 = ax.plot(v + 1, Mean_forecast_error_MA_buoys['all'][lt] - Mean_forecast_error_MA_buoys[dvar][lt], color = colors[lt], marker = 'X', markersize = sizemarker, alpha = alphamarker)
ax.grid()
#ax.legend(fontsize = sizefont_legend_performances, loc = 'upper left', ncol = 1)
ax.set_title('Speed \n models trained with buoy observations ', fontsize = 25, fontweight = 'bold')
ax.set_ylabel('Difference in mean absolute error \n (meters / day)', fontsize = sizefont)
ax.set_ylim([-470, 30])
plt.xticks(np.arange(len(feat_list_ticks)) + 1, feat_list_ticks, rotation = 30, ha = 'right', fontsize = sizefont)
ax.text(-0.23, 0.00, 'c)', fontsize = sizefont * 1.2, color = 'k', transform=ax.transAxes)
#####
ax = plt.subplot(2,2,3)
for v in range(0, len(feat_list) - 1):
	dvar = feat_list[v]
	#
	for lt in range(0, 10):
		if v == 0:
			if lt == 0:
				label_str = str(lt + 1) + ' day'
			else:
				label_str = str(lt + 1) + ' days'
			#	
			l1 = ax.plot(v + 1, Mean_forecast_error_IB_SAR['all'][lt] - Mean_forecast_error_IB_SAR[dvar][lt], color = colors[lt], marker = 'X', markersize = sizemarker, alpha = alphamarker, label = label_str)
		else:
			l1 = ax.plot(v + 1, Mean_forecast_error_IB_SAR['all'][lt] - Mean_forecast_error_IB_SAR[dvar][lt], color = colors[lt], marker = 'X', markersize = sizemarker, alpha = alphamarker)
ax.grid()
#ax.legend(fontsize = sizefont_legend_performances, loc = 'lower right', ncol = 2)
ax.set_title('Direction \n models trained with SAR observations', fontsize = 25, fontweight = 'bold')
ax.set_ylabel('Difference in mean absolute error \n (degrees)', fontsize = sizefont)
ax.set_ylim([-7.5, 0.5])
plt.xticks(np.arange(len(feat_list_ticks)) + 1, feat_list_ticks, rotation = 30, ha = 'right', fontsize = sizefont)
ax.text(-0.18, 0.00, 'b)', fontsize = sizefont * 1.2, color = 'k', transform=ax.transAxes)
#####
ax = plt.subplot(2,2,4)
for v in range(0, len(feat_list) - 1):
	dvar = feat_list[v]
	#
	for lt in range(0, 10):
		if v == 0:
			if lt == 0:
				label_str = 'lead time: ' + str(lt + 1) + ' day'
			else:
				label_str = 'lead time: ' + str(lt + 1) + ' days'
			#
			l1 = ax.plot(v + 1, Mean_forecast_error_MA_SAR['all'][lt] - Mean_forecast_error_MA_SAR[dvar][lt], color = colors[lt], marker = 'X', markersize = sizemarker, alpha = alphamarker, label = label_str)
		else:
			l1 = ax.plot(v + 1, Mean_forecast_error_MA_SAR['all'][lt] - Mean_forecast_error_MA_SAR[dvar][lt], color = colors[lt], marker = 'X', markersize = sizemarker, alpha = alphamarker)
ax.grid()
#ax.legend(fontsize = sizefont_legend_performances, loc = 'upper left', ncol = 1)
ax.set_title('Speed \n models trained with SAR observations', fontsize = 25, fontweight = 'bold')
ax.set_ylabel('Difference in mean absolute error \n (meters / day)', fontsize = sizefont)
ax.set_ylim([-470, 30])
plt.xticks(np.arange(len(feat_list_ticks)) + 1, feat_list_ticks, rotation = 30, ha = 'right', fontsize = sizefont)
ax.text(-0.23, 0.00, 'd)', fontsize = sizefont * 1.2, color = 'k', transform=ax.transAxes)
#####
plt.savefig(path_output + 'Drop_features_training_period.png' , bbox_inches='tight', dpi = 200)
fig.clf()
