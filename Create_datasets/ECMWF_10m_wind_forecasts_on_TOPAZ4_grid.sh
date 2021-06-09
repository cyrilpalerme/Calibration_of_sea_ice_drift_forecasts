#$ -S /bin/bash
#$ -l h_rt=00:30:00
#$ -q research-el7.q
#$ -l h_vmem=10G
#$ -t 366-366
#$ -o /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -cwd

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python/3.7.2

cat > "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/make_ECMWF_10mwind_PA_""$SGE_TASK_ID"".py" << EOF

################################################
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import itertools
import datetime
from pylab import *
from math import *
from matplotlib.colors import BoundaryNorm
from netCDF4 import Dataset
from itertools import chain
from scipy.interpolate import griddata
from pyproj import Proj, transform
#################################################
# Constants
#################################################
path_input = '/lustre/storeB/project/copernicus/svalnav/Data/ECMWF_forecasts/10m_wind_daily/'
path_output = '/lustre/storeB/project/nwp/SALIENSEAS/Ice_drift_calibration_Pan_Arctic/Data/ECMWF_10m_wind_on_T4grid/'
#
proj_EPSG4326 = Proj(init = 'epsg:4326')
proj_T4 = Proj('+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere')
x_T4grid = np.linspace(-38, 38, 609) * 100 * 1000
y_T4grid = np.linspace(-55, 55, 881) * 100 * 1000
xx_T4grid, yy_T4grid = np.meshgrid(x_T4grid, y_T4grid)
lon_T4grid, lat_T4grid = transform(proj_T4, proj_EPSG4326, xx_T4grid, yy_T4grid)
#################################################
# Load ECMWF data
#################################################
dataset_ECMWF = sorted(glob.glob(path_input + 'ECMWF_operational_forecasts_10mwind_*_NH.nc'))
print('len(dataset_ECMWF)', len(dataset_ECMWF))
file_ECMWF = dataset_ECMWF[$SGE_TASK_ID - 1]
start_date_ECMWF = file_ECMWF[file_ECMWF.find('_NH.nc') - 8 : file_ECMWF.find('_NH.nc')]
nc_ECMWF = Dataset(file_ECMWF, 'r')
time_ECMWF = nc_ECMWF.variables['time'][:]
lon_ECMWF = nc_ECMWF.variables['lon'][:]
lat_ECMWF = nc_ECMWF.variables['lat'][:]
u10m_ECMWF = nc_ECMWF.variables['U10M'][:,:,:]
v10m_ECMWF = nc_ECMWF.variables['V10M'][:,:,:]
################################################
# 10m wind for each OSISAF time window
################################################
lead_time_ECMWF = time_ECMWF - time_ECMWF[0]
time_window_SAR_start = np.linspace(0 * 24, 9 * 24, 10)
time_window_SAR_end = np.linspace(1 * 24, 10 * 24, 10)
#
for tw in range(0, len(time_window_SAR_start)):
	lead_time_idx = np.squeeze(np.where(np.logical_and(lead_time_ECMWF >= time_window_SAR_start[tw], lead_time_ECMWF < time_window_SAR_end[tw])))
	if tw == 3:
		u10m_ECMWF_scaled_tw = (18 * np.nanmean(ma.squeeze(u10m_ECMWF[lead_time_idx[0:18],:,:]), axis = 0) + 6 * np.nanmean(ma.squeeze(u10m_ECMWF[lead_time_idx[18:20],:,:]), axis = 0)) / 24
		v10m_ECMWF_scaled_tw = (18 * np.nanmean(ma.squeeze(v10m_ECMWF[lead_time_idx[0:18],:,:]), axis = 0) + 6 * np.nanmean(ma.squeeze(v10m_ECMWF[lead_time_idx[18:20],:,:]), axis = 0)) / 24
	else:	
		u10m_ECMWF_scaled_tw = np.nanmean(ma.squeeze(u10m_ECMWF[lead_time_idx,:,:]), axis = 0)
		v10m_ECMWF_scaled_tw = np.nanmean(ma.squeeze(v10m_ECMWF[lead_time_idx,:,:]), axis = 0)
	################################################
	# Interpolation on TOPAZ4 grid 
	################################################
	lons_ECMWF, lats_ECMWF = np.meshgrid(lon_ECMWF, lat_ECMWF)
	xx_ECMWF_reg, yy_ECMWF_reg = transform(proj_EPSG4326, proj_T4, lons_ECMWF, lats_ECMWF)
	xx_ECMWF_reg_flat = np.ndarray.flatten(xx_ECMWF_reg)
	yy_ECMWF_reg_flat = np.ndarray.flatten(yy_ECMWF_reg)
	u10m_ECMWF_scaled_tw_flat = np.ndarray.flatten(u10m_ECMWF_scaled_tw)
	v10m_ECMWF_scaled_tw_flat = np.ndarray.flatten(v10m_ECMWF_scaled_tw)
	u10m_ECMWF_reg = griddata((xx_ECMWF_reg_flat, yy_ECMWF_reg_flat), u10m_ECMWF_scaled_tw_flat, (x_T4grid[None,:], y_T4grid[:,None]), method = 'nearest')
	v10m_ECMWF_reg = griddata((xx_ECMWF_reg_flat, yy_ECMWF_reg_flat), v10m_ECMWF_scaled_tw_flat, (x_T4grid[None,:], y_T4grid[:,None]), method = 'nearest')
	################################################
	# Calculate wind speed and direction
	################################################
	ws10m_ECMWF_reg = np.sqrt(u10m_ECMWF_reg ** 2 + v10m_ECMWF_reg ** 2)
	wd10m_ECMWF_reg = np.full(np.shape(ws10m_ECMWF_reg), np.nan)
	for yi in range(0, np.shape(ws10m_ECMWF_reg)[0]):
		for xi in range(0, np.shape(ws10m_ECMWF_reg)[1]):
			wd10m_ECMWF_reg[yi, xi] = (180/pi) * atan2(u10m_ECMWF_reg[yi, xi], v10m_ECMWF_reg[yi, xi]) % 360
	################################################
	# Concatenate over lead time
	################################################
	if tw == 0:
		u10m_ECMWF_all = np.expand_dims(u10m_ECMWF_reg, axis = 0)
		v10m_ECMWF_all = np.expand_dims(v10m_ECMWF_reg, axis = 0)
		ws10m_ECMWF_all = np.expand_dims(ws10m_ECMWF_reg, axis = 0)
		wd10m_ECMWF_all = np.expand_dims(wd10m_ECMWF_reg, axis = 0)
	else:
		u10m_ECMWF_all = np.concatenate((u10m_ECMWF_all, np.expand_dims(u10m_ECMWF_reg, axis = 0)), axis = 0)
		v10m_ECMWF_all = np.concatenate((v10m_ECMWF_all, np.expand_dims(v10m_ECMWF_reg, axis = 0)), axis = 0)
		ws10m_ECMWF_all = np.concatenate((ws10m_ECMWF_all, np.expand_dims(ws10m_ECMWF_reg, axis = 0)), axis = 0)
		wd10m_ECMWF_all = np.concatenate((wd10m_ECMWF_all, np.expand_dims(wd10m_ECMWF_reg, axis = 0)), axis = 0)
################################################
# Output NetCDF file
################################################
output_filename = 'ECMWF_wind_forecasts_' + start_date_ECMWF + '.nc'
output_netcdf = Dataset(path_output + output_filename,'w',format='NETCDF4')
#
x = output_netcdf.createDimension('x', len(x_T4grid))
y = output_netcdf.createDimension('y', len(y_T4grid))
time = output_netcdf.createDimension('time', len(time_window_SAR_start))
#
x = output_netcdf.createVariable('x', 'd', ('x'))
y = output_netcdf.createVariable('y', 'd', ('y'))
time_start = output_netcdf.createVariable('time_start', 'd', ('time'))
time_end = output_netcdf.createVariable('time_end', 'd', ('time'))
latitude = output_netcdf.createVariable('lat', 'd', ('y','x'))
longitude = output_netcdf.createVariable('lon', 'd', ('y','x'))
u10m = output_netcdf.createVariable('u10m', 'd', ('time','y','x'))
v10m = output_netcdf.createVariable('v10m', 'd', ('time','y','x'))
ws10m = output_netcdf.createVariable('ws10m', 'd', ('time','y','x'))
wd10m = output_netcdf.createVariable('wd10m', 'd', ('time','y','x'))
#
x.units = '100 km'
x.standard_name = 'projection_x_coordinate'
y.units = '100 km'
y.standard_name = 'projection_y_coordinate'
time_start.units = 'hours since the start date'
time_start.standard_name = 'time_start'
time_end.units = 'hours since the start date'
time_end.standard_name = 'time_end'
latitude.units = 'degrees_north'
latitude.standard_name = 'latitude'
longitude.units = 'degrees_east'
longitude.standard_name = 'longitude'
u10m.units = 'm/s'
u10m.standard_name = '10 meter U wind component'
v10m.units = 'm/s'
v10m.standard_name = '10 meter V wind component'
ws10m.units = 'm/s'
ws10m.standard_name = '10 meter wind speed'
wd10m.units = 'degree'
wd10m.standard_name = '10 meter wind direction'
#
x[:] = x_T4grid * 0.00001
y[:] = y_T4grid * 0.00001
longitude[:,:] = lon_T4grid
latitude[:,:] = lat_T4grid
time_start[:] = time_window_SAR_start
time_end[:] = time_window_SAR_end
u10m[:,:,:] = u10m_ECMWF_all
v10m[:,:,:] = v10m_ECMWF_all
ws10m[:,:,:] = ws10m_ECMWF_all
wd10m[:,:,:] = wd10m_ECMWF_all
#
output_netcdf.description = '+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere'
output_netcdf.close()
#################################################
EOF

python3 "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/make_ECMWF_10mwind_PA_""$SGE_TASK_ID"".py"
