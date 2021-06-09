#$ -S /bin/bash
#$ -l h_rt=00:30:00
#$ -q research-el7.q
#$ -l h_vmem=1G
#$ -t 1-33
#$ -o /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/OUT/SAR_OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/ERR/SAR_ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -cwd

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python/3.7.2

cat > "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/SAR_magnitude_direction_PA_""$SGE_TASK_ID"".py" << EOF
#################################################
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import cmath
import itertools
from itertools import chain
from math import *
from pylab import *
from matplotlib.colors import BoundaryNorm
from netCDF4 import Dataset
from pyproj import Proj, transform
from scipy.interpolate import griddata
#################################################
# Constants
#################################################
path_input = '/lustre/storeB/project/copernicus/svalnav/Data/SAR_observations/SAR_CMEMS_version_2.0/'
path_output = '/lustre/storeB/project/copernicus/svalnav/Data_operational_ice_drift_forecasts/Pan_Arctic/SAR_v2/'
#
proj_EPSG4326 = Proj(init = 'epsg:4326')
proj_SAR = Proj('+proj=stere +R=6370997 +lat_0=90 +lat_ts=70 +lon_0=0')
proj_T4 = Proj('+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere')
#
x_T4grid = np.linspace(-38, 38, 609) * 100 * 1000
y_T4grid = np.linspace(-55, 55, 881) * 100 * 1000
xx_T4grid, yy_T4grid = np.meshgrid(x_T4grid, y_T4grid)
lon_T4grid, lat_T4grid = transform(proj_T4, proj_EPSG4326, xx_T4grid, yy_T4grid)
#################################################
# Dataset SAR
#################################################
paths = []
for year in range(2021, 2022):
	yearstr=str(year)
	for month in range(5, 7):
		monthstr = "{:02d}".format(month)
		p = path_input + yearstr + '/' + monthstr + '/'
		if os.path.isdir(p) == True:
			paths.append(p)
#
dataset = []
for path, dirs, files in chain.from_iterable(os.walk(path) for path in paths):
	nh_files = [s for s in files if "ice_drift_mosaic_polstereo_sarbased_north_" in s]
	sorted_files = sorted(nh_files)
	for i in range(0, len(sorted_files)):
		if sorted_files[i][-9:-3] == '000000':
			dataset.append(sorted(glob.glob(path + sorted_files[i])))
print('len(dataset)', len(dataset))
#################################################
# Functions
#################################################
# Computes the great circle distance between two points using the haversine formula. Values can be vectors.
def great_circle_distance(lon1, lat1, lon2, lat2):
	# Convert from degrees to radians
	pi = 3.14159265
	lon1 = lon1 * 2 * pi / 360.
	lat1 = lat1 * 2 * pi / 360.
	lon2 = lon2 * 2 * pi / 360.
	lat2 = lat2 * 2 * pi / 360.
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = np.sin(dlat / 2.) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.) ** 2
	c = 2 * np.arcsin(np.sqrt(a))
	distance = 6.371e6 * c
	return distance
###
# Calculates the bearing between two points.
def calculate_initial_compass_bearing(pointA, pointB):
	if (type(pointA) != tuple) or (type(pointB) != tuple):
		raise TypeError("Only tuples are supported as arguments")
	lat1 = math.radians(pointA[0])
	lat2 = math.radians(pointB[0])
	diffLong = math.radians(pointB[1] - pointA[1])
	x = math.sin(diffLong) * math.cos(lat2)
	y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
	initial_bearing = math.atan2(x, y)
	initial_bearing = math.degrees(initial_bearing)
	compass_bearing = (initial_bearing + 360) % 360
	return compass_bearing
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
#################################################
# Load data
#################################################
file_SAR = dataset[$SGE_TASK_ID - 1][0]
filename_SAR = os.path.basename(file_SAR)
file_start_date = filename_SAR[-32:-18]
file_end_date = filename_SAR[-17:-3]
#
dX_mean_SAR = np.full((900, 900), np.nan)
dY_mean_SAR = np.full((900, 900), np.nan)
dX_std_SAR = np.full((900, 900), np.nan)
dY_std_SAR = np.full((900, 900), np.nan)
dXY_count_SAR = np.full((900, 900), np.nan)
time_first_SAR = np.full((900, 900), np.nan)
time_last_SAR = np.full((900, 900), np.nan)
#
nc_SAR = Dataset(file_SAR, 'r')
time_SAR = nc_SAR.variables['time'][:]
xc_SAR = nc_SAR.variables['xc'][:] * 1000
yc_SAR = nc_SAR.variables['yc'][:] * 1000
lat_SAR = nc_SAR.variables['lat'][:,:]
lon_SAR = nc_SAR.variables['lon'][:,:]
status_flag_SAR = nc_SAR.variables['status_flag'][:]
#
dX_mean = nc_SAR.variables['dX_mean'][0, 1:-1, 1:-1] * 1000
dY_mean = nc_SAR.variables['dY_mean'][0, 1:-1, 1:-1] * 1000
dX_std = nc_SAR.variables['dX_std'][0, 1:-1, 1:-1] * 1000
dY_std = nc_SAR.variables['dY_std'][0, 1:-1, 1:-1] * 1000
dXY_count = nc_SAR.variables['dXY_count'][0, 1:-1, 1:-1]
time_first = nc_SAR.variables['time_first'][0, 1:-1, 1:-1]
time_last = nc_SAR.variables['time_last'][0, 1:-1, 1:-1]
#
dX_mean[dX_mean.mask == True] = np.nan
dY_mean[dY_mean.mask == True] = np.nan
dX_std[dX_std.mask == True] = np.nan
dY_std[dY_std.mask == True] = np.nan
dXY_count[dXY_count.mask == True] = np.nan
time_first[time_first.mask == True] = np.nan
time_last[time_last.mask == True] = np.nan
#
dX_mean_SAR[1:-1, 1:-1] = dX_mean
dY_mean_SAR[1:-1, 1:-1] = dY_mean
dX_std_SAR[1:-1, 1:-1] = dX_std
dY_std_SAR[1:-1, 1:-1] = dY_std
dXY_count_SAR[1:-1, 1:-1] = dXY_count
time_first_SAR[1:-1, 1:-1] = time_first
time_last_SAR[1:-1, 1:-1] = time_last
#################################################
# Initial bearing and magnitude
#################################################
xx_SAR, yy_SAR = np.meshgrid(xc_SAR, yc_SAR)
xx_ice_end_SAR = xx_SAR + dX_mean_SAR
yy_ice_end_SAR = yy_SAR + dY_mean_SAR
lon_ice_end_SAR, lat_ice_end_SAR = transform(proj_SAR, proj_EPSG4326, xx_ice_end_SAR, yy_ice_end_SAR)
Dist_ice_SAR = great_circle_distance(lon_SAR, lat_SAR, lon_ice_end_SAR, lat_ice_end_SAR)
initial_bearing_ice_SAR = np.full(np.shape(Dist_ice_SAR), np.nan)
for i in range(0, np.shape(Dist_ice_SAR)[0]):
	for j in range(0, np.shape(Dist_ice_SAR)[1]):
		A_coord_SAR = (lat_SAR[i,j], lon_SAR[i,j])
		B_coord_ice_SAR = (lat_ice_end_SAR[i,j], lon_ice_end_SAR[i,j])
		initial_bearing_ice_SAR[i,j] = calculate_initial_compass_bearing(A_coord_SAR, B_coord_ice_SAR)
#################################################
# Interpolation on TOPAZ4 grid
#################################################
xx_SAR_reg, yy_SAR_reg = transform(proj_SAR, proj_T4, xx_SAR, yy_SAR)
xx_SAR_reg_flat = np.ndarray.flatten(xx_SAR_reg)
yy_SAR_reg_flat = np.ndarray.flatten(yy_SAR_reg)
Dist_ice_SAR_flat = np.ndarray.flatten(Dist_ice_SAR)
initial_bearing_ice_SAR_flat = np.ndarray.flatten(initial_bearing_ice_SAR)
dXY_count_SAR_flat = np.ndarray.flatten(dXY_count_SAR)
time_first_SAR_flat = np.ndarray.flatten(time_first_SAR)
time_last_SAR_flat = np.ndarray.flatten(time_last_SAR)
#
Dist_ice_T4 = griddata((xx_SAR_reg_flat, yy_SAR_reg_flat), Dist_ice_SAR_flat, (x_T4grid[None,:], y_T4grid[:,None]), method = 'nearest')
initial_bearing_ice_T4 = griddata((xx_SAR_reg_flat, yy_SAR_reg_flat), initial_bearing_ice_SAR_flat, (x_T4grid[None,:], y_T4grid[:,None]), method = 'nearest')
dXY_count_reg = griddata((xx_SAR_reg_flat, yy_SAR_reg_flat), dXY_count_SAR_flat, (x_T4grid[None,:], y_T4grid[:,None]), method = 'nearest')
time_first_reg = griddata((xx_SAR_reg_flat, yy_SAR_reg_flat), time_first_SAR_flat, (x_T4grid[None,:], y_T4grid[:,None]), method = 'nearest')
time_last_reg = griddata((xx_SAR_reg_flat, yy_SAR_reg_flat), time_last_SAR_flat, (x_T4grid[None,:], y_T4grid[:,None]), method = 'nearest')
#
Dist_ice_T4[Dist_ice_T4 > 6000000] = np.nan
initial_bearing_ice_T4[Dist_ice_T4 > 6000000] = np.nan
##################################################
## x and y displacements
print('x and y displacements')
##################################################
lat_ice_end_T4 = np.full(np.shape(Dist_ice_T4), np.nan)
lon_ice_end_T4 = np.full(np.shape(Dist_ice_T4), np.nan)
for i in range(0, len(y_T4grid)):
	for j in range(0, len(x_T4grid)):
		lat_ice_end_T4[i,j], lon_ice_end_T4[i,j] = destination_coordinates(lat_T4grid[i,j], lon_T4grid[i,j], initial_bearing_ice_T4[i,j], Dist_ice_T4[i,j])
xx_ice_end_T4, yy_ice_end_T4 = transform(proj_EPSG4326, proj_T4, lon_ice_end_T4, lat_ice_end_T4)
xice_T4 = xx_ice_end_T4 - xx_T4grid
yice_T4 = yy_ice_end_T4 - yy_T4grid
#################################################
# NetCDF
#################################################
output_filename = 'SAR_drift_' + file_start_date + '-' + file_end_date + '.nc'
output_netcdf = Dataset(path_output + output_filename, 'w', format = 'NETCDF4')
#
x = output_netcdf.createDimension('x', len(x_T4grid))
y = output_netcdf.createDimension('y', len(y_T4grid))
time = output_netcdf.createDimension('time', 1)
status = output_netcdf.createDimension('status', 1)
#
x = output_netcdf.createVariable('x', 'd', ('x'))
y = output_netcdf.createVariable('y', 'd', ('y'))
time = output_netcdf.createVariable('time', 'd', ('time'))
status_flag = output_netcdf.createVariable('status_flag', 'd', ('status'))
latitude = output_netcdf.createVariable('latitude', 'd', ('y','x'))
longitude = output_netcdf.createVariable('longitude', 'd', ('y','x'))
uice = output_netcdf.createVariable('uice', 'd', ('y','x'))
vice = output_netcdf.createVariable('vice', 'd', ('y','x'))
dXY_count = output_netcdf.createVariable('dXY_count', 'd', ('y','x'))
time_first = output_netcdf.createVariable('time_first', 'd', ('y','x'))
time_last = output_netcdf.createVariable('time_last', 'd', ('y','x'))
drift_magnitude = output_netcdf.createVariable('drift_magnitude', 'd', ('y','x'))
drift_initial_bearing = output_netcdf.createVariable('drift_initial_bearing', 'd', ('y','x'))
#
x.units = 'km'
x.standard_name = 'projection_x_coordinate'
y.units = 'km'
y.standard_name = 'projection_y_coordinate'
time.units = 'seconds since 2000-01-01 00:00:00 UTC'
time.standard_name = 'reference time of product'
status_flag.units = 'quality control indicator'
status_flag.standard_name = 'nominal_quality low_quality rejected_by_filter no_input_data'
latitude.units = 'degrees_north'
latitude.standard_name = 'latitude'
longitude.units = 'degrees_east'
longitude.standard_name = 'longitude'
uice.units = 'sea_ice_x_displacement (m)'
uice.standard_name = 'mean component of the displacement along the x axis of the grid'
vice.units = 'sea_ice_y_displacement (m)'
vice.standard_name = 'mean component of the displacement along the y axis of the grid'
dXY_count.units = 'number of samples per value'
dXY_count.standard_name = 'number of samples per value'
time_first.units = 'seconds since 2000-01-01 00:00:00 UTC'
time_first.standard_name = 'time of earliest observed sample'
time_last.units = 'seconds since 2000-01-01 00:00:00 UTC'
time_last.standard_name = 'time of latest observed sample'
drift_magnitude.units = 'm'
drift_magnitude.standard_name = 'drift_magnitude'
drift_initial_bearing.units = 'degree (compass)'
drift_initial_bearing.standard_name = 'drift_initial_bearing'
#
x[:] = x_T4grid * 0.001
y[:] = y_T4grid * 0.001
time[:] = time_SAR
status_flag[:] = status_flag_SAR
longitude[:,:] = lon_T4grid
latitude[:,:] = lat_T4grid
uice[:,:] = xice_T4
vice[:,:] = yice_T4
dXY_count[:,:] = dXY_count_reg
time_first[:,:] = time_first_reg 
time_last[:,:] = time_last_reg
drift_magnitude[:,:] = Dist_ice_T4 
drift_initial_bearing[:,:] = initial_bearing_ice_T4
#
output_netcdf.description = '+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere'
output_netcdf.close()
#################################################
EOF

python3 "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/SAR_magnitude_direction_PA_""$SGE_TASK_ID"".py"
