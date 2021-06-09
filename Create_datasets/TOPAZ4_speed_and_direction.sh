#$ -S /bin/bash
#$ -l h_rt=00:30:00
#$ -q ded-parallelx.q
#$ -l h_vmem=8G
#$ -t 1-312
#$ -o /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -cwd

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

cat > "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/Convert_TOPAZ4_daily_""$SGE_TASK_ID"".py" << EOF

################################################
import matplotlib
matplotlib.use('Agg')
import os
import glob
import datetime
from pylab import *
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj, transform
import cmath
################################################
# Paths, files
################################################
path_input = '/lustre/storeB/project/copernicus/ARC-MFC/ARC-METNO-ARC-TOPAZ4_2_PHYS-FOR/arctic/mersea-class1/'
path_output = '/lustre/storeB/project/nwp/SALIENSEAS/Ice_drift_calibration_Pan_Arctic/Data/TOPAZ4/'
first_date = '20111006'
last_date = '20160505'
proj_EPSG4326 = Proj(init = 'epsg:4326')
proj_T4 = Proj('+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere')
SGE_TASK_ID = 100
################################################
# Functions
################################################
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
#################################################
# Dataset TOPAZ4
#################################################
start_date = datetime.datetime.strptime(first_date, '%Y%m%d')
end_date = datetime.datetime.strptime(last_date, '%Y%m%d')
date_vector = []
date_i = start_date
delta = datetime.timedelta(days = 7)
while date_i <= end_date:
	if date_i.weekday() == 3:
		date_vector.append(date_i.strftime('%Y%m%d'))
		date_i = date_i + delta
dataset_T4 = sorted(glob.glob(path_input + '*_dm-metno-MODEL-topaz4-ARC-b' + date_vector[$SGE_TASK_ID - 1] + '-fv02.0.nc'))
if len(dataset_T4) == 10:
	print('len(date_vector)', len(date_vector))
	#################################################
	# Load data
	#################################################
	for lt in range(0, len(dataset_T4)):
		file_T4 = dataset_T4[lt]
		filename_T4 = os.path.basename(file_T4)
		file_T4_start_date = filename_T4[36:44]
		file_T4_forec_date = filename_T4[0:8]
		nc_T4 = Dataset(file_T4, 'r')
		if lt == 0:
			x_T4 = nc_T4.variables['x'][:] * 100000
			y_T4 = nc_T4.variables['y'][:] * 100000
			lat_T4 = nc_T4.variables['latitude'][:,:]
			lon_T4 = nc_T4.variables['longitude'][:,:]
			time_T4 = nc_T4.variables['time'][:]
			temperature_T4 = nc_T4.variables['temperature'][:,0,:,:]
			fice_T4 = nc_T4.variables['fice'][:,:,:]
			hice_T4 = nc_T4.variables['hice'][:,:,:]
			hsnow_T4 = nc_T4.variables['hsnow'][:,:,:]
			ui_T4 = nc_T4.variables['uice'][:,:,:] * 24 * 3600 # unit m.s-1 => m.day-1
			vi_T4 = nc_T4.variables['vice'][:,:,:] * 24 * 3600 # unit m.s-1 => m.day-1
		else:
			time_T4 = np.hstack((time_T4, nc_T4.variables['time'][:]))
			temperature_T4 = ma.concatenate((temperature_T4, nc_T4.variables['temperature'][:,0,:,:]), axis = 0)
			fice_T4 = ma.concatenate((fice_T4, nc_T4.variables['fice'][:,:,:]), axis = 0)
			hice_T4 = ma.concatenate((hice_T4, nc_T4.variables['hice'][:,:,:]), axis = 0)
			hsnow_T4 = ma.concatenate((hsnow_T4, nc_T4.variables['hsnow'][:,:,:]), axis = 0)
			ui_T4 = ma.concatenate((ui_T4, nc_T4.variables['uice'][:,:,:] * 24 * 3600), axis = 0)
			vi_T4 = ma.concatenate((vi_T4, nc_T4.variables['vice'][:,:,:] * 24 * 3600), axis = 0)
	#################################################
	# Masks
	#################################################
	xx_T4, yy_T4 = np.meshgrid(x_T4, y_T4)
	#
	temperature_T4[temperature_T4.mask == True] = np.nan
	fice_T4[fice_T4.mask == True] = np.nan
	hice_T4[hice_T4.mask == True] = np.nan
	hsnow_T4[hsnow_T4.mask == True] = np.nan
	ui_T4[ui_T4.mask == True] = np.nan
	vi_T4[vi_T4.mask == True] = np.nan
	#################################################
	# Distance and initial bearing
	#################################################
	Dist_ice_T4 = np.full((len(time_T4), len(y_T4), len(x_T4)), np.nan)
	initial_bearing_ice_T4 = np.full((len(time_T4), len(y_T4), len(x_T4)), np.nan)
	#
	for ts in range(0, len(time_T4)):
		print(ts)
		xx_ice_end_T4 = xx_T4 + ui_T4[ts,:,:]
		yy_ice_end_T4 = yy_T4 + vi_T4[ts,:,:]
		lon_ice_end_T4, lat_ice_end_T4 = transform(proj_T4, proj_EPSG4326, xx_ice_end_T4, yy_ice_end_T4)
		Dist_ice_T4[ts,:,:] = great_circle_distance(lon_T4, lat_T4, lon_ice_end_T4, lat_ice_end_T4)
		#
		for i in range(0, np.shape(Dist_ice_T4)[1]):
			for j in range(0, np.shape(Dist_ice_T4)[2]):
				A_coord_T4 = (lat_T4[i,j], lon_T4[i,j])
				B_coord_ice_T4 = (lat_ice_end_T4[i,j], lon_ice_end_T4[i,j]) 
				initial_bearing_ice_T4[ts,i,j] = calculate_initial_compass_bearing(A_coord_T4, B_coord_ice_T4)
	#
	Dist_ice_T4[np.logical_or(ui_T4.mask == True, vi_T4.mask == True)] = np.nan
	Dist_ice_T4[(abs(ui_T4) + abs(vi_T4)) == 0] = np.nan
	Dist_ice_T4[Dist_ice_T4 == 6371000] = np.nan
	initial_bearing_ice_T4[np.logical_or(ui_T4.mask == True, vi_T4.mask == True)] = np.nan
	initial_bearing_ice_T4[(abs(ui_T4) + abs(vi_T4)) == 0] = np.nan
	initial_bearing_ice_T4[Dist_ice_T4 == 6371000] = np.nan
	#################################################
	# Output netCDF file
	#################################################
	output_filename = 'TOPAZ4_daily_' + file_T4_start_date + '.nc'
	output_netcdf = Dataset(path_output + output_filename, 'w', format='NETCDF4')
	#
	x = output_netcdf.createDimension('x', len(x_T4))
	y = output_netcdf.createDimension('y', len(y_T4))
	time = output_netcdf.createDimension('time', len(time_T4))
	#
	x = output_netcdf.createVariable('x', 'd', ('x'))
	y = output_netcdf.createVariable('y', 'd', ('y'))
	time = output_netcdf.createVariable('time', 'd', ('time'))
	latitude = output_netcdf.createVariable('latitude', 'd', ('y','x'))
	longitude = output_netcdf.createVariable('longitude', 'd', ('y','x'))
	temperature = output_netcdf.createVariable('temperature', 'd', ('time','y','x'))
	fice = output_netcdf.createVariable('fice', 'd', ('time','y','x'))
	hice = output_netcdf.createVariable('hice', 'd', ('time','y','x'))
	hsnow = output_netcdf.createVariable('hsnow', 'd', ('time','y','x'))
	drift_magnitude = output_netcdf.createVariable('drift_magnitude', 'd', ('time','y','x'))
	drift_initial_bearing = output_netcdf.createVariable('drift_initial_bearing', 'd', ('time','y','x'))
	uice = output_netcdf.createVariable('uice', 'd', ('time', 'y', 'x'))
	vice = output_netcdf.createVariable('vice', 'd', ('time', 'y', 'x'))
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
	temperature.units = 'Celsius'
	temperature.standard_name = 'sea_water_potential_temperature'
	fice.units = 'fraction'
	fice.standard_name = 'sea_ice_area_fraction'
	hice.units = 'm'
	hice.standard_name = 'sea_ice_thickness'
	hsnow.units = 'm'
	hsnow.standard_name = 'surface_snow_thickness'
	drift_magnitude.units = 'm/day'
	drift_magnitude.standard_name = 'drift_magnitude'
	drift_initial_bearing.units = 'degree (compass)'
	drift_initial_bearing.standard_name = 'drift_initial_bearing'
	uice.units = 'm/day'
	uice.standard_name = 'sea_ice_x_velocity'
	vice.units = 'm/day'
	vice.standard_name = 'sea_ice_y_velocity'
	#
	x[:] = x_T4 * 0.00001
	y[:] = y_T4 * 0.00001
	time[:] = time_T4
	longitude[:,:] = lon_T4
	latitude[:,:] = lat_T4
	temperature[:,:,:] = temperature_T4
	fice[:,:,:] = fice_T4
	hice[:,:,:] = hice_T4
	hsnow[:,:,:] = hsnow_T4
	drift_magnitude[:,:,:] = Dist_ice_T4
	drift_initial_bearing[:,:,:] = initial_bearing_ice_T4
	uice[:,:,:] = ui_T4 
	vice[:,:,:] = vi_T4
	#
	output_netcdf.description = 'Grid projection: +proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere'
	output_netcdf.close()
#################################################
EOF

python3 "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/Convert_TOPAZ4_daily_""$SGE_TASK_ID"".py"
