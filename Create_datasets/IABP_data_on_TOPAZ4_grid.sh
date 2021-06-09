#$ -S /bin/bash
#$ -l h_rt=00:05:00
#$ -q research-el7.q
#$ -l h_vmem=1G
#$ -t 1-37
#$ -o /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/OUT/SAR_OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/ERR/SAR_ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -cwd

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python/3.7.2

cat > "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/IABP_buoys_gridding_""$SGE_TASK_ID"".py" << EOF
#################################################
import os
import numpy as np
import pandas as pd
import math
import cmath
from datetime import datetime, timedelta
from netCDF4 import Dataset
from pyproj import Proj, transform
from scipy.interpolate import griddata
from scipy.spatial.distance import *
import sys
function_path='/lustre/storeB/users/cyrilp/Python_functions/'
sys.path.insert(0, function_path)
from position_sea_ice_edge_including_coastlines import *
#####################################
# Constants
#####################################
path_buoys = '/lustre/storeB/project/copernicus/svalnav/Data/IABP/Data/'
path_osisaf = '/lustre/storeB/project/copernicus/svalnav/Data_operational_ice_drift_forecasts/Pan_Arctic/OSISAF_SIC_on_T4grid/'
path_output = '/lustre/storeB/project/copernicus/svalnav/Data/IABP/Data_T4grid_SIC10/'
#
date_min = '20210501'
date_max = '20210606'
#
proj_EPSG4326 = Proj(init = 'epsg:4326')
proj_T4 = Proj('+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere')
#
x_T4grid = np.linspace(-38, 38, 609) * 100 * 1000
y_T4grid = np.linspace(-55, 55, 881) * 100 * 1000
xx_T4grid, yy_T4grid = np.meshgrid(x_T4grid, y_T4grid)
lon_T4grid, lat_T4grid = transform(proj_T4, proj_EPSG4326, xx_T4grid, yy_T4grid)
#
threshold_ice_egde = 10 # sea ice concentration (%)
threshold_magnitude = 0
threshold_disticeedge = 0
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
#####################################
# Dates
#####################################
start_date = []
end_date = []
start_datetime = datetime.strptime(str(date_min), '%Y%m%d')
while start_datetime <= datetime.strptime(str(date_max), '%Y%m%d'):
	start_date.append(start_datetime.strftime('%Y%m%d'))
	start_datetime = start_datetime + timedelta(days = 1)
	end_date.append(start_datetime.strftime('%Y%m%d'))
#
start_date_task = start_date[$SGE_TASK_ID - 1]
end_date_task = end_date[$SGE_TASK_ID - 1]
#####################################
# Masks
#####################################
file_osisaf = path_osisaf + 'OSISAF_SIC_distance_ice_edge_' + start_date_task + '.nc'
nc_osisaf = Dataset(file_osisaf, 'r')
SIC_osisaf = nc_osisaf.variables['SIC'][:,:]
#
xx_T4grid_flat = np.ndarray.flatten(xx_T4grid)
yy_T4grid_flat = np.ndarray.flatten(yy_T4grid)
SIE_osisaf = np.zeros(np.shape(SIC_osisaf))
SIE_osisaf[SIC_osisaf > threshold_ice_egde] = 1
SIE_osisaf_flat = np.ndarray.flatten(SIE_osisaf)
ice_edge_bool = np.ndarray.flatten(position_sea_ice_edge_including_coastlines(SIE_osisaf))
ice_edge_osisaf = np.zeros(np.shape(ice_edge_bool))
ice_edge_osisaf[ice_edge_bool == True] = 1
coord_ice_edge_osisaf = np.array([xx_T4grid_flat[ice_edge_osisaf == 1], yy_T4grid_flat[ice_edge_osisaf == 1]]).T  #  ice edge including coastlines
#####################################
# Start coordinates
#####################################
BuoyID = {}
BuoyID_T4pos = {}
class IDclass:
	pass
#
file_start_date = path_buoys + start_date_task[0:4] + '/' + start_date_task[4:6] + '/' + 'Buoys_' + start_date_task + '.dat'
df_start_date = pd.read_csv(file_start_date, delimiter = '\t')
#
BuoyID_start_date = np.unique(df_start_date['BuoyID'])
for idb in range(0, len(BuoyID_start_date)):
	BuoyID_str = str(int(BuoyID_start_date[idb]))
	idx_idb = df_start_date['BuoyID'] == BuoyID_start_date[idb]
	df_start_idb = df_start_date[idx_idb]
	hour_start_idb = df_start_idb['Hour']
	min_start_idb = df_start_idb['Min']
	all_pos_midnight = np.where(np.logical_and(hour_start_idb == 0, min_start_idb == 0) == True)
	try:
		pos_midnight = np.min(all_pos_midnight)
		latitude_start = np.array(df_start_idb['Lat'])[pos_midnight]
		longitude_start = np.array(df_start_idb['Lon'])[pos_midnight]
		#
		if (latitude_start > -990 and longitude_start > -990):
			BID = IDclass()
			BID.name = BuoyID_str
			BID.DOY_start = np.array(df_start_idb['DOY'])[pos_midnight]
			BID.latitude_start = latitude_start
			BID.longitude_start = longitude_start
			BID.x_start, BID.y_start = transform(proj_EPSG4326, proj_T4, BID.longitude_start, BID.latitude_start)
			BID.latitude_traj = np.array(df_start_idb['Lat'])
			BID.longitude_traj = np.array(df_start_idb['Lon'])
			#
			BID.latitude_end = np.nan
			BID.longitude_end = np.nan
			BID.x_end = np.nan
			BID.y_end = np.nan
			BID.dx = np.nan
			BID.dy = np.nan
			BID.magnitude = np.nan
			BID.initial_bearing = np.nan
			BID.distance_grid_point = np.nan
			#
			BuoyID[BuoyID_str] = BID
	except:
		pass
#####################################
# End coordinates
#####################################
file_end_date = path_buoys + end_date_task[0:4] + '/' + end_date_task[4:6] + '/' + 'Buoys_' + end_date_task + '.dat'
df_end_date = pd.read_csv(file_end_date, delimiter = '\t')
#
for idb in BuoyID:
	idx_idb = df_end_date['BuoyID'] == int(idb)
	df_end_idb = df_end_date[idx_idb]
	hour_end_idb = df_end_idb['Hour']
	min_end_idb = df_end_idb['Min']
	all_pos_midnight = np.where(np.logical_and(hour_end_idb == 0, min_end_idb == 0) == True)
	try:
		pos_midnight = np.min(all_pos_midnight)
		latitude_end = np.array(df_end_idb['Lat'])[pos_midnight]
		longitude_end = np.array(df_end_idb['Lon'])[pos_midnight]
		#
		if (latitude_end > -990 and longitude_end > -990):
			BID = BuoyID.get(idb)
			BID.DOY_end = np.array(df_end_idb['DOY'])[pos_midnight]
			BID.latitude_end = latitude_end
			BID.longitude_end = longitude_end
			BID.x_end, BID.y_end = transform(proj_EPSG4326, proj_T4, BID.longitude_end, BID.latitude_end)
			#
			BID.latitude_traj = np.hstack((BID.latitude_traj, latitude_end))
			BID.longitude_traj = np.hstack((BID.longitude_traj, longitude_end))
			buoy_xtraj, buoy_ytraj = transform(proj_EPSG4326, proj_T4, BID.longitude_traj, BID.latitude_traj)
			#
			buoy_traj = np.array([buoy_xtraj, buoy_ytraj]).T
			D_mat = cdist(buoy_traj, coord_ice_edge_osisaf, metric = 'euclidean')
			Dist_ice_edge = np.nanmin(D_mat, axis = 1)
			BID.dist_ice_edge = np.min(Dist_ice_edge)
			#####################################
			# Mask drift trajectory
			#####################################
			min_sic = 100
			for i in range(0, len(buoy_xtraj)):
				x_dist = np.abs(buoy_xtraj[i] - x_T4grid)
				y_dist = np.abs(buoy_ytraj[i] - y_T4grid)
				x_mindist = np.min(x_dist)
				y_mindist = np.min(y_dist)
				#       
				if (x_mindist <= 6250 and y_mindist <= 6250):   # 12.5 km / 2
					x_pos = np.where(x_dist == x_mindist)[0][0]
					y_pos = np.where(y_dist == y_mindist)[0][0]
					sic_pos = SIC_osisaf[y_pos, x_pos]
					if sic_pos < min_sic:
						min_sic = sic_pos
				else:
					min_sic = 0
			#
			if min_sic >= threshold_ice_egde:
				#####################################
				# Vector
				#####################################
				BID.dx = BID.x_end - BID.x_start
				BID.dy = BID.y_end - BID.y_start
				BID.magnitude = great_circle_distance(BID.longitude_start, BID.latitude_start, BID.longitude_end, BID.latitude_end)
				BID.initial_bearing = calculate_initial_compass_bearing((BID.latitude_start, BID.longitude_start), (BID.latitude_end, BID.longitude_end))
				#
				if np.logical_or(BID.magnitude <= threshold_magnitude, BID.dist_ice_edge < threshold_disticeedge):
					BID.latitude_end = np.nan
					BID.longitude_end = np.nan
					BID.x_end = np.nan
					BID.y_end = np.nan
					BID.dx = np.nan
					BID.dy = np.nan
					BID.magnitude = np.nan
					BID.initial_bearing = np.nan
				#####################################
				# Position of vector on TOPAZ4 grid
				#####################################
				else:
					x_distance = np.abs(BID.x_start - x_T4grid)
					y_distance = np.abs(BID.y_start - y_T4grid)
					x_mindistance = np.min(x_distance)
					y_mindistance = np.min(y_distance)
					# 
					if (x_mindistance <= 6250 and y_mindistance <= 6250):	# 12.5 km / 2 
						x_pos = np.where(x_distance == x_mindistance)[0][0]
						y_pos = np.where(y_distance == y_mindistance)[0][0]
						BID.distance_grid_point = great_circle_distance(BID.longitude_start, BID.latitude_start, lon_T4grid[y_pos, x_pos], lat_T4grid[y_pos, x_pos])
						#
						pos_yx = str(y_pos) + '_' + str(x_pos)
						if pos_yx in BuoyID_T4pos:
							current_closest_distance = BuoyID_T4pos.get(pos_yx).distance_grid_point
							if BID.distance_grid_point < current_closest_distance:
								BuoyID_T4pos[pos_yx] = BID
						else:
							BuoyID_T4pos[pos_yx] = BID
	except:
		pass
#####################################
# Gridding
#####################################
Variables_gridded = {}
for attr, value in BID.__dict__.items():
	if (attr != 'latitude_traj' and attr != 'longitude_traj'):
		Variables_gridded[attr] = np.full(np.shape(lon_T4grid), np.nan)
#
for pos_yx in BuoyID_T4pos:
	y_pos = int(pos_yx[0 : pos_yx.find('_')])
	x_pos = int(pos_yx[pos_yx.find('_') + 1 : len(pos_yx)])
	BID = BuoyID_T4pos.get(pos_yx)
	#
	for attr, value in BID.__dict__.items():
		if (attr != 'latitude_traj' and attr != 'longitude_traj'):
			if SIC_osisaf[y_pos, x_pos] >= threshold_ice_egde:
				Variables_gridded[attr][y_pos, x_pos] = value
#####################################
# NetCDF file
#####################################
output_file = path_output + 'Buoys_' + start_date_task + '.nc'
output_netcdf = Dataset(output_file, 'w', format='NETCDF4')
#########
x = output_netcdf.createDimension('x', len(x_T4grid))
y = output_netcdf.createDimension('y', len(y_T4grid))
#########
x = output_netcdf.createVariable('x', 'd', ('x'))
y = output_netcdf.createVariable('y', 'd', ('y'))
latitude = output_netcdf.createVariable('latitude', 'd', ('y','x'))
longitude = output_netcdf.createVariable('longitude', 'd', ('y','x'))
#
x_start = output_netcdf.createVariable('x_start', 'd', ('y','x'))
y_start = output_netcdf.createVariable('y_start', 'd', ('y','x'))
latitude_start = output_netcdf.createVariable('latitude_start', 'd', ('y','x'))
longitude_start = output_netcdf.createVariable('longitude_start', 'd', ('y','x'))
DOY_start = output_netcdf.createVariable('DOY_start', 'd', ('y','x'))
#
x_end = output_netcdf.createVariable('x_end', 'd', ('y','x'))
y_end = output_netcdf.createVariable('y_end', 'd', ('y','x'))
latitude_end = output_netcdf.createVariable('latitude_end', 'd', ('y','x'))
longitude_end = output_netcdf.createVariable('longitude_end', 'd', ('y','x'))
DOY_end = output_netcdf.createVariable('DOY_end', 'd', ('y','x'))
#
Buoy_ID = output_netcdf.createVariable('Buoy_ID', 'd', ('y', 'x'))
distance_grid_point = output_netcdf.createVariable('distance_grid_point', 'd', ('y', 'x'))
drift_magnitude = output_netcdf.createVariable('drift_magnitude', 'd', ('y', 'x'))
drift_initial_bearing = output_netcdf.createVariable('drift_initial_bearing', 'd', ('y', 'x'))
uice = output_netcdf.createVariable('uice', 'd', ('y', 'x'))
vice = output_netcdf.createVariable('vice', 'd', ('y', 'x'))
#
buoy_min_dist_ice_edge = output_netcdf.createVariable('buoy_min_dist_ice_edge', 'd', ('y', 'x'))
#########
x_start.units = '100 km'
x_start.standard_name = 'projection_x_coordinate'
y_start.units = '100 km'
y_start.standard_name = 'projection_y_coordinate'
latitude_start.units = 'degrees_north'
latitude_start.standard_name = 'latitude_start'
longitude_start.units = 'degrees_east'
longitude_start.standard_name = 'longitude_start'
DOY_start.units = 'day of the year'
DOY_start.standard_name = 'day of the year'
#
x_end.units = '100 km'
x_end.standard_name = 'projection_x_coordinate'
y_end.units = '100 km'
y_end.standard_name = 'projection_y_coordinate'
latitude_end.units = 'degrees_north'
latitude_end.standard_name = 'latitude_end'
longitude_end.units = 'degrees_east'
longitude_end.standard_name = 'longitude_end'
DOY_end.units = 'day of the year'
DOY_end.standard_name = 'day of the year'
#
Buoy_ID.standard_name = 'Buoy_ID'
distance_grid_point.standard_name = 'distance_grid_point'
distance_grid_point.units = 'm'
drift_magnitude.units = 'm/day'
drift_magnitude.standard_name = 'drift_magnitude'
drift_initial_bearing.units = 'degree (compass)'
drift_initial_bearing.standard_name = 'drift_initial_bearing'
uice.units = 'm/s'
uice.standard_name = 'sea_ice_x_velocity'
vice.units = 'm/s'
vice.standard_name = 'sea_ice_y_velocity'
#
buoy_min_dist_ice_edge.units = 'm'
buoy_min_dist_ice_edge.standard_name = 'minimum distance to the ice edge (including coastlines) along the buoy trajectory'
##########
x[:] = x_T4grid * 0.001 * 0.01
y[:] = y_T4grid * 0.001 * 0.01
longitude[:,:] = lon_T4grid
latitude[:,:] = lat_T4grid
#
x_start[:,:] = Variables_gridded['x_start'] * 0.001 * 0.01
y_start[:,:] = Variables_gridded['y_start'] * 0.001 * 0.01
latitude_start[:,:] = Variables_gridded['latitude_start']
longitude_start[:,:] = Variables_gridded['longitude_start']
DOY_start[:,:] = Variables_gridded['DOY_start']
#
x_end[:,:] = Variables_gridded['x_end'] * 0.001 * 0.01
y_end[:,:] = Variables_gridded['y_end'] * 0.001 * 0.01
latitude_end[:,:] = Variables_gridded['latitude_end']
longitude_end[:,:] = Variables_gridded['longitude_end']
DOY_end[:,:] = Variables_gridded['DOY_end']
#
Buoy_ID[:,:] = Variables_gridded['name']
distance_grid_point[:,:] = Variables_gridded['distance_grid_point']
drift_magnitude[:,:] = Variables_gridded['magnitude']
drift_initial_bearing[:,:] = Variables_gridded['initial_bearing']
uice[:,:] = Variables_gridded['dx'] / (24 * 3600)
vice[:,:] = Variables_gridded['dy'] / (24 * 3600)
#
buoy_min_dist_ice_edge[:,:] = Variables_gridded['dist_ice_edge']
#########
output_netcdf.description = '+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere'
output_netcdf.close()
#################################################
EOF

python3 "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/IABP_buoys_gridding_""$SGE_TASK_ID"".py"

