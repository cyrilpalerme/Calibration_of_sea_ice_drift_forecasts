#$ -S /bin/bash
#$ -l h_rt=00:05:00
#$ -q ded-parallelx.q
#$ -l h_vmem=12G
#$ -t 31-31
#$ -o /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -cwd

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

cat > "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/Interp_OSISAF_""$SGE_TASK_ID"".py" << EOF

################################################
import matplotlib
matplotlib.use('Agg')
import os
from matplotlib.colors import BoundaryNorm
from netCDF4 import Dataset
import numpy as np
import glob
import sys
import itertools
from itertools import chain
from scipy.interpolate import griddata
from pyproj import Proj, transform
from scipy.spatial.distance import *
function_path='/lustre/storeB/users/cyrilp/Python_functions/'
sys.path.insert(0, function_path)
from position_sea_ice_edge import *
################################################
path_input = '/lustre/storeB/project/copernicus/osisaf/data/reprocessed/ice/conc-cont-reproc/v2p0/'
#path_output = '/lustre/storeB/project/nwp/SALIENSEAS/Ice_drift_calibration_Pan_Arctic/Data/OSISAF_SIC_on_T4grid/'
path_output = '/lustre/storeB/project/copernicus/svalnav/Data_operational_ice_drift_forecasts/Pan_Arctic/OSISAF_SIC_on_T4grid/'
SIE_threshold = 15
#
proj_EPSG4326 = Proj(init = 'epsg:4326')
proj_OSIsic = Proj('+proj=laea +lon_0=0 +datum=WGS84 +ellps=WGS84 +lat_0=90.0')
proj_T4 = Proj('+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere')
#
x_T4grid = np.linspace(-38, 38, 609) * 100 * 1000
y_T4grid = np.linspace(-55, 55, 881) * 100 * 1000
xx_T4grid, yy_T4grid = np.meshgrid(x_T4grid, y_T4grid)
lon_T4grid, lat_T4grid = transform(proj_T4, proj_EPSG4326, xx_T4grid, yy_T4grid) 
################################################
# Dataset list
################################################
paths = []
for year in range(2020, 2021):
        yearstr = str(year)
        for month in range(5, 6):
		monthstr = "{:02d}".format(month)
                p = path_input + yearstr + '/' + monthstr + '/'
                paths.append(p)
#
dataset = []
for path, dirs, files in chain.from_iterable(os.walk(path) for path in paths):
        nh_files = [s for s in files if "ice_conc_nh_ease2-250_icdr-v2p0_" in s]
        sorted_files=sorted(nh_files)
        for i in range(0,len(sorted_files)):
                dataset.append(sorted(glob.glob(path+sorted_files[i])))
print('len(dataset)', len(dataset))
################################################
# Loading data
################################################
nc_OSIsic = Dataset(dataset[$SGE_TASK_ID - 1][0],'r')
filename = os.path.basename(dataset[$SGE_TASK_ID - 1][0])
#
x_OSIsic = nc_OSIsic.variables['xc'][:] * 1000
y_OSIsic = nc_OSIsic.variables['yc'][:] * 1000
xx_OSIsic, yy_OSIsic = np.meshgrid(x_OSIsic, y_OSIsic)
#
lat_OSIsic = nc_OSIsic.variables['lat'][:,:]
lon_OSIsic = nc_OSIsic.variables['lon'][:,:]
SIC_OSIsic = nc_OSIsic.variables['ice_conc'][0,:,:]
status_flag_OSIsic = nc_OSIsic.variables['status_flag'][0,:,:]
################################################
# Interpolation on TOPAZ4 grid 
################################################
xx_OSIsic_reg, yy_OSIsic_reg = transform(proj_OSIsic, proj_T4, xx_OSIsic, yy_OSIsic)
xx_OSIsic_reg_flat = np.ndarray.flatten(xx_OSIsic_reg)
yy_OSIsic_reg_flat = np.ndarray.flatten(yy_OSIsic_reg)
SIC_OSIsic_flat = np.ndarray.flatten(SIC_OSIsic)
status_flag_OSIsic_flat = np.ndarray.flatten(status_flag_OSIsic)
#
SIC_T4 = griddata((xx_OSIsic_reg_flat, yy_OSIsic_reg_flat), SIC_OSIsic_flat, (x_T4grid[None,:], y_T4grid[:,None]), method = 'nearest')
status_flag_T4 = griddata((xx_OSIsic_reg_flat, yy_OSIsic_reg_flat), status_flag_OSIsic_flat, (x_T4grid[None,:], y_T4grid[:,None]), method = 'nearest')
#################################################
# Sea land mask and position of the ice edge 
#################################################
sealandmask = np.zeros(np.shape(status_flag_T4))
idx_status_flag = np.logical_and(status_flag_T4 != 1, status_flag_T4 != 2)
sealandmask[idx_status_flag==True] = 1
#
xx_T4_flat = np.ndarray.flatten(xx_T4grid)
yy_T4_flat = np.ndarray.flatten(yy_T4grid)
#
SIC_T4[SIC_T4 < -32700] = np.nan
SIE = np.zeros(np.shape(SIC_T4))
SIE[SIC_T4 >= SIE_threshold] = 1
sie_position = position_sea_ice_edge(SIE, sealandmask)
sie_position_flat = np.ndarray.flatten(sie_position)
#
Coord_T4 = np.array([xx_T4_flat, yy_T4_flat]).T
Coord_sie = np.array([xx_T4_flat[sie_position_flat == 1], yy_T4_flat[sie_position_flat == 1]]).T
#################################################
## Assessing distance to the ice edge
#################################################
D_mat = cdist(Coord_T4, Coord_sie, metric='euclidean')
Dist_ice_edge = np.nanmin(D_mat, axis = 1)
Dist_ice_edge_mat = Dist_ice_edge.reshape(np.shape(SIC_T4))
Index_SIE = np.zeros(np.shape(SIE))
Index_SIE[SIE == 0] = -1
Index_SIE[SIE == 1] = 1
Dist_ice_edge_mat_index = Dist_ice_edge_mat * Index_SIE
#################################################
## Land-sea mask
#################################################
SIC_T4[sealandmask == 0] = np.nan
Dist_ice_edge_mat_index[sealandmask == 0] = np.nan
#################################################
## Output NetCDF file
#################################################
output_file = filename.replace('ice_conc_nh_ease2-250_icdr-v2p0_', 'OSISAF_SIC_distance_ice_edge_')
output_file = output_file.replace('1200.nc', '.nc')
output_netcdf = Dataset(path_output+output_file,'w',format='NETCDF4')
##
x = output_netcdf.createDimension('x',len(x_T4grid))
y = output_netcdf.createDimension('y',len(y_T4grid))
##
xc = output_netcdf.createVariable('xc', 'd', ('x'))
yc = output_netcdf.createVariable('yc', 'd', ('y'))
lat = output_netcdf.createVariable('lat', 'd', ('y','x'))
lon = output_netcdf.createVariable('lon', 'd', ('y','x'))
SIC = output_netcdf.createVariable('SIC', 'd', ('y','x'))
Disticeedge = output_netcdf.createVariable('Disticeedge', 'd', ('y','x'))
status_flag = output_netcdf.createVariable('status_flag', 'd', ('y','x'))
##
xc.units = 'm'
yc.units = 'm'
lat.units='degree'
lon.units='degree'
SIC.units = 'Sea ice concentration (%)'
Disticeedge.units = 'Distance to the ice edge in m (positive if SIC >= 15 %, negative if SIC < 15 %)'
status_flag.units = 'flag (see description in original OSI-SAF files)'
status_flag.long_name = 'status flag bit array for sea ice concentration retrieval'
##
xc[:] = x_T4grid
yc[:] = y_T4grid
lat[:,:] = lat_T4grid
lon[:,:] = lon_T4grid
SIC[:,:] = SIC_T4
Disticeedge[:,:] = Dist_ice_edge_mat_index
status_flag[:,:] = status_flag_T4
##
output_netcdf.description = '+proj=stere +lon_0=-45. +lat_ts=90. +lat_0=90. +a=6378273. +b=6378273. +ellps=sphere'
output_netcdf.close()
#################################################
EOF

python "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/Interp_OSISAF_""$SGE_TASK_ID"".py"
