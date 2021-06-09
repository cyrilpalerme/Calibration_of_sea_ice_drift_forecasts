#$ -S /bin/bash
#$ -l h_rt=00:01:00
#$ -q research-el7.q
#$ -l h_vmem=1G
#$ -t 1-38
#$ -o /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/OUT/SAR_OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/ERR/SAR_ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -cwd

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module load Python/3.7.2

cat > "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/Download_IABP_""$SGE_TASK_ID"".py" << EOF
#################################################
import urllib.request
import os
import csv
import ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
	ssl._create_default_https_context = ssl._create_unverified_context
#
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
#####################################
# Constants
#####################################
#url_data = 'http://iabp.apl.washington.edu/Data_Products/Daily_Full_Res_Data/Arctic/'
#url_data = 'https://iabp.apl.uw.edu/Data_Products/Daily_Full_Res_Data/Arctic/'
url_data = 'http://35.184.124.35/Data_Products/Daily_Full_Res_Data/Arctic/'
path_output = '/lustre/storeB/project/copernicus/svalnav/Data/IABP/Data/'
#
date_min = '20210501'
date_max = '20210607'
#
extracted_variables = ['BuoyID', 'Year', 'Hour', 'Min', 'DOY', 'Lat', 'Lon']
string_legend = (', '.join(extracted_variables)).replace(', ', '\t')
#####################################
# Dates
#####################################
file_date = []
file_datetime = datetime.strptime(str(date_min), '%Y%m%d')
while file_datetime <= datetime.strptime(str(date_max), '%Y%m%d'):
	file_date.append(file_datetime.strftime('%Y%m%d'))
	file_datetime = file_datetime + timedelta(days = 1)
print(len(file_date))
#####################################
# Load data
#####################################
file_data = urllib.request.urlopen(url_data + 'FR_' + file_date[$SGE_TASK_ID - 1] + '.dat')
df = pd.read_csv(file_data, delimiter = ';')
#
BuoyID = df['BuoyID']
All_buoys_ID = np.unique(BuoyID)
for id in range(0, len(All_buoys_ID)):
	idx_id = BuoyID == All_buoys_ID[id]
	df_id = df[idx_id == True]
	#
	for v in range(0, len(extracted_variables)):
		var = extracted_variables[v]
		var_data = np.expand_dims(df_id[var], axis = 1)
		#
		if v == 0:
			Data_id = var_data
		else:
			Data_id = np.concatenate((Data_id, var_data), axis = 1)
	#
	if id == 0:
		Data_all = Data_id
	else:
		Data_all = np.concatenate((Data_all, Data_id), axis = 0) 
#############################
# Save data
#############################
file_output = path_output + 'Buoys_' + file_date[$SGE_TASK_ID - 1] + '.dat'
with open(file_output, 'a') as f:
	writer = csv.writer(f, delimiter = '\t')
	f.write(string_legend + '\n')
f.close()
#
with open(file_output, 'a') as f:
	writer = csv.writer(f, delimiter = '\t')
	writer.writerows(Data_all)
f.close()
#############################
EOF

python3 "/lustre/storeB/project/copernicus/svalnav/Data/data_processing_files/PROG/Download_IABP_""$SGE_TASK_ID"".py"
