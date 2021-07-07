"""
    Search for outliers in 2020 in the S02 files.
    It needs a CSV with all the meters which have data for all 2020 (see get_overlapping_dates_meters.py).

"""
import pandas as pd
import os

valid_files_csv_filename = 'valid_meters_2020.csv'
reactive_values_files_folder = 'reactive_values'
output_folder = 'outliers'

outliers_threshold = 4

# Time window
from_day = '2020-01-01'
to_day = '2020-12-31'

print("Aggregating S02 data...\n")
csv_file_columns = ["timestamp", "R1", "R2", "R3", "R4"]

with open(valid_files_csv_filename, 'r') as f:
    # Read the file list form the valid meters list
    file_list = list(map(lambda x: x.strip(), f.readlines()))
    file_list = list(filter(lambda x: 'S02' in x, file_list))

    # The DataFrame where the meters data will be stored
    df = pd.DataFrame({}, columns=csv_file_columns.append('meter_id'))

    # Read each CSV file for the current aggregation level and add its contents to the output aggregation CSV file
    for index, csv_file in enumerate(file_list):
        print("Progress: %.2f%%" % (100 * (index / len(file_list))), end="\r", flush=True)
        
        # Read the CSV file as a DataFrame and parse the timestamp
        df_new = pd.read_csv(f'{reactive_values_files_folder}/{csv_file}', index_col='timestamp')
        df_new.index = pd.to_datetime(df_new.index)
        df_new = df_new.sort_values(by='timestamp')     
        df_new['meter_id'] = csv_file.split('_')[-2]     

        # Get the data for the desired time window
        df_new = df_new.loc[from_day:to_day]   

        # Add the contents of this file to the others
        df = pd.concat([df, df_new])

    # Calculate the outlier threshold for each date and meter ID
    upper_limit = df.groupby('timestamp').median() + outliers_threshold*df.groupby('timestamp').std()
    upper_limit = upper_limit.rename(columns={
        'R1a': 'R1_limit', 'R2a': 'R2_limit', 'R3a': 'R3_limit', 'R4a': 'R4_limit',
        'R1': 'R1_limit', 'R2': 'R2_limit', 'R3': 'R3_limit', 'R4': 'R4_limit',
    })

    df_merged = pd.merge(df, upper_limit, how='left', on='timestamp')

    # Print the number of meters with outliers in R1, and the number of outliers for each of those meters
    meters_with_outliers = df_merged[(df_merged[csv_file_columns[1]] >= df_merged['R1_limit'])]
    meters_with_outliers.to_csv(f'{output_folder}/outliers_S02.csv')
    print("\nMeters with outliers: " + meters_with_outliers['meter_id'].shape[0])


os.makedirs(output_folder, exist_ok=True)

