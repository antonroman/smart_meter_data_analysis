"""
    Aggregate S02 and S05 CSV files in 10%, 20% ... 100%.
    It needs a CSV with all the meters which have data for all 2020 (see get_overlapping_dates_meters.py).

"""
import pandas as pd
from random import shuffle
from datetime import datetime as dt
import os

valid_files_csv_filename = 'valid_meters_2020.csv'
reactive_values_files_folder = 'reactive_values'
aggregated_files_output_folder = 'aggregated_reactive'

# Time window
from_day = '2020-01-01'
to_day = '2020-12-31'

def aggregate_files(file_type):
    with open(valid_files_csv_filename, 'r') as f:
        # Read the file list form the valid meters list
        file_list = list(map(lambda x: x.strip(), f.readlines()))
        file_list = list(filter(lambda x: file_type in x, file_list))

    shuffle(file_list)  # the aggregation has to be randomized

    aggregation_levels = [x / 10 for x in range(1, 11)]
    for agg_level in aggregation_levels:
        print("Aggregating %i%% of the files..." % (100 * agg_level))

        # For each aggregation level, select the corresponding files
        agg_files_list = file_list[0:int(agg_level * len(file_list) - 1)]
        print(len(agg_files_list))

        # Name of the output aggregated file
        agg_filename = f'agg_values_{file_type}_{str(int(100 * agg_level))}.csv'

        # Read each CSV file for the current aggregation level and add its contents to the output aggregation CSV file
        for index, csv_file in enumerate(agg_files_list):
            print("Progress: %.2f%%" % (100 * (index / len(agg_files_list))), end="\r", flush=True)

            # Read the CSV file as a DataFrame and parse the timestamp
            df = pd.read_csv(f'{reactive_values_files_folder}/{csv_file}', index_col='timestamp')
            df.index = pd.to_datetime(df.index)
            df = df.sort_values(by='timestamp')

            # Get the data for the desired time window
            df = df.loc[from_day:to_day] 

            print_header = not(bool(index))  # the CSV header should be printed only the first time
            df.to_csv(f'{aggregated_files_output_folder}/{agg_filename}', header=print_header, index=True, mode='a')


os.makedirs(aggregated_files_output_folder, exist_ok=True)

print("Aggregating S02 data...\n")
aggregate_files('S02')

print("\n\nAggregating S05 data...\n")
aggregate_files('S05')
