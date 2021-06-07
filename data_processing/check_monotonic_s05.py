"""
    Script to detect if there is a S05 CSV file with non-monotonic data, that means, that there are rows with lower
    values than the previous one.
"""

import pandas as pd
import os

reactive_values_files_folder = 'reactive_values'
non_monotonic_file_output = 'non_monotonic_values.csv'

file_list = list(filter(lambda x: x.endswith("S05.csv"), os.listdir(reactive_values_files_folder)))
for index, csv_file in enumerate(file_list):
    print("Progress: %.2f%%" % (100 * (index / len(file_list))), end="\r", flush=True)

    # Read the CSV file as a DataFrame
    df = pd.read_csv(f'{reactive_values_files_folder}/{csv_file}')

    # Calculate the deltas between rows, and select those which are negative
    df_diff = df[['R1a', 'R2a', 'R3a', 'R4a']].diff()
    df_diff = pd.concat([df[['timestamp']], df_diff], axis=1)  # add the timestamp to the diff
    df_diff = df_diff[(df_diff['R1a'] < 0) | (df_diff['R2a'] < 0) | (df_diff['R3a'] < 0) | (df_diff['R4a'] < 0)]

    # Write the negatives deltas to the output CSV, together with the ID of the meter
    df_diff['meterId'] = csv_file.split('_')[2]
    df_diff.to_csv(non_monotonic_file_output, mode='a', header=not(bool(index)), index=False)
