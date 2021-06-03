"""
    Search for NA and null values in the reactive CSV files
"""

import pandas as pd
import os

reactive_values_files_folder = 'reactive_values'
na_values_file = 'na_values.txt'

print("Searching for NA values...")

# Read the CSV reactive files one by one
file_list = list(filter(lambda x: x.endswith('.csv'), os.listdir(reactive_values_files_folder)))
total_number_of_files = len(file_list)

for index, csv_file in enumerate(file_list):
    # Print the progress
    print("Progress: %.2f%%" %(100*(index/total_number_of_files)), end="\r", flush=True)

    # Open the file as a Pandas DataFrame
    df = pd.read_csv(f'{reactive_values_files_folder}/{csv_file}')

    # Get the null / NA values and print them to a file
    null_values = df[df.isnull().values.any(axis=1)]
    if len(null_values):
        print(f"\nNull data in {csv_file}:")
        with open(na_values_file, 'a') as output_file:
            output_file.write(f"\nNull data in {csv_file}:")
            output_file.write(null_values.to_dict('records'))

print("Finished")