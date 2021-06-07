"""
    Script to detect in which meters there are S02 records with the quality-byte (Bc) with a value greater than 0x80,
    that means, that the data is invalid.
"""

import pandas as pd
import json
import os


def is_invalid_record(row):
    """Detects if the data in this row is invalid from the value in the Bc field"""
    bc_value = row['Bc']
    return type(bc_value) != int or bc_value >= 80


json_files_path = 'json_contadores'
csv_output_filename = 'invalid_data.csv'

print("Looking for invalid data in the S02 records...")

# Read all the JSON files
file_list = list(filter(lambda x: x.endswith(".json"), os.listdir(json_files_path)))
total_number_of_files = len(file_list)
for index, filename in enumerate(file_list):
    # Print the progress
    print("Progress: %.2f%%" % (100 * (index / total_number_of_files)), end="\r", flush=True)

    # Read the JSON file and get those rows with the quality-byte equal or greater than 0x80
    json_file = json.load(open(json_files_path + '/' + filename, 'r'))
    meter_id = json_file['id']
    df = pd.DataFrame(json_file['timeline']['S02'])
    if len(df):
        df = df[['Fh', 'Bc', 'R1', 'R2', 'R3', 'R4']]  # select only the interest columns
        df = df.rename(columns={'Fh': 'timestamp'})  # rename the column "Fh" to "timestamp"
        df['Bc'] = df['Bc'].astype(str)
        df['Bc'] = df['Bc'].apply(lambda x: int(x, 16))  # convert the Bc field from hex to decimal
        df_invalid_records = df[df['Bc'] >= 128]  # filter out the valid rows
        df_invalid_records['meterId'] = meter_id

        # Write them into the CSV file
        df_invalid_records.to_csv(csv_output_filename, header=not(bool(index)), mode='a', index=False)
