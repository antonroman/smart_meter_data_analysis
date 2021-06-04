"""
    Extract the timeline (TL) values from the meters JSON files, and store them in a CSV file.
    For each JSON file, extract the timeline.tl field, and store it in a new CSV file with the name of the meter ID.
"""

import json
import pandas as pd
import os

json_files_path = 'json_contadores'
csv_output_path = 'tl_values'

print("Extracting TL values from meters...")

# Create, if not exists, the output folder
os.mkdir(csv_output_path)

# For each JSON file, get the reactive data and store it in two CSV files
file_list = list(filter(lambda x: x.endswith(".json"), os.listdir(json_files_path)))
total_number_of_files = len(file_list)
for index, filename in enumerate(file_list):
    # Print the progress
    print("Progress: %.2f%%" % (100 * (index / total_number_of_files)), end="\r", flush=True)

    try:
        # Read the JSON file and extract the meter ID and the s05 and s02 data
        data = json.load(open(json_files_path + '/' + filename, 'r'))
        meter_id = data['id']
        df_tl = pd.DataFrame(data['timeline']['tl'])

        # Write the data into two CSV files
        df_tl.to_csv(f"{csv_output_path}/{meter_id}_tl.csv", index=False)

    except Exception:
        print(f"Something went wrong with file {filename}")

print("Finished")
