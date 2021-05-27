import json
import pandas as pd
import os

json_files_path = 'json_contadores'
csv_output_path = 'reactive_values'

print("Extracting reactive values from meters...")

# Create, if not exists, the output folder
os.mkdir(csv_output_path)

# For each JSON file, get the reactive data and store it in two CSV files
file_list = list(filter(lambda x: x.endswith(".json"), os.listdir(json_files_path)))
total_number_of_files = len(file_list)
for index, filename in enumerate(file_list):
    # Print the progress
    print("Progress: %.2f%%" %(100*(index/total_number_of_files)), end="\r", flush=True)

    try:
        # Read the JSON file and extract the meter ID and the s05 and s02 data
        data = json.load(open(json_files_path + '/' + filename, 'r'))
        meter_id = data['id']
        df_hourly_deltas = pd.DataFrame(data['timeline']['S02'])
        df_daily_deltas = pd.DataFrame(data['timeline']['S05'])

        # Rename the "Fh" column in both DataFrames
        df_hourly_deltas = df_hourly_deltas.rename(columns={'Fh': 'timestamp'})
        df_daily_deltas = df_daily_deltas.rename(columns={'Fh': 'timestamp'})

        # Explode the "Value" column in df_daily_deltas
        right_df = pd.DataFrame(df_daily_deltas['Value'].to_list())
        left_df = df_daily_deltas
        df_daily_deltas = pd.concat([left_df, right_df], axis=1, ignore_index=False).drop(columns='Value')

        # Get only the reactive data
        df_hourly_deltas = df_hourly_deltas[['timestamp', 'R1', 'R2', 'R3', 'R4']]
        df_daily_deltas = df_daily_deltas[['timestamp', 'R1a', 'R2a', 'R3a', 'R4a']]

        # Write the data into two CSV files
        df_hourly_deltas.to_csv(f"{csv_output_path}/meter_data_{meter_id}_S02.csv", index=False)
        df_hourly_deltas.to_csv(f"{csv_output_path}/meter_data_{meter_id}_S05.csv", index=False)

    except Exception:
        print(f"Something went wrong with file {filename}")

print("Finished")