import pandas as pd
import os
from matplotlib import cm
import matplotlib.pyplot as plt

reactive_aggregated_values_folder = 'aggregated_reactive'
plots_output_folder = 'plots'

os.makedirs(plots_output_folder, exist_ok=True)

for reactive_aggregation_file in sorted(filter(lambda x: '_S02_' in x, os.listdir(reactive_aggregated_values_folder))):
    aggregation_level = reactive_aggregation_file.split('_')[-1][:-4]
    print(f"Plotting aggregated data for {aggregation_level}% aggregation")

    # Read the aggregated S02 CSV file
    df = pd.read_csv(f'{reactive_aggregated_values_folder}/{reactive_aggregation_file}')
    df = df[['timestamp', 'R1']]

    # Process the date
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_of_year'] = df['timestamp'].apply(lambda x: x.date().timetuple().tm_yday)
    df['day_of_week'] = df['timestamp'].apply(lambda x: x.dayofweek)
    df['hour'] = df['timestamp'].apply(lambda x: x.hour)

    # Create a plot with the yearly consumption
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(14,10))
    grouped_df = df.groupby(['day_of_year', 'hour']).sum().reset_index()
    ax.plot_trisurf(grouped_df['day_of_year'], grouped_df['hour'], grouped_df['R1'], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title('Power consumption across a natural year')
    ax.set_xlabel('Day of the year')
    ax.set_ylabel('Hour of the day')
    ax.set_zlabel('Power consumption (Wh)')
    plt.savefig(f'{plots_output_folder}/yearly_{aggregation_level}.png')

    # Create a plot with the weekly consumption along a natural year
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(18,20))
    grouped_df = df.groupby(['day_of_week', 'hour']).sum().reset_index()
    ax.plot_trisurf(grouped_df['day_of_week'], grouped_df['hour'], grouped_df['R1'], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title('Power consumption across the week days')
    ax.set_xlabel('Day of the week')
    ax.set_ylabel('Hour of the day')
    ax.set_zlabel('Power consumption (Wh)')
    plt.savefig(f'{plots_output_folder}/weekly_{aggregation_level}.png')

