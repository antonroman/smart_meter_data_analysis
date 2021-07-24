"""
    Aggregate S02 and S05 CSV files in 10%, 20% ... 100%.
    It needs a CSV with all the meters which have data for all 2020 (see get_overlapping_dates_meters.py).
    Optionally, a CSV file with the detected outliers can be used, so the outliers will be removed from the aggregations.

"""
import pandas as pd
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, mean, stddev

valid_files_csv_filename = 'valid_meters_2020.csv'
meters_with_outliers_filename = 'outliers_S02.csv'
reactive_values_files_folder = 'reactive_values'
aggregated_files_output_folder = 'aggregated_reactive_spark'

# Time window
from_day = '2020-01-01'
to_day = '2020-12-31'

spark = SparkSession.builder.appName("Aggregate reactive values").getOrCreate()


def aggregate_files(file_type):
    with open(valid_files_csv_filename, 'r') as f:
        # Read the file list form the valid meters list
        file_list = list(map(lambda x: x.strip(), f.readlines()))
        file_list = list(filter(lambda x: file_type in x, file_list))

    # Remove the outliers
    if file_type == 'S02':
        outliers_df = pd.read_csv(meters_with_outliers_filename, low_memory=False, skiprows=1)
        meters_with_outliers = list(outliers_df['meter_id'].value_counts().index)
        file_list = list(filter(lambda x: x.split('_')[-2] not in meters_with_outliers, file_list))

    aggregation_levels = [x / 10 for x in range(1, 11)]
    for agg_level in aggregation_levels:
        print("Aggregating %i%% of the files..." % (100 * agg_level))

        # For each aggregation level, select the corresponding files
        agg_files_list = list(map(lambda x: f'{reactive_values_files_folder}/{x}',
                                  file_list[0:int(agg_level * len(file_list) - 1)]))
        print(len(agg_files_list))

        # Name of the output aggregated file
        agg_filename = f'agg_values_{file_type}_{str(int(100 * agg_level))}.csv'

        # Read the aggregated CSVs in a DataFrame
        df = spark.read.csv(agg_files_list, header=True, inferSchema=True)

        # Parse the timestamp and get only the data from 2020
        df = df.withColumn("timestamp_norm", to_timestamp(col("timestamp"))).drop("timestamp") \
            .withColumnRenamed("timestamp_norm", "timestamp")
        df = df.filter((col("timestamp") >= from_day) & (col("timestamp") <= to_day))

        # Aggregate the data by time
        df = df.groupBy('timestamp').sum().orderBy('timestamp')
        df = df.withColumnRenamed("sum(R1)", "R1")\
            .withColumnRenamed("sum(R2)", "R2")\
            .withColumnRenamed("sum(R3)", "R3")\
            .withColumnRenamed("sum(R4)", "R4") \
            .withColumnRenamed("sum(R1a)", "R1a") \
            .withColumnRenamed("sum(R2a)", "R2a") \
            .withColumnRenamed("sum(R3a)", "R3a") \
            .withColumnRenamed("sum(R4a)", "R4a")

        # Save the data
        df.write.csv(f'{aggregated_files_output_folder}/{agg_filename}', mode='overwrite')


os.makedirs(aggregated_files_output_folder, exist_ok=True)

print("Aggregating S02 data...\n")
aggregate_files('S02')

print("\n\nAggregating S05 data...\n")
aggregate_files('S05')
