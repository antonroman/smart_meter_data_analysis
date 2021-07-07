"""
    Search for outliers in 2020 in the S02 files.
    It needs a CSV with all the meters which have data for all 2020 (see get_overlapping_dates_meters.py).

"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, input_file_name, to_timestamp, mean, stddev
import os

valid_files_csv_filename = 'valid_meters_2020.csv'
reactive_values_files_folder = 'reactive_values'
output_folder = 'outliers_spark'

outliers_threshold = 4

# Time window
from_day = '2020-01-01'
to_day = '2020-12-31'

os.makedirs(output_folder, exist_ok=True)

print("Searching for outliers...")

with open(valid_files_csv_filename, 'r') as f:
    # Read the file list form the valid meters list
    file_list = map(lambda x: x.strip(), f.readlines())
    file_list = filter(lambda x: 'S02' in x, file_list)
    file_list = list(map(lambda x: f'{reactive_values_files_folder}/{x}', file_list))

# Read all the CSVs
spark = SparkSession.builder.appName("Outliers search").getOrCreate()
df = spark.read.csv(file_list, header=True, inferSchema=True)

# Get the meter id for each row
df = df.withColumn("filename", input_file_name())
df = df.rdd.map(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[5].split("/")[-1].split('_')[-2])) \
    .toDF(['timestamp', 'R1', 'R2', 'R3', 'R4', 'filename', 'meter_id']).drop("filename")

# Parse the timestamp and get only the data from 2020
df = df.withColumn("timestamp_norm", to_timestamp(col("timestamp"))).drop("timestamp") \
    .withColumnRenamed("timestamp_norm", "timestamp")
df = df.filter((col("timestamp") >= from_day) & (col("timestamp") <= to_day))

# Calculate the outlier threshold for each date
df_outliers = df.groupBy("timestamp").agg(mean("R1").alias("avg_R1"), stddev("R1").alias("std_R1"))\
    .withColumn("upper_limit", col("avg_R1") + 4*col("std_R1")).drop("avg_R1", "std_R1")
df = df.join(df_outliers, 'timestamp', 'left')  # add the upper limit per timestamp to the main DataFrame

# Print the number of meters with outliers in R1, and the number of outliers for each of those meters
outliers = df.filter(col("R1") > col("upper_limit"))
outliers.write.format("csv").save(f'{output_folder}/outliers_S02.csv', mode='overwrite')

meters_with_outliers = outliers.groupBy("meter_id").count().orderBy("count", ascending=False)
meters_with_outliers.write.format("csv").save(f'{output_folder}/number_of_outliers_by_meter.csv', mode='overwrite')

print("\nMeters with outliers: " + meters_with_outliers.select('meter_id').distinct().count())

spark.stop()
