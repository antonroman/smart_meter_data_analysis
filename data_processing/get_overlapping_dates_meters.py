"""
    Make a list with all the meters with data for the 2020 year, from 1st January 2020 to 31st December 2020, both included.
    If there are gaps between those two dates, it will be added to the list anyway. 

"""

import pandas as pd
from datetime import datetime as dt
import os

csv_output_filename = 'valid_meters_2020.csv'

good_files = []  # CSV files with data for the complete 2020

files_list = list(filter(lambda x: x.endswith('S05.csv'), os.listdir('reactive_values')))
for index, file in enumerate(files_list):
    # Iterate file by file
    print("Progress: %.2f%%" %(index/len(files_list)), end="\r", flush=True)
    df = pd.read_csv(f'reactive_values/{file}', index_col='timestamp')
    df.index = pd.to_datetime(df.index)

    # There is data for 1st January 2020 and 31st December 2020?
    if '2020-1-1' in df.index and '2020-12-31' in df.index:
        good_files.append(file)

# Store all good meters filenames in a CSV
with open(csv_output_filename, 'w') as f:
    for filename in good_files:
        f.write(filename + '\n')

# Print the number of meters with data for the complete 2020 year
print("Number of meters with data for the complete 2020: " + str(len(good_files)))


