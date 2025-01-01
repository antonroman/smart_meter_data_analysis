import pandas as pd

def convert_csv(input_file, output_file):
    """
    Converts a CSV with the specified format to a new CSV with the desired columns.

    Args:
        input_file: Path to the input CSV file.
        output_file: Path to the output CSV file.
    """

    try:
        # Read the input CSV file
        df = pd.read_csv(input_file, header=None, names=['timestamp', 'value1', 'value2', 'value3', 'value4'], skiprows=1)

        # Extract the desired columns and format them
        df['timestamp'] = df['timestamp'].astype(str)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S%z')  
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%m/%m/%y %H:%M')
        df = df[['timestamp', 'value1']]

        # Write the converted data to a new CSV file
        df.to_csv(output_file, index=False, header=False)

        print(f"Conversion successful. Output saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
input_file = '/home/antonroman/src/smart_meter_data_analysis/data_processing/aggregated_reactive/agg_values_S02_100.csv'  # Replace with the actual input file path
output_file = '/home/antonroman/src/smart_meter_data_analysis/data_processing/aggregated_reactive/agg_values_S02_100_adapte_htm_format.csv'  # Replace with the desired output file path
convert_csv(input_file, output_file)