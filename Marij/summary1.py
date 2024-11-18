import os
import pandas as pd
import re

directory = r"./ind_homes/"  # Directory where the homeXXX.csv files are located

# Function to extract general sensor type from column names
def parse_sensor_type(column_name):
    # Remove home prefix and split by underscores
    parts = column_name.split('_')
    # Ignore the home number ('homeXX') in the first part
    if parts[0].startswith('home'):
        parts = parts[1:]
    # Remove numerical suffixes and 'sensor' word
    cleaned_parts = []
    for part in parts:
        # Remove numeric characters
        part = re.sub(r'\d+', '', part)
        # Remove 'sensor' word
        part = part.replace('sensor', '')
        # Only add non-empty parts
        if part:
            cleaned_parts.append(part)
    # Rejoin the parts to form the general sensor type
    sensor_type = '_'.join(cleaned_parts)
    return sensor_type

# Initialize a set to collect all sensor types
all_sensor_types = set()

# Collect all home CSV files
home_files = [f for f in os.listdir(directory) if f.startswith('home') and f.endswith('.csv')]

# First pass to collect all sensor types
for file in home_files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    # Exclude 'timestamp' column
    sensor_columns = [col for col in df.columns if col != 'timestamp']
    for col in sensor_columns:
        sensor_type = parse_sensor_type(col)
        all_sensor_types.add(sensor_type)

# Convert set to sorted list for consistent column order
all_sensor_types = sorted(all_sensor_types)

# Initialize list to hold summary data
summary_data = []

# Second pass to compute summary per home
for file in home_files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    # Ensure 'timestamp' column is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    home_number_match = re.search(r'home_(\w+).csv', file)
    if home_number_match:
        home_number = home_number_match.group(1)
    else:
        home_number = file.replace('home', '').replace('.csv', '')
    earliest_timestamp = df['timestamp'].min()
    latest_timestamp = df['timestamp'].max()
    num_rows = len(df)

    # Initialize dict for sensor counts
    sensor_counts = dict.fromkeys(all_sensor_types, 0)

    # Count non-null data points for each sensor type
    for col in df.columns:
        if col != 'timestamp':
            sensor_type = parse_sensor_type(col)
            count_non_null = df[col].count()
            sensor_counts[sensor_type] += count_non_null

    # Prepare the row data
    row_data = {
        'home_number': home_number,
        'earliest_timestamp': earliest_timestamp,
        'latest_timestamp': latest_timestamp,
        'number_of_rows': num_rows
    }
    # Add sensor counts to row data
    row_data.update(sensor_counts)
    # Append to summary data list
    summary_data.append(row_data)

# Create DataFrame for summary
summary_df = pd.DataFrame(summary_data)

# Save summary DataFrame to CSV
summary_df.to_csv('summary.csv', index=False)

print("Summary CSV has been generated as 'summary.csv'.")
