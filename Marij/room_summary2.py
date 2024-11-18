import pandas as pd
import re

# Read the summary.csv file
summary_df = pd.read_csv('summary.csv')

# Exclude non-sensor columns
sensor_columns = summary_df.columns.difference(
    ['home_number', 'earliest_timestamp', 'latest_timestamp', 'number_of_rows'])

# Initialize a nested dictionary to hold counts
sensor_room_counts = {}
sensor_stats = {}

# Process each sensor column to extract room and sensor type
for col in sensor_columns:
    # Clean the column name by removing numeric suffixes and 'sensor' word
    col_cleaned = re.sub(r'\d+', '', col)
    col_cleaned = col_cleaned.replace('sensor', '')
    parts = col_cleaned.split('_')

    if len(parts) >= 2:
        room = parts[0]
        sensor_type = '_'.join(parts[1:])
    else:
        continue  # Skip columns that don't match the expected pattern

    # Ensure the sensor_type exists in the dictionary for room counts
    if sensor_type not in sensor_room_counts:
        sensor_room_counts[sensor_type] = {}
    if room not in sensor_room_counts[sensor_type]:
        sensor_room_counts[sensor_type][room] = 0

    # Count the number of homes where the number of valid datapoints is greater than 0
    count_homes = (summary_df[col] > 0).sum()
    sensor_room_counts[sensor_type][room] += count_homes

    # Calculate min/max statistics across all homes for the sensor
    min_value = summary_df[col].min()
    max_value = summary_df[col].max()

    # Infer units based on sensor type
    if 'temperature' in sensor_type:
        unit = '°C'
    elif 'humidity' in sensor_type:
        unit = '%'
    elif 'electric' in sensor_type or 'power' in sensor_type:
        unit = 'W'
    elif 'gas' in sensor_type:
        unit = 'Wh'
    elif 'light' in sensor_type:
        unit = 'Undetermined'
    else:
        unit = 'Unknown'

    # Store sensor statistics
    if sensor_type not in sensor_stats:
        sensor_stats[sensor_type] = {'min': min_value, 'max': max_value, 'unit': unit}

# Convert the nested dictionary for room counts to a DataFrame
sensor_summary_df = pd.DataFrame.from_dict(sensor_room_counts, orient='index').fillna(0).astype(int)

# Sort the rows and columns for better readability
sensor_summary_df = sensor_summary_df.sort_index()
sensor_summary_df = sensor_summary_df.reindex(sorted(sensor_summary_df.columns), axis=1)

# Save the room summary to CSV
sensor_summary_df.to_csv('room_summary2.csv')

# Convert sensor stats to DataFrame and save to CSV
sensor_stats_df = pd.DataFrame(sensor_stats).T
sensor_stats_df.to_csv('sensor_stats_summary.csv')

print("Room summary CSV has been generated as 'room_summary.csv'.")
print("Sensor stats summary CSV has been generated as 'sensor_stats_summary.csv'.")
