import pandas as pd
import re

# Read the summary.csv file
summary_df = pd.read_csv('summary.csv')

# Exclude non-sensor columns
sensor_columns = summary_df.columns.difference(
    ['home_number', 'earliest_timestamp', 'latest_timestamp', 'number_of_rows'])

# Initialize a nested dictionary to hold counts
# Structure: {sensor_type: {room: count, ...}, ...}
sensor_room_counts = {}

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

    # Ensure the sensor_type exists in the dictionary
    if sensor_type not in sensor_room_counts:
        sensor_room_counts[sensor_type] = {}
    # Initialize the room type count if not present
    if room not in sensor_room_counts[sensor_type]:
        sensor_room_counts[sensor_type][room] = 0

    # Count the number of homes where the number of valid datapoints is greater than 0
    count_homes = (summary_df[col] > 0).sum()
    sensor_room_counts[sensor_type][room] += count_homes

# Convert the nested dictionary to a DataFrame
sensor_summary_df = pd.DataFrame.from_dict(sensor_room_counts, orient='index').fillna(0).astype(int)

# Optionally, sort the rows and columns for better readability
sensor_summary_df = sensor_summary_df.sort_index()
sensor_summary_df = sensor_summary_df.reindex(sorted(sensor_summary_df.columns), axis=1)

# Save the DataFrame to a CSV file
sensor_summary_df.to_csv('room_summary.csv')

print("Room summary CSV (with sensors as rows and rooms as columns) has been generated as 'room_summary.csv'.")
