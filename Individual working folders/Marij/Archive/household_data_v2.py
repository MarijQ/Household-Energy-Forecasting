import os
import pandas as pd
import re

# Specify your folder path
folder_path = './raw data/sensordata/'


def clean_name(name):
    """
    This function removes trailing sensor numbers, variations like 'sensorXXXXc', and specific room IDs,
    keeping only the core 'sensor' or room name.
    """
    # Clean room numbers (e.g., hall654 -> hall)
    name = re.sub(r'\d+$', '', name)
    # Normalize all sensor variations 'sensorXXX', 'sensorXXXc', 'sensorcXXX' to just 'sensor'
    name = re.sub(r'sensorc?\d*c?', 'sensor', name)
    return name


def get_files_info(folder_path):
    # List all files in folder (only CSVs)
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Total number of CSV files
    total_files = len(file_list)

    # Extract room and sensor information
    room_sensor_data = []
    for file_name in file_list:
        parts = file_name.split('_')
        if len(parts) >= 3:
            room = clean_name(parts[1])  # Clean room name by removing trailing digits
            sensor = clean_name(parts[2])  # Clean sensor name and normalize variations
            room_sensor_data.append((room, sensor))

    # Create a dataframe for analysis
    df = pd.DataFrame(room_sensor_data, columns=['Room', 'Sensor'])

    return total_files, df


# Call the function to retrieve file info
total_files, df = get_files_info(folder_path)

# Get unique rooms (after cleaning)
unique_rooms = df['Room'].unique()

# Aggregating data for sensor types and room occurrences
sensor_summary = df.groupby(['Sensor', 'Room']).size().reset_index(name='Count')

# Print results
print(f"Total number of CSV files: {total_files}")
print("\nList of Unique Rooms:")
print(unique_rooms)

print("\nTable Summary (Sensor Type, Room, Count):")
print(sensor_summary.to_string(index=False))
