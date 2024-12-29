import os
import pandas as pd

directory = r"./raw data/auxiliarydata/hourly_readings/"

# Function to extract components (home, room, sensor, probe) from filenames
def parse_filename(filename):
    # Assumes filename format: home<number>_<room><number>_sensor<number>_<probe>.csv
    parts = filename.replace('.csv', '').split('_')
    home = parts[0]
    room = parts[1]
    sensor = parts[2]
    probe = "_".join(parts[3:])
    return home, room, sensor, probe

# Initialize an empty merged DataFrame
merged_df = pd.DataFrame()

# Iterate over all CSV files in the directory
for i, file in enumerate(os.listdir(directory)):
    if file.endswith(".csv"):
        file_path = os.path.join(directory, file)

        # Read csv
        df = pd.read_csv(file_path, header=None, names=['timestamp', 'value'])

        # Convert timestamp into datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Extract metadata from the filename to create unique column names
        home, room, sensor, probe = parse_filename(file)
        sensor_column_name = f'{home}_{room}_{sensor}_{probe}'

        # Add the sensor data as a new column with the unique sensor name
        df[sensor_column_name] = df['value']
        df.drop(columns='value', inplace=True)

        # Merge the current DataFrame with the main merged dataframe, aligning by timestamp
        if merged_df.empty:
            # The first file determines the initial structure of the DataFrame
            merged_df = df
        else:
            # Incrementally merge with how='outer' to include all timestamps
            merged_df = pd.merge(merged_df, df, on='timestamp', how='outer')

        # For progress feedback
        if i % 100 == 0:
            print(f"Processed {i} files...")

# Sort the DataFrame by the timestamp after merging
merged_df.sort_values(by="timestamp", inplace=True)

merged_df.to_csv('merged_sensor_data.csv', index=False)

print("Merging complete. The file 'merged_sensor_data.csv' is ready.")
