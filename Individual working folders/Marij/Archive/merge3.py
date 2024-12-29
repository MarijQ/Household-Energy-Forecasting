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

# Initialize a dictionary to hold DataFrames for each home
home_dfs = {}

# Iterate over all CSV files in the directory
for i, file in enumerate(os.listdir(directory)):
    if file.endswith(".csv"):
        file_path = os.path.join(directory, file)

        # Read the CSV
        df = pd.read_csv(file_path, header=None, names=['timestamp', 'value'])

        # Convert the timestamp into datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Extract metadata from the filename to create unique column names
        home, room, sensor, probe = parse_filename(file)
        sensor_column_name = f'{home}_{room}_{sensor}_{probe}'

        # Add the sensor data as a new column with the unique sensor name
        df[sensor_column_name] = df['value']
        df.drop(columns='value', inplace=True)

        # Check if this home already has a DataFrame, if not, initialize one
        if home not in home_dfs:
            home_dfs[home] = df  # First file for this home
        else:
            # Incrementally merge with how='outer' to include all timestamps
            home_dfs[home] = pd.merge(home_dfs[home], df, on='timestamp', how='outer')

        # For progress feedback
        if i % 100 == 0:
            print(f"Processed {i} files...")

# Save each home's DataFrame to a separate CSV file
for home, home_df in home_dfs.items():
    # Sort the DataFrame by timestamp
    home_df.sort_values(by="timestamp", inplace=True)

    # Define the output file name
    output_file_path = f"./home_{home}.csv"

    # Save the DataFrame to CSV
    home_df.to_csv(output_file_path, index=False)

    print(f"Saved {output_file_path}")

print("Processing complete. Separate files for each home have been generated.")
