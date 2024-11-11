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
    home_number_match = re.search(r'home(\d+).csv', file)
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
summary_df.to_csv('summary2.csv', index=False)

# Function to compute min, max, and average values for each sensor type across all homes
def compute_sensor_stats(directory, all_sensor_types):
    sensor_stats = {sensor: {'min': float('inf'), 'max': float('-inf'), 'sum': 0.0, 'count': 0} for sensor in all_sensor_types}
    for file in os.listdir(directory):
        if file.startswith('home') and file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            for col in df.columns:
                if col != 'timestamp':
                    sensor_type = parse_sensor_type(col)
                    if sensor_type in sensor_stats:
                        column_data = df[col].dropna()
                        if not column_data.empty:
                            current_min = column_data.min()
                            current_max = column_data.max()
                            sensor_stats[sensor_type]['min'] = min(sensor_stats[sensor_type]['min'], current_min)
                            sensor_stats[sensor_type]['max'] = max(sensor_stats[sensor_type]['max'], current_max)
                            sensor_stats[sensor_type]['sum'] += column_data.sum()
                            sensor_stats[sensor_type]['count'] += column_data.count()

    # Infer units based on sensor type
    sensor_units = {
        'temperature': '°C',
        'humidity': '% RH',
        'light': 'Uncalibrated units',
        'electric_combined': 'Watts',
        'electric_mains': 'Watts',
        'electric_subcircuit': 'Watts',
        'electric_appliance': 'Watts',
        'electric': 'Watts',
        'gas_pulse': 'Watt hours',
        'gas': 'Watt hours',
        'power': 'Watts',
        'clamp_temperature': '°C',
        # Add more sensor types and their units as needed
    }

    # Prepare summary data
    summary_list = []
    for sensor, values in sensor_stats.items():
        unit = sensor_units.get(sensor, 'Unknown')
        min_val = values['min'] if values['min'] != float('inf') else None
        max_val = values['max'] if values['max'] != float('-inf') else None
        average_val = values['sum'] / values['count'] if values['count'] > 0 else None
        summary_list.append({
            'sensor_type': sensor,
            'min_value': min_val,
            'max_value': max_val,
            'average_value': average_val,
            'unit': unit
        })

    # Create DataFrame and save to CSV
    stats_df = pd.DataFrame(summary_list)
    stats_df.to_csv('sensor_stats.csv', index=False)
    print("Sensor stats CSV has been generated as 'sensor_stats.csv'.")

# Call the function to compute min, max, and average values
compute_sensor_stats(directory, all_sensor_types)

print("Summary CSV has been generated as 'summary.csv'.")
