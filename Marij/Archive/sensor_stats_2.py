import pandas as pd
import re

input_file = 'sensor_stats.csv'
output_file = 'sensor_summary_combined.csv'

# Function to extract general sensor type from the full sensor type string
def extract_sensor_type(full_sensor_type):
    parts = full_sensor_type.split('_')
    return '_'.join(parts[1:])  # Ignore the room part (first element)

# Read the input CSV file
df = pd.read_csv(input_file)

# Extract sensor types without room prefixes
df['clean_sensor_type'] = df['sensor_type'].apply(extract_sensor_type)

# Define correct units for each sensor type
sensor_units = {
    'temperature': '°C',
    'humidity': '% RH',
    'light': 'Uncalibrated units',
    'electric-combined': 'Watts',
    'electric': 'Watts',
    'electric_appliance': 'Watts',
    'gas-pulse_gas': 'Watt hours',
    'power': 'Watts',
    'clamp_temperature': '°C',
    'tempprobe': '°C',
    'room_humidity': '% RH',
    'room_temperature': '°C',
    'room_light': 'Uncalibrated units',
    'heater_temperature': '°C',
    'heater_humidity': '% RH',
    # Add more sensor types and their units as needed
}

# Function to infer unit from sensor type
def infer_unit(sensor_type):
    for key in sensor_units:
        if key in sensor_type:
            return sensor_units[key]
    return 'Unknown'

# Group by sensor type, ignoring room prefixes, and calculate combined stats
summary = df.groupby('clean_sensor_type').agg(
    min_value=('min_value', 'min'),
    max_value=('max_value', 'max'),
    sum_value=('average_value', 'sum'),
    count=('average_value', 'count')  # To calculate global average
).reset_index()

# Calculate average values
summary['average_value'] = summary['sum_value'] / summary['count']

# Infer units based on sensor type
summary['unit'] = summary['clean_sensor_type'].apply(infer_unit)

# Drop the sum and count columns as they are no longer needed
summary.drop(columns=['sum_value', 'count'], inplace=True)

# Rename the columns for clarity
summary.rename(columns={'clean_sensor_type': 'sensor_type'}, inplace=True)

# Save the summary to a new CSV file
summary.to_csv(output_file, index=False)

print(f'Summary has been saved to {output_file}.')
