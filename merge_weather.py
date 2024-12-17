import os
import pandas as pd

# Paths
INPUT_DIR = 'ind-homes-clean-modified'
WEATHER_DATA_DIR = 'weather_data'
OUTPUT_DIR = 'ind-homes-with-weather'

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Iterate over each CSV file in the modified data directory
for filename in os.listdir(INPUT_DIR):
    if filename.endswith('.csv'):
        try:
            file_path = os.path.join(INPUT_DIR, filename)
            
            # Extract the location from the filename
            # e.g., '306_Edinburgh.csv' -> 'Edinburgh'
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                location = parts[1]
            else:
                print(f"Could not extract location from filename: {filename}")
                continue  # Skip files that don't match the expected pattern
            
            # Construct the weather data filename
            weather_filename = f'hourly_temperatures_{location}_data.csv'
            weather_file_path = os.path.join(WEATHER_DATA_DIR, weather_filename)
            
            # Check if the weather data file exists
            if not os.path.exists(weather_file_path):
                print(f"Weather data file not found for location '{location}': {weather_filename}")
                continue
            
            # Load the home's data file
            home_df = pd.read_csv(file_path)
            
            # Load the weather data file
            weather_df = pd.read_csv(weather_file_path)
            
            # Ensure 'timestamp' columns are in datetime format
            home_df['timestamp'] = pd.to_datetime(home_df['timestamp'], errors='coerce')
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')
            
            # Merge the weather data into the home data on 'timestamp'
            merged_df = pd.merge(home_df, weather_df[['timestamp', 'temperature']], on='timestamp', how='left')
            
            # Insert the temperature column after 'gas'
            cols = list(merged_df.columns)
            gas_index = cols.index('gas')
            # Remove 'temperature' column and re-insert it after 'gas'
            cols.remove('temperature')
            cols.insert(gas_index + 1, 'temperature')
            merged_df = merged_df[cols]
            
            # Construct the output file path
            output_file_path = os.path.join(OUTPUT_DIR, filename)
            
            # Save the updated DataFrame to the output directory
            merged_df.to_csv(output_file_path, index=False)
            
            print(f"Processed and saved {filename} to {output_file_path}.")
        
        except Exception as e:
            print(f"Skipping {filename} due to an error: {e}")
