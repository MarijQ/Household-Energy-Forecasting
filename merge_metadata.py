import os
import pandas as pd

# Paths
METADATA_FILE = 'home_metadata.csv'
DATA_DIR = 'ind-homes-clean'
OUTPUT_DIR = 'ind-homes-clean-modified'

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the home metadata into a DataFrame
metadata_df = pd.read_csv(METADATA_FILE)

# Iterate over each CSV file in the data directory
for filename in os.listdir(DATA_DIR):
    if filename.endswith('.csv'):
        file_path = os.path.join(DATA_DIR, filename)
        
        # Extract the home ID from the filename
        # e.g., 'home_home306.csv' -> '306'
        parts = filename.replace('.csv', '').split('_')
        if len(parts) >= 2:
            home_id_str = parts[1].replace('home', '')
        else:
            continue  # Skip files that don't match the expected pattern

        # Convert home ID to integer
        try:
            home_id = int(home_id_str)
        except ValueError:
            print(f"Invalid home ID in filename: {filename}")
            continue

        # Find the metadata for this home ID
        home_metadata = metadata_df[metadata_df['homeid'] == home_id]
        if home_metadata.empty:
            print(f"No metadata found for home ID {home_id}")
            continue
        
        # Get the location and other metadata
        location = home_metadata['location'].values[0]
        metadata_dict = home_metadata.to_dict(orient='records')[0]
        metadata_dict.pop('homeid', None)  # Remove 'homeid' from metadata

        # Load the home's data file
        df = pd.read_csv(file_path)

        # Rename the second column to 'elec' and third column to 'gas'
        if df.shape[1] >= 3:
            df.columns.values[1] = 'elec'
            df.columns.values[2] = 'gas'

        # Add metadata columns to the DataFrame
        for key, value in metadata_dict.items():
            df[key] = value  # Fill the entire column with the metadata value

        # Construct the new filename
        new_filename = f"{home_id}_{location}.csv"
        new_file_path = os.path.join(OUTPUT_DIR, new_filename)

        # Save the modified DataFrame
        df.to_csv(new_file_path, index=False)

        print(f"Processed {filename} -> {new_filename}")
