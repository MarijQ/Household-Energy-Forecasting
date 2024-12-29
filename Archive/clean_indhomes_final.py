import os
import pandas as pd

# Define the source and destination directories
source_dir = "ind-homes-with-weather"
dest_dir = "ind-homes-final"

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Columns to drop
columns_to_remove = [
    "starttime", 
    "starttime_enhanced", 
    "endtime", 
    "cohortid", 
    "urban_rural_class", 
    "new_build_year"
]

# Iterate through each CSV file in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(".csv"):  # Process only CSV files
        # Define full paths for source and destination files
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(dest_dir, filename)

        # Read the CSV file
        df = pd.read_csv(source_file)

        # Drop the specified columns
        df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')

        # Save the cleaned data to the destination directory
        df_cleaned.to_csv(dest_file, index=False)

        print(f"Processed: {filename}")

print(f"Cleaned CSV files have been saved to: {dest_dir}")
