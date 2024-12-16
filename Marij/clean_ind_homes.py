import os
import pandas as pd

# Directories
INPUT_DIR = "./ind-homes"
OUTPUT_DIR = "./ind-homes-clean"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lists to track ignored files
ignored_files = []

# Function to clean a single file
def clean_file(file_path, output_path):
    """
    Cleans the given CSV file from the ind-homes folder and saves the cleaned version.

    Args:
        file_path (str): The path to the input file.
        output_path (str): The path to save the cleaned file.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Filter columns: keep 'timestamp' and columns containing 'elec' or 'gas'
    relevant_columns = ['timestamp'] + [col for col in df.columns if 'elec' in col or 'gas' in col]
    df = df[relevant_columns]

    # Ensure 'timestamp' is in datetime format
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Check percentage of missing values for 'elec' and 'gas'
    for col in df.columns:
        if 'elec' in col or 'gas' in col:
            missing_percentage = df[col].isna().mean() * 100
            if missing_percentage > 50:
                ignored_files.append((os.path.basename(file_path), col, missing_percentage))
                return  # Ignore this file completely if one column exceeds 50% missing

    # Impute missing values
    for col in df.columns:
        if 'elec' in col:
            df[col] = df[col].ffill().bfill()  # Forward and backward fill
        elif 'gas' in col:
            df[col] = df[col].fillna(0)  # Fill missing gas values with 0

    # Save the cleaned file
    df.to_csv(output_path, index=False)

# Function to check for missing values in cleaned files
def check_missing_values(directory):
    """
    Checks for missing values in all cleaned files and provides a summary.

    Args:
        directory (str): Path to the directory containing cleaned CSV files.

    Returns:
        None
    """
    missing_summary = []
    print("\nMissing Value Check:")
    for file_name in os.listdir(directory):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory, file_name)
            df = pd.read_csv(file_path)
            missing_counts = df.isna().sum()
            total_missing = missing_counts.sum()
            
            if total_missing > 0:
                print(f"{file_name}: {total_missing} missing values.")
                missing_summary.append((file_name, total_missing))
            else:
                print(f"{file_name}: No missing values.")

    if missing_summary:
        print("\nSummary of Files with Missing Data:")
        for file, count in missing_summary:
            print(f"{file}: {count} missing values")
    else:
        print("\nAll cleaned files have no missing values.")

# Process files in input directory
for file_name in os.listdir(INPUT_DIR):
    if file_name.endswith(".csv"):
        input_file_path = os.path.join(INPUT_DIR, file_name)
        output_file_path = os.path.join(OUTPUT_DIR, file_name)

        # Clean and save the file (or ignore if conditions aren't met)
        clean_file(input_file_path, output_file_path)
        if os.path.exists(output_file_path):
            print(f"Cleaned and saved: {output_file_path}")

# Report ignored files
if ignored_files:
    print("\nIgnored Files Summary:")
    for file_name, col, percent_missing in ignored_files:
        print(f"{file_name}: Column '{col}' has {percent_missing:.2f}% missing values (> 50%).")
else:
    print("\nNo files were ignored due to missing value thresholds.")

# Perform a final check for missing values in cleaned files
check_missing_values(OUTPUT_DIR)

print("All files have been processed, cleaned, and checked for missing values.")