import os
import pandas as pd

# Path to the folder containing the sensor data files
folder_path = './raw data/sensordata/'

# Initialize counters and file structure samples for each category
categories = {
    'electric-mains': 0,
    'gas-pulse': 0,
    'tempprobe': 0,
    'electric-subcircuit': 0,
    'unlabelled-subcircuit': 0,
}

# Dictionary to store a sample structure of each category
file_structures = {key: None for key in categories}

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    if 'electric-mains' in file_name:
        categories['electric-mains'] += 1
        if file_structures['electric-mains'] is None:
            file_structures['electric-mains'] = pd.read_csv(os.path.join(folder_path, file_name)).head()
    elif 'gas-pulse' in file_name:
        categories['gas-pulse'] += 1
        if file_structures['gas-pulse'] is None:
            file_structures['gas-pulse'] = pd.read_csv(os.path.join(folder_path, file_name)).head()
    elif 'tempprobe' in file_name:
        categories['tempprobe'] += 1
        if file_structures['tempprobe'] is None:
            file_structures['tempprobe'] = pd.read_csv(os.path.join(folder_path, file_name)).head()
    elif 'electric-subcircuit' in file_name:
        categories['electric-subcircuit'] += 1
        if file_structures['electric-subcircuit'] is None:
            file_structures['electric-subcircuit'] = pd.read_csv(os.path.join(folder_path, file_name)).head()
    elif 'unlabelled' in file_name:
        categories['unlabelled-subcircuit'] += 1
        if file_structures['unlabelled-subcircuit'] is None:
            file_structures['unlabelled-subcircuit'] = pd.read_csv(os.path.join(folder_path, file_name)).head()

# Output file counts for each category
print("File counts per category:")
for category, count in categories.items():
    print(f"{category}: {count}")

# Display a sample structure of each category for review
print("\nSample structure of files:")
for category, df in file_structures.items():
    print(f"\n{category} sample:")
    print(df)
