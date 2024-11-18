import os
import pandas as pd

directory = r"./raw data/auxiliarydata/hourly_readings/"

# Check length of files
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

for file in csv_files[:]:
    file_path = os.path.join(directory, file)
    data = pd.read_csv(file_path, header=None)
    print(f"{file} - Number of rows: {len(data)}")
