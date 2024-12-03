import os
import shutil

def copy_files_to_outer_folder():
    base_dir = r"./raw data/auxiliarydata/hourly_readings"

    # Check if the base directory exists
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    # Go through all subfolders inside the base directory
    for root, dirs, files in os.walk(base_dir):
        # Skip the base directory itself (we only need the subfolders)
        if root == base_dir:
            continue

        for file_name in files:
            # Get full file path
            file_path = os.path.join(root, file_name)

            # Define the destination path (which is the base directory itself)
            dest_path = os.path.join(base_dir, file_name)

            # Copy the file to the base directory
            try:
                print(f"Copying {file_name} from {root} to {base_dir}")
                shutil.copy(file_path, dest_path)
            except shutil.SameFileError:
                print(f"{file_name} already exists in {base_dir}, skipping...")
            except Exception as e:
                print(f"Error copying {file_name}: {e}")

copy_files_to_outer_folder()
