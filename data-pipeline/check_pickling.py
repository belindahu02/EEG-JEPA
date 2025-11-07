import csv
import numpy as np
from pathlib import Path

# List of CSV files containing .npy file paths
csv_files = [
    "/media/SATA_2/belinda_hu/data/files_audioset.csv",
    # add more CSV paths if needed
]

# Base directory for .npy files (to resolve relative paths in CSV)
base_dir = Path("/media/SATA_2/belinda_hu/data")

problem_files = []
total_files = 0

for csv_file in csv_files:
    print(f"\nChecking CSV: {csv_file}")
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            file_path = base_dir / row['file_name']  # Adjust key if your CSV column is different
            print(f"Checking file {i}: {file_path}", end='\r', flush=True)
            total_files += 1
            try:
                _ = np.load(file_path, allow_pickle=False)
            except ValueError as e:
                if "allow_pickle=False" in str(e):
                    print(f"\nRequires pickling: {file_path}")
                    problem_files.append(file_path)

print(f"\nChecked {total_files} files in total.")
if problem_files:
    print(f"\nFiles that require pickling ({len(problem_files)}):")
    for f in problem_files:
        print(f)
else:
    print("\nAll files can be loaded without pickling.")
