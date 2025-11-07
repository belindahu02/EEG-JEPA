"""
Quick test to debug data path issue
Run this to see what's happening with your data files
"""

from pathlib import Path

# Test your data path
path = "/app/data/1.0.0"

print("="*70)
print("DEBUG: Checking data paths")
print("="*70)

# Check if base path exists
base_path = Path(path)
print(f"\n1. Checking base path: {path}")
print(f"   Exists? {base_path.exists()}")

if base_path.exists():
    print(f"   Contents:")
    for item in list(base_path.iterdir())[:10]:  # Show first 10 items
        print(f"   - {item.name}")
else:
    print("   ERROR: Base path does not exist!")
    print("\n   Your Docker mount should map:")
    print("   Host: /disks/SATA_2/belinda_h/data/1.0.0")
    print("   Container: /app/data/1.0.0")
    exit(1)

# Check for user folders
print(f"\n2. Checking for user folders (S001, S002, etc.)")
user_folders_found = []
for user_id in range(1, 10):  # Check first 9 users
    user_folder = f"S{user_id:03d}"
    user_path = base_path / user_folder
    if user_path.exists():
        user_folders_found.append(user_folder)
        print(f"   ✓ Found: {user_folder}")
    else:
        print(f"   ✗ Missing: {user_folder}")

if not user_folders_found:
    print("\n   ERROR: No user folders found!")
    print("   Expected folders like: S001, S002, S003, ...")
    exit(1)

# Check for EDF files in first found user
print(f"\n3. Checking for EDF files in {user_folders_found[0]}")
first_user_path = base_path / user_folders_found[0]
edf_files = list(first_user_path.glob("*.edf"))

if edf_files:
    print(f"   Found {len(edf_files)} EDF files:")
    for edf_file in edf_files[:5]:  # Show first 5
        print(f"   - {edf_file.name}")
else:
    print(f"   ERROR: No .edf files found in {first_user_path}")
    print(f"   Contents of {first_user_path}:")
    for item in list(first_user_path.iterdir())[:10]:
        print(f"   - {item.name}")
    exit(1)

# Test expected filename format
print(f"\n4. Checking filename format")
expected_format = f"{user_folders_found[0]}R01_1.edf"
test_file = first_user_path / expected_format
print(f"   Looking for: {expected_format}")
print(f"   Expected path: {test_file}")
print(f"   Exists? {test_file.exists()}")

if not test_file.exists():
    print(f"\n   Your files might have a different naming convention.")
    print(f"   Expected: S001R01_1.edf")
    print(f"   Actual files in directory:")
    for edf_file in edf_files[:10]:
        print(f"   - {edf_file.name}")

# Test loading an EDF file
print(f"\n5. Testing EDF file loading")
try:
    import mne
    
    # Try to load the first EDF file found
    if edf_files:
        test_file = edf_files[0]
        print(f"   Attempting to load: {test_file.name}")
        raw = mne.io.read_raw_edf(str(test_file), preload=True, verbose=False)
        data = raw.get_data()[:64, :]
        print(f"   ✓ Successfully loaded!")
        print(f"   Shape: {data.shape}")
        print(f"   Channels: {data.shape[0]}, Samples: {data.shape[1]}")
    else:
        print(f"   No EDF files to test")
        
except Exception as e:
    print(f"   ✗ Error loading EDF file: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DEBUG TEST COMPLETE")
print("="*70)
