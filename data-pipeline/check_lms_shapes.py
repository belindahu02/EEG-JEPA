import os
import numpy as np

EXPECTED_SHAPE = (1, 80, 2000)
EXPECTED_SIZE = np.prod(EXPECTED_SHAPE)
ROOT_DIR = "."  # adjust if needed

bad_files = []
total_files = 0
total_bad = 0

subject_dirs = sorted([d for d in os.listdir(ROOT_DIR) if d.startswith("S") and os.path.isdir(os.path.join(ROOT_DIR, d))])

for subj in subject_dirs:
    subj_path = os.path.join(ROOT_DIR, subj)
    subj_file_count = 0
    subj_bad_count = 0

    for root, _, files in os.walk(subj_path):
        for file in files:
            if not file.endswith(".npy"):
                continue
            path = os.path.join(root, file)
            total_files += 1
            subj_file_count += 1
            try:
                arr = np.load(path, mmap_mode='r')  # ✅ Correct usage
            except Exception as e:
                print(f"[UNREADABLE] {path} - {e}")
                bad_files.append(path)
                subj_bad_count += 1
                continue

            if arr.size != EXPECTED_SIZE:
                print(f"[BAD SIZE] {path} - shape: {arr.shape}, size: {arr.size}")
                bad_files.append(path)
                subj_bad_count += 1
            elif arr.shape != EXPECTED_SHAPE:
                print(f"[BAD SHAPE] {path} - shape: {arr.shape}")
                bad_files.append(path)
                subj_bad_count += 1

    total_bad += subj_bad_count
    print(f"[DONE] {subj}: {subj_file_count} files checked, {subj_bad_count} issues found.")

print(f"\n Total files checked: {total_files}")
print(f" Total problematic files: {total_bad}")

