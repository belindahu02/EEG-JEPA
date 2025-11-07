#!/usr/bin/env python3
"""
Compare two directories containing nested .npy files.
Checks for missing files and functional (not just exact) equivalence.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


def get_npy_files(directory: Path) -> Dict[str, Path]:
    """Get all .npy files in directory and subdirectories."""
    npy_files = {}
    for npy_file in directory.rglob("*.npy"):
        relative_path = npy_file.relative_to(directory)
        npy_files[str(relative_path)] = npy_file
    return npy_files


def compare_arrays(file1: Path, file2: Path, atol=1e-6, corr_threshold=0.99) -> Tuple[bool, str]:
    """
    Compare two .npy files for functional equivalence.
    Returns (are_equivalent, message).
    """
    try:
        arr1 = np.load(file1)
        arr2 = np.load(file2)
        
        if arr1.shape != arr2.shape:
            return False, f"Different shapes: {arr1.shape} vs {arr2.shape}"
        if arr1.dtype != arr2.dtype:
            return False, f"Different dtypes: {arr1.dtype} vs {arr2.dtype}"

        arr1 = arr1.astype(np.float32).flatten()
        arr2 = arr2.astype(np.float32).flatten()
        
        # Compute difference stats
        abs_diff = np.abs(arr1 - arr2)
        mean_abs_diff = np.mean(abs_diff)
        max_diff = np.max(abs_diff)
        mean_rel_diff = np.mean(abs_diff / (np.abs(arr1) + 1e-8))

        # Compute correlation (if not constant arrays)
        if np.std(arr1) > 0 and np.std(arr2) > 0:
            corr = np.corrcoef(arr1, arr2)[0, 1]
        else:
            corr = 1.0 if np.allclose(arr1, arr2, atol=atol) else 0.0

        # Determine functional equivalence
        if corr >= corr_threshold:
            return True, f"Equivalent (corr={corr:.4f}, mean_abs_diff={mean_abs_diff:.3e}, max_diff={max_diff:.3e})"
        else:
            return False, f"Different (corr={corr:.4f}, mean_abs_diff={mean_abs_diff:.3e}, max_diff={max_diff:.3e})"

    except Exception as e:
        return False, f"Error loading files: {str(e)}"


def compare_directories(dir1: Path, dir2: Path) -> None:
    """Compare two directories containing .npy files."""
    print(f"\n{'='*70}")
    print(f"Comparing directories:")
    print(f"  Directory 1: {dir1}")
    print(f"  Directory 2: {dir2}")
    print(f"{'='*70}\n")

    files1 = get_npy_files(dir1)
    files2 = get_npy_files(dir2)

    all_files = set(files1.keys()) | set(files2.keys())
    common_files = set(files1.keys()) & set(files2.keys())
    only_in_dir1 = set(files1.keys()) - set(files2.keys())
    only_in_dir2 = set(files2.keys()) - set(files1.keys())

    print(f"📊 SUMMARY")
    print(f"  Total unique files: {len(all_files)}")
    print(f"  Files in both directories: {len(common_files)}")
    print(f"  Files only in directory 1: {len(only_in_dir1)}")
    print(f"  Files only in directory 2: {len(only_in_dir2)}\n")

    if only_in_dir1:
        print(f"⚠️  Missing in directory 2:")
        for f in sorted(only_in_dir1):
            print(f"  ❌ {f}")
        print()
    if only_in_dir2:
        print(f"⚠️  Missing in directory 1:")
        for f in sorted(only_in_dir2):
            print(f"  ❌ {f}")
        print()

    if not common_files:
        print("No common files to compare.\n")
        return

    print(f"🔍 Comparing {len(common_files)} common files...\n")

    equivalent_count = 0
    different_files = []

    for i, rel_path in enumerate(sorted(common_files), 1):
        file1, file2 = files1[rel_path], files2[rel_path]
        is_equiv, msg = compare_arrays(file1, file2)

        if is_equiv:
            equivalent_count += 1
        else:
            different_files.append((rel_path, msg))

        if i % 100 == 0:
            print(f"  Progress: {i}/{len(common_files)} files checked...")

    print()
    print(f"✅ FUNCTIONALLY EQUIVALENT FILES: {equivalent_count}/{len(common_files)}")

    if different_files:
        print(f"❌ NON-EQUIVALENT FILES: {len(different_files)}")
        for rel_path, msg in different_files[:20]:  # limit output
            print(f"  ❌ {rel_path}")
            print(f"     → {msg}")
        if len(different_files) > 20:
            print(f"  ... ({len(different_files) - 20} more not shown)")
    print()

    print(f"{'='*70}")
    if not only_in_dir1 and not only_in_dir2 and not different_files:
        print("✅ RESULT: Directories are FUNCTIONALLY IDENTICAL!")
    elif equivalent_count == len(common_files):
        print("✅ RESULT: All common files are FUNCTIONALLY EQUIVALENT!")
    else:
        print("⚠️ RESULT: Some files differ significantly.")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two directories containing nested .npy files (functional equivalence check)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_dirs_functional.py /path/to/dir1 /path/to/dir2
  python compare_dirs_functional.py ./spec_v1 ./spec_v2
        """
    )
    parser.add_argument("dir1", type=str, help="First directory path")
    parser.add_argument("dir2", type=str, help="Second directory path")
    args = parser.parse_args()

    dir1, dir2 = Path(args.dir1), Path(args.dir2)
    if not dir1.exists() or not dir2.exists():
        print("❌ One or both directories do not exist.")
        return 1
    if not dir1.is_dir() or not dir2.is_dir():
        print("❌ One or both paths are not directories.")
        return 1

    compare_directories(dir1, dir2)
    return 0


if __name__ == "__main__":
    exit(main())

