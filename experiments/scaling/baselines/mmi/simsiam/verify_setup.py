#!/usr/bin/env python3
"""
Verification script to check if the updated files are installed correctly
Run this BEFORE starting your experiments to catch issues early
"""

import os
import sys

def check_file(filepath, required_strings, file_description):
    """Check if a file contains required strings"""
    print(f"\nChecking {file_description}...")
    print(f"File: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  ✗ FAIL: File not found!")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    all_found = True
    for req_string, description in required_strings:
        if req_string in content:
            print(f"  ✓ Found: {description}")
        else:
            print(f"  ✗ MISSING: {description}")
            all_found = False
    
    if all_found:
        print(f"  ✓ {file_description} is CORRECT")
    else:
        print(f"  ✗ {file_description} is INCORRECT or OUTDATED")
    
    return all_found


def main():
    print("="*60)
    print("EEG CLASSIFICATION - SETUP VERIFICATION")
    print("="*60)
    
    base_dir = "/app/scaling/baselines/simsiam"
    
    # Check data_loader.py
    data_loader_checks = [
        ('def augment_data_in_memory(x_data, y_data):', 'augment_data_in_memory function'),
        ('✓ AUGMENTATION SUMMARY:', 'Augmentation summary message'),
        ('Augmented samples:  {x_augmented.shape[0]} (4x)', '4x augmentation confirmation'),
        ('use_chunked_loading = num_users > 70', 'Chunked loading for large datasets'),
    ]
    
    data_loader_ok = check_file(
        os.path.join(base_dir, 'data_loader.py'),
        data_loader_checks,
        'data_loader.py'
    )
    
    # Check trainers.py
    trainers_checks = [
        ('train_ds, val_ds, test_ds, steps = data_load_with_tf_datasets', 'Uses TF datasets'),
        ('test_results = model.evaluate(test_ds', 'Evaluates with test_ds (not test_gen)'),
        ('for _, labels in test_ds:', 'Collects labels from test_ds'),
    ]
    
    trainers_ok = check_file(
        os.path.join(base_dir, 'trainers.py'),
        trainers_checks,
        'trainers.py'
    )
    
    # Check transformations.py
    transformations_checks = [
        ("if hasattr(X, 'numpy'):", 'Numpy safety checks'),
        ('def DA_MagWarp(X, sigma=0.3):', 'MagWarp augmentation'),
        ('def DA_Scaling(X, sigma=0.1):', 'Scaling augmentation'),
        ('def DA_Negation(X, sigma=0):', 'Negation augmentation'),
    ]
    
    transformations_ok = check_file(
        os.path.join(base_dir, 'transformations.py'),
        transformations_checks,
        'transformations.py'
    )
    
    # Overall result
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_ok = data_loader_ok and trainers_ok and transformations_ok
    
    if all_ok:
        print("✓ ALL FILES ARE CORRECT!")
        print("\nYou can now run your experiments:")
        print("  cd /app/scaling/baselines/simsiam")
        print("  python3 plot_loop.py")
        print("\nExpected output for 10 users:")
        print("  - 'Augmented samples: 113880 (4x)' during loading")
        print("  - Training accuracy: ~95%")
        print("  - Validation accuracy: ~90%")
        print("  - Test accuracy: ~92%")
    else:
        print("✗ SOME FILES ARE INCORRECT!")
        print("\nYou need to replace the following files:")
        if not data_loader_ok:
            print("  - data_loader.py (CRITICAL - affects accuracy!)")
        if not trainers_ok:
            print("  - trainers.py")
        if not transformations_ok:
            print("  - transformations.py")
        print("\nGet the complete files from the artifacts provided by Claude.")
    
    print("="*60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
