"""
EEG Data Augmentation Script

This script creates augmented versions of EEG CSV files by applying:
1. Frequency wrapping (time axis scaling)
2. Amplitude scaling

Each original sample is augmented 10 times, creating 11 total versions (1 original + 10 augmented).
The augmentation parameters are randomly selected from a uniform distribution U(0.8, 1.2)
for both time scaling and amplitude scaling.

This follows the methodology from BreathRNNet for creating augmented datasets.

Usage:
    python augment_eeg_data.py input_dir output_dir --n_augmentations=10 --verbose=True
    
Input structure:
    input_dir/
        user1_same_session1.csv
        user1_diff_session1.csv
        ...

Output structure:
    output_dir/
        user1_same_session1.csv (original copy)
        user1_same_session1_aug_001.csv
        user1_same_session1_aug_002.csv
        ...
        user1_same_session1_aug_010.csv
        user1_diff_session1.csv (original copy)
        ...
"""

import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import fire


def parse_timestamp(timestamp_str):
    """
    Parse timestamp in MM:SS.S or HH:MM:SS.S format to seconds.
    
    Args:
        timestamp_str: String in format "MM:SS.S" or "HH:MM:SS.S"
    
    Returns:
        Float value in seconds, or None if parsing fails
    """
    try:
        parts = str(timestamp_str).split(':')
        
        if len(parts) == 2:
            # Format: MM:SS.S
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3:
            # Format: HH:MM:SS.S
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        else:
            return None
    except:
        return None


def timestamp_to_string(seconds):
    """
    Convert seconds back to MM:SS.S format.
    
    Args:
        seconds: Float value in seconds
    
    Returns:
        String in format "MM:SS.S"
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:.1f}"


def apply_frequency_wrapping(signal, time_scale_factor):
    """
    Apply frequency wrapping (time axis scaling) to a signal.
    
    Time scaling factor > 1.0 speeds up the signal (compresses time)
    Time scaling factor < 1.0 slows down the signal (stretches time)
    
    Args:
        signal: 1D numpy array of signal values
        time_scale_factor: Scaling factor for time axis (e.g., 0.8 to 1.2)
    
    Returns:
        Scaled signal (resampled)
    """
    from scipy import signal as sp_signal
    
    n_samples_original = len(signal)
    n_samples_new = int(n_samples_original / time_scale_factor)
    
    # Resample to new length
    scaled_signal = sp_signal.resample(signal, n_samples_new)
    
    return scaled_signal


def apply_amplitude_scaling(signal, amplitude_scale_factor):
    """
    Apply amplitude scaling to a signal.
    
    Args:
        signal: 1D numpy array of signal values
        amplitude_scale_factor: Scaling factor for amplitude (e.g., 0.8 to 1.2)
    
    Returns:
        Amplitude-scaled signal
    """
    return signal * amplitude_scale_factor


def augment_eeg_csv(df, time_scale, amplitude_scale, eeg_columns):
    """
    Apply augmentation to a single EEG CSV DataFrame.
    
    Args:
        df: pandas DataFrame with EEG data
        time_scale: Time axis scaling factor
        amplitude_scale: Amplitude scaling factor
        eeg_columns: List of column names containing EEG data
    
    Returns:
        Augmented DataFrame
    """
    df_aug = df.copy()
    
    # Apply frequency wrapping (time scaling) to timestamps
    timestamps = df['Var1'].apply(parse_timestamp).values
    scaled_timestamps = timestamps / time_scale
    df_aug['Var1'] = [timestamp_to_string(t) for t in scaled_timestamps]
    
    # Apply augmentations to EEG channels
    for col in eeg_columns:
        if col in df.columns:
            signal = df[col].values.astype(np.float32)
            
            # Apply frequency wrapping
            signal = apply_frequency_wrapping(signal, time_scale)
            
            # Apply amplitude scaling
            signal = apply_amplitude_scaling(signal, amplitude_scale)
            
            # Handle length mismatch due to resampling
            if len(signal) < len(df_aug):
                # Pad if shorter
                signal = np.pad(signal, (0, len(df_aug) - len(signal)), mode='edge')
            elif len(signal) > len(df_aug):
                # Truncate if longer
                signal = signal[:len(df_aug)]
            
            df_aug[col] = signal
    
    return df_aug


def augment_file_worker(args):
    """
    Worker function to augment a single file multiple times.
    """
    input_file, output_dir, n_augmentations, eeg_columns, verbose = args
    
    input_path = Path(input_file)
    output_path = Path(output_dir)
    file_stem = input_path.stem
    
    try:
        # Load original CSV
        df_original = pd.read_csv(input_file)
        
        # Save original copy
        original_output = output_path / f"{file_stem}.csv"
        df_original.to_csv(original_output, index=False)
        
        files_created = [original_output]
        
        # Create augmented versions
        for aug_idx in range(1, n_augmentations + 1):
            # Sample augmentation parameters from U(0.8, 1.2)
            time_scale = np.random.uniform(0.8, 1.2)
            amplitude_scale = np.random.uniform(0.8, 1.2)
            
            # Apply augmentation
            df_aug = augment_eeg_csv(df_original, time_scale, amplitude_scale, eeg_columns)
            
            # Save augmented file
            aug_output = output_path / f"{file_stem}_aug_{aug_idx:03d}.csv"
            df_aug.to_csv(aug_output, index=False)
            files_created.append(aug_output)
            
            if verbose:
                print(f"  {file_stem}_aug_{aug_idx:03d}.csv: time_scale={time_scale:.3f}, amp_scale={amplitude_scale:.3f}")
        
        return f"{file_stem}: created {len(files_created)} files (1 original + {n_augmentations} augmented)"
    
    except Exception as e:
        return f"{file_stem}: ERROR - {str(e)}"


def augment_dataset(input_dir, output_dir, n_augmentations=10, 
                    eeg_columns=None, suffix='.csv', verbose=False, seed=None):
    """
    Augment all EEG CSV files in input directory.
    
    Args:
        input_dir: Directory containing original CSV files
        output_dir: Directory to save original + augmented CSV files
        n_augmentations: Number of augmented versions to create per file (default: 10)
        eeg_columns: List of column names containing EEG data. If None, uses RAW channels.
        suffix: File extension to process (default: '.csv')
        verbose: Print detailed progress
        seed: Random seed for reproducibility (default: None)
    """
    if seed is not None:
        np.random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default to RAW EEG channels if not specified
    if eeg_columns is None:
        eeg_columns = ['Var22', 'Var23', 'Var24', 'Var25']  # RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10
    
    print(f"EEG Data Augmentation")
    print(f"=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Augmentations per file: {n_augmentations}")
    print(f"Total files per original: {n_augmentations + 1} (1 original + {n_augmentations} augmented)")
    print(f"Augmentation method:")
    print(f"  - Frequency wrapping (time scaling): U(0.8, 1.2)")
    print(f"  - Amplitude scaling: U(0.8, 1.2)")
    print(f"EEG columns to augment: {eeg_columns}")
    if seed is not None:
        print(f"Random seed: {seed}")
    print()
    
    # Find all CSV files
    csv_files = list(input_path.glob(f'*{suffix}'))
    csv_files = sorted(csv_files)
    
    if len(csv_files) == 0:
        print(f"ERROR: No {suffix} files found in {input_path}")
        return
    
    print(f"Found {len(csv_files)} files to augment")
    print()
    
    # Prepare worker arguments
    worker_args = []
    for csv_file in csv_files:
        worker_args.append([
            str(csv_file),
            str(output_path),
            n_augmentations,
            eeg_columns,
            verbose
        ])
    
    # Process files with multiprocessing
    print(f"Processing files...")
    with Pool() as p:
        results = list(tqdm(p.imap(augment_file_worker, worker_args), total=len(worker_args)))
    
    # Print results
    print()
    print("Results:")
    print("-" * 60)
    successful = 0
    failed = 0
    for result in results:
        if "ERROR" in result:
            print(f"❌ {result}")
            failed += 1
        else:
            if verbose:
                print(f"✓ {result}")
            successful += 1
    
    print()
    print("=" * 60)
    print(f"Summary:")
    print(f"  Original files processed: {len(csv_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total output files: {successful * (n_augmentations + 1)}")
    print(f"  Data augmentation factor: {n_augmentations + 1}x")
    print("=" * 60)


if __name__ == "__main__":
    fire.Fire(augment_dataset)
