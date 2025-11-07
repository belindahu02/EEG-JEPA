"""
EEG Data Augmentation Script - Identity-Preserving for Biometrics

This script creates augmented versions of EEG CSV files using identity-preserving
augmentations suitable for biometric identification tasks. Unlike traditional
augmentations that modify frequency or amplitude, these augmentations preserve
user-specific signal characteristics while adding realistic variability.

Augmentation techniques applied:
1. Additive Gaussian noise (preserves signal morphology)
2. Baseline wander (simulates electrode drift)
3. Random segment dropout/masking (simulates artifacts)
4. Channel-specific noise (simulates electrode impedance changes)
5. Temporal jitter (simulates timing variations)

Each original sample is augmented N times, creating N+1 total versions 
(1 original + N augmented).

Usage:
    python augment_eeg_biometric.py input_dir output_dir --n_augmentations=10 --verbose=True
    
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
"""

import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import fire
from scipy import signal as sp_signal


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


def add_gaussian_noise(signal, noise_level_range=(0.01, 0.05)):
    """
    Add Gaussian noise to signal. Preserves signal morphology and user-specific patterns.
    
    Args:
        signal: 1D numpy array of signal values
        noise_level_range: Tuple of (min, max) noise level as fraction of signal std
    
    Returns:
        Noisy signal
    """
    noise_level = np.random.uniform(*noise_level_range)
    signal_std = np.std(signal)
    noise = np.random.normal(0, noise_level * signal_std, len(signal))
    return signal + noise


def add_baseline_wander(signal, sampling_rate=160, freq_range=(0.1, 0.5), 
                       amplitude_range=(0.01, 0.05)):
    """
    Add baseline wander to simulate electrode drift. Common artifact in EEG recordings.
    
    Args:
        signal: 1D numpy array of signal values
        sampling_rate: Sampling rate in Hz
        freq_range: Tuple of (min, max) frequency for baseline wander in Hz
        amplitude_range: Tuple of (min, max) amplitude as fraction of signal std
    
    Returns:
        Signal with baseline wander
    """
    freq = np.random.uniform(*freq_range)
    amplitude = np.random.uniform(*amplitude_range) * np.std(signal)
    
    t = np.arange(len(signal)) / sampling_rate
    baseline = amplitude * np.sin(2 * np.pi * freq * t)
    
    return signal + baseline


def apply_segment_dropout(signal, dropout_prob=0.3, segment_length_range=(0.05, 0.15)):
    """
    Randomly zero out segments of the signal to simulate artifacts or lost data.
    
    Args:
        signal: 1D numpy array of signal values
        dropout_prob: Probability of applying dropout
        segment_length_range: Tuple of (min, max) segment length as fraction of signal length
    
    Returns:
        Signal with potential segment dropout
    """
    if np.random.random() > dropout_prob:
        return signal
    
    signal = signal.copy()
    segment_length_frac = np.random.uniform(*segment_length_range)
    segment_length = int(segment_length_frac * len(signal))
    
    if segment_length >= len(signal):
        return signal
    
    dropout_start = np.random.randint(0, len(signal) - segment_length)
    signal[dropout_start:dropout_start + segment_length] = 0
    
    return signal


def apply_temporal_jitter(signal, max_shift_samples=5):
    """
    Apply small temporal shifts to simulate timing variations.
    
    Args:
        signal: 1D numpy array of signal values
        max_shift_samples: Maximum number of samples to shift
    
    Returns:
        Temporally jittered signal
    """
    if max_shift_samples == 0:
        return signal
    
    shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
    
    if shift > 0:
        # Shift right (delay)
        return np.pad(signal[:-shift] if shift < len(signal) else signal[:0], 
                     (shift, 0), mode='edge')
    elif shift < 0:
        # Shift left (advance)
        return np.pad(signal[-shift:], (0, -shift), mode='edge')
    else:
        return signal


def apply_channel_specific_noise(signal, noise_level_range=(0.005, 0.02)):
    """
    Add channel-specific noise to simulate electrode impedance changes.
    Different from additive noise - this is multiplicative variation.
    
    Args:
        signal: 1D numpy array of signal values
        noise_level_range: Tuple of (min, max) noise level
    
    Returns:
        Signal with channel-specific noise
    """
    noise_factor = 1.0 + np.random.uniform(*noise_level_range) * np.random.randn()
    return signal * noise_factor


def apply_powerline_interference(signal, sampling_rate=160, freq=50, 
                                amplitude_range=(0.005, 0.02)):
    """
    Add powerline interference (50 or 60 Hz) to simulate electrical noise.
    
    Args:
        signal: 1D numpy array of signal values
        sampling_rate: Sampling rate in Hz
        freq: Powerline frequency (50 or 60 Hz)
        amplitude_range: Tuple of (min, max) amplitude as fraction of signal std
    
    Returns:
        Signal with powerline interference
    """
    amplitude = np.random.uniform(*amplitude_range) * np.std(signal)
    t = np.arange(len(signal)) / sampling_rate
    interference = amplitude * np.sin(2 * np.pi * freq * t)
    return signal + interference


def augment_eeg_signal(signal, sampling_rate=160, aug_config=None):
    """
    Apply identity-preserving augmentations to a single EEG channel.
    
    Args:
        signal: 1D numpy array of signal values
        sampling_rate: Sampling rate in Hz
        aug_config: Dictionary of augmentation parameters (None = use defaults)
    
    Returns:
        Augmented signal
    """
    if aug_config is None:
        aug_config = {
            'gaussian_noise': True,
            'baseline_wander': True,
            'segment_dropout': True,
            'temporal_jitter': True,
            'channel_noise': True,
            'powerline_interference': False,  # Optional, can be noisy
        }
    
    # Start with original signal
    augmented = signal.copy()
    
    # Apply augmentations
    if aug_config.get('gaussian_noise', True):
        augmented = add_gaussian_noise(augmented, noise_level_range=(0.01, 0.05))
    
    if aug_config.get('baseline_wander', True):
        augmented = add_baseline_wander(augmented, sampling_rate=sampling_rate,
                                       freq_range=(0.1, 0.5),
                                       amplitude_range=(0.01, 0.05))
    
    if aug_config.get('segment_dropout', True):
        augmented = apply_segment_dropout(augmented, dropout_prob=0.3,
                                         segment_length_range=(0.05, 0.15))
    
    if aug_config.get('temporal_jitter', True):
        augmented = apply_temporal_jitter(augmented, max_shift_samples=5)
    
    if aug_config.get('channel_noise', True):
        augmented = apply_channel_specific_noise(augmented, 
                                                noise_level_range=(0.005, 0.02))
    
    if aug_config.get('powerline_interference', False):
        # Randomly choose 50 or 60 Hz
        freq = 50 if np.random.random() < 0.5 else 60
        augmented = apply_powerline_interference(augmented, sampling_rate=sampling_rate,
                                                freq=freq,
                                                amplitude_range=(0.005, 0.02))
    
    return augmented


def augment_eeg_csv(df, eeg_columns, sampling_rate=160, aug_config=None):
    """
    Apply identity-preserving augmentation to a single EEG CSV DataFrame.
    
    Args:
        df: pandas DataFrame with EEG data
        eeg_columns: List of column names containing EEG data
        sampling_rate: Sampling rate in Hz
        aug_config: Dictionary of augmentation parameters
    
    Returns:
        Augmented DataFrame
    """
    df_aug = df.copy()
    
    # Apply augmentations to each EEG channel
    for col in eeg_columns:
        if col in df.columns:
            signal = df[col].values.astype(np.float32)
            augmented_signal = augment_eeg_signal(signal, sampling_rate, aug_config)
            df_aug[col] = augmented_signal
    
    # Timestamps remain unchanged (no time scaling)
    
    return df_aug


def augment_file_worker(args):
    """
    Worker function to augment a single file multiple times.
    """
    input_file, output_dir, n_augmentations, eeg_columns, sampling_rate, aug_config, verbose = args
    
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
            # Apply augmentation
            df_aug = augment_eeg_csv(df_original, eeg_columns, sampling_rate, aug_config)
            
            # Save augmented file
            aug_output = output_path / f"{file_stem}_aug_{aug_idx:03d}.csv"
            df_aug.to_csv(aug_output, index=False)
            files_created.append(aug_output)
            
            if verbose:
                print(f"  {file_stem}_aug_{aug_idx:03d}.csv: identity-preserving augmentation applied")
        
        return f"{file_stem}: created {len(files_created)} files (1 original + {n_augmentations} augmented)"
    
    except Exception as e:
        return f"{file_stem}: ERROR - {str(e)}"


def augment_dataset(input_dir, output_dir, n_augmentations=10, 
                    eeg_columns=None, sampling_rate=160,
                    gaussian_noise=True, baseline_wander=True,
                    segment_dropout=True, temporal_jitter=True,
                    channel_noise=True, powerline_interference=False,
                    suffix='.csv', verbose=False, seed=None):
    """
    Augment all EEG CSV files in input directory with identity-preserving augmentations.
    
    Args:
        input_dir: Directory containing original CSV files
        output_dir: Directory to save original + augmented CSV files
        n_augmentations: Number of augmented versions to create per file (default: 10)
        eeg_columns: List of column names containing EEG data. If None, uses RAW channels.
        sampling_rate: Sampling rate in Hz (default: 160)
        gaussian_noise: Apply Gaussian noise augmentation
        baseline_wander: Apply baseline wander augmentation
        segment_dropout: Apply segment dropout augmentation
        temporal_jitter: Apply temporal jitter augmentation
        channel_noise: Apply channel-specific noise augmentation
        powerline_interference: Apply powerline interference augmentation
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
    
    # Augmentation configuration
    aug_config = {
        'gaussian_noise': gaussian_noise,
        'baseline_wander': baseline_wander,
        'segment_dropout': segment_dropout,
        'temporal_jitter': temporal_jitter,
        'channel_noise': channel_noise,
        'powerline_interference': powerline_interference,
    }
    
    print(f"EEG Data Augmentation - Identity-Preserving for Biometrics")
    print(f"=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Augmentations per file: {n_augmentations}")
    print(f"Total files per original: {n_augmentations + 1} (1 original + {n_augmentations} augmented)")
    print(f"\nAugmentation methods:")
    print(f"  - Gaussian noise: {'✓' if gaussian_noise else '✗'} (preserves morphology)")
    print(f"  - Baseline wander: {'✓' if baseline_wander else '✗'} (simulates electrode drift)")
    print(f"  - Segment dropout: {'✓' if segment_dropout else '✗'} (simulates artifacts)")
    print(f"  - Temporal jitter: {'✓' if temporal_jitter else '✗'} (simulates timing variations)")
    print(f"  - Channel-specific noise: {'✓' if channel_noise else '✗'} (simulates impedance changes)")
    print(f"  - Powerline interference: {'✓' if powerline_interference else '✗'} (50/60 Hz noise)")
    print(f"\nEEG columns to augment: {eeg_columns}")
    print(f"Sampling rate: {sampling_rate} Hz")
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
            sampling_rate,
            aug_config,
            verbose
        ])
    
    # Process files with multiprocessing
    print(f"Processing files...")
    with Pool() as p:
        results = list(tqdm(p.imap(augment_file_worker, worker_args), total=len(worker_args)))
    
    # Print results
    print()
    print("Results:")
    print("-" * 70)
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
    print("=" * 70)
    print(f"Summary:")
    print(f"  Original files processed: {len(csv_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total output files: {successful * (n_augmentations + 1)}")
    print(f"  Data augmentation factor: {n_augmentations + 1}x")
    print(f"\nAugmentation approach: Identity-preserving (suitable for biometrics)")
    print(f"These augmentations preserve user-specific signal characteristics")
    print(f"while adding realistic variability from recording conditions.")
    print("=" * 70)


if __name__ == "__main__":
    fire.Fire(augment_dataset)
