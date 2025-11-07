import pandas as pd
import numpy as np
import os
from transformations import *
import mne
import gc


def load_edf_file(filepath, max_samples=None):
    """
    Load EDF file and extract EEG data with optional downsampling
    
    Args:
        filepath: Path to EDF file
        max_samples: If set, downsample to this many samples max
    """
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        data = raw.get_data().T  # Transpose to (samples, channels)

        if max_samples is not None and data.shape[0] > max_samples:
            step = data.shape[0] // max_samples
            data = data[::step]

        # Explicitly delete raw to free memory
        del raw
        
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def compute_normalization_stats(path, users, sessions, frame_size=30, max_samples_per_session=None):
    """
    Compute mean and std statistics from training data for normalization.
    Memory efficient single-pass computation.
    
    Returns:
        mean, std: Arrays of shape (n_channels,)
    """
    print("Computing normalization statistics...")

    n_samples = 0
    sum_x = None
    sum_x2 = None
    files_processed = 0

    for user in users:
        user_folder = f"S{user:03d}"

        for session in sessions:
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(path, user_folder, filename)

            try:
                data = load_edf_file(filepath, max_samples=max_samples_per_session)
                if data is None or data.shape[0] < frame_size:
                    continue

                # Truncate to multiple of frame_size
                data = data[:(data.shape[0] // frame_size) * frame_size]

                if data.shape[0] == 0:
                    continue

                # Initialize arrays on first valid data
                if sum_x is None:
                    sum_x = np.zeros(data.shape[1])
                    sum_x2 = np.zeros(data.shape[1])

                # Update running statistics
                sum_x += np.sum(data, axis=0)
                sum_x2 += np.sum(data ** 2, axis=0)
                n_samples += data.shape[0]
                files_processed += 1

                del data
                
                # Periodic garbage collection
                if files_processed % 20 == 0:
                    gc.collect()

            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

    if sum_x is None or n_samples == 0:
        raise ValueError("No valid data found for normalization!")

    # Compute mean and std
    mean = sum_x / n_samples
    std = np.sqrt(sum_x2 / n_samples - mean ** 2)
    std = np.where(std == 0, 1, std)  # Avoid division by zero

    print(f"Computed stats from {n_samples:,} samples across {files_processed} files")
    return mean, std


class EEGDataGenerator:
    """
    Memory-efficient generator for streaming EEG data in batches
    Optimized for large datasets with 109 users and all sessions
    """

    def __init__(self, path, users, sessions, frame_size=30, batch_size=32,
                 max_samples_per_session=None, mean=None, std=None, shuffle=True):
        """
        Args:
            path: Base path to dataset
            users: List of user IDs
            sessions: List of session numbers
            frame_size: Window size for sliding windows
            batch_size: Number of samples per batch
            max_samples_per_session: Limit samples per session (None = use all)
            mean, std: Normalization statistics
            shuffle: Whether to shuffle data each epoch
        """
        self.path = path
        self.users = users
        self.sessions = sessions
        self.frame_size = frame_size
        self.batch_size = batch_size
        self.max_samples_per_session = max_samples_per_session
        self.mean = mean
        self.std = std
        self.shuffle = shuffle

        # Build file list
        self.file_list = []
        for user_id, user in enumerate(users):
            user_folder = f"S{user:03d}"
            for session in sessions:
                filename = f"S{user:03d}R{session:02d}.edf"
                filepath = os.path.join(path, user_folder, filename)
                if os.path.exists(filepath):
                    self.file_list.append((filepath, user_id))

        self.n_files = len(self.file_list)
        print(f"Generator initialized with {self.n_files} files")
        
        if max_samples_per_session is None:
            print("⚠️  Using ALL available samples (no limit)")
        else:
            print(f"Limiting to {max_samples_per_session} samples per session")

    def _process_file(self, filepath, user_id):
        """Process a single file and return windows - MEMORY EFFICIENT"""
        data = load_edf_file(filepath, max_samples=self.max_samples_per_session)
        if data is None or data.shape[0] < self.frame_size:
            return None, None

        # Truncate to multiple of frame_size
        data = data[:(data.shape[0] // self.frame_size) * self.frame_size]

        if data.shape[0] == 0:
            return None, None

        # Normalize BEFORE creating windows to save memory
        if self.mean is not None and self.std is not None:
            data = (data - self.mean) / self.std

        # Create sliding windows with 50% overlap
        # Using stride_tricks is memory efficient (creates view, not copy)
        windows = np.lib.stride_tricks.sliding_window_view(
            data, (self.frame_size, data.shape[1])
        )[::self.frame_size // 2, :]

        # Reshape if needed
        if len(windows.shape) == 4:
            windows = windows.squeeze(axis=1)
        elif len(windows.shape) != 3:
            print(f"Warning: Unexpected shape {windows.shape}")
            return None, None

        # Copy to contiguous array for efficiency
        windows = np.ascontiguousarray(windows, dtype=np.float32)
        labels = np.full(windows.shape[0], user_id, dtype=np.int32)

        # Free original data immediately
        del data
        
        return windows, labels

    def __iter__(self):
        """Iterator that yields batches - INFINITE LOOP for multiple epochs"""
        while True:
            # Shuffle file order if requested
            file_indices = np.arange(self.n_files)
            if self.shuffle:
                np.random.shuffle(file_indices)

            batch_x = []
            batch_y = []
            files_processed = 0

            for idx in file_indices:
                filepath, user_id = self.file_list[idx]

                try:
                    windows, labels = self._process_file(filepath, user_id)

                    if windows is None:
                        continue

                    # Shuffle windows within file if requested
                    if self.shuffle:
                        perm = np.random.permutation(len(windows))
                        windows = windows[perm]
                        labels = labels[perm]

                    # Add to batch
                    for i in range(len(windows)):
                        batch_x.append(windows[i])
                        batch_y.append(labels[i])

                        if len(batch_x) == self.batch_size:
                            # Convert to arrays and yield
                            yield (np.array(batch_x, dtype=np.float32), 
                                   np.array(batch_y, dtype=np.int32))
                            batch_x = []
                            batch_y = []

                    del windows, labels
                    files_processed += 1
                    
                    # Aggressive garbage collection for large datasets
                    if files_processed % 10 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    continue

            # Yield remaining samples at end of epoch
            if len(batch_x) > 0:
                yield (np.array(batch_x, dtype=np.float32), 
                       np.array(batch_y, dtype=np.int32))

            # Clean up between epochs
            gc.collect()

    def get_steps_per_epoch(self):
        """
        Estimate number of batches per epoch
        FAST estimation - doesn't load all data
        """
        # For speed, we'll estimate based on first few files
        sample_size = min(10, self.n_files)
        avg_samples_per_file = 0
        valid_files = 0

        for i in range(sample_size):
            filepath, _ = self.file_list[i]
            try:
                data = load_edf_file(filepath, max_samples=self.max_samples_per_session)
                if data is not None and data.shape[0] >= self.frame_size:
                    n_samples = (data.shape[0] // self.frame_size) * self.frame_size
                    n_windows = len(range(0, n_samples - self.frame_size + 1, 
                                         self.frame_size // 2))
                    avg_samples_per_file += n_windows
                    valid_files += 1
                del data
            except:
                continue

        if valid_files == 0:
            return 100  # Default fallback

        avg_samples_per_file /= valid_files
        total_samples = int(avg_samples_per_file * self.n_files)
        steps = max(1, total_samples // self.batch_size)
        
        gc.collect()
        return steps


def data_load_with_generators(path, users, frame_size=30, batch_size=32,
                              max_samples_per_session=None):
    """
    Create data generators for train/val/test splits with proper normalization.
    
    Args:
        path: Base path to dataset
        users: List of user IDs
        frame_size: Window size
        batch_size: Batch size per replica
        max_samples_per_session: Limit samples (None = use all)
    
    Returns:
        train_gen, val_gen, test_gen, steps_per_epoch dict
    """
    # Session-level split
    train_sessions = list(range(1, 11))  # R01-R10
    val_sessions = [11, 12]  # R11-R12
    test_sessions = [13, 14]  # R13-R14

    print(f"\n{'='*60}")
    print("DATA LOADING CONFIGURATION:")
    print(f"Users: {len(users)} (ID {users[0]} to {users[-1]})")
    print(f"Train sessions: {train_sessions}")
    print(f"Val sessions: {val_sessions}")
    print(f"Test sessions: {test_sessions}")
    print(f"Frame size: {frame_size}")
    print(f"Batch size: {batch_size}")
    if max_samples_per_session is None:
        print("Using ALL available samples (no limit)")
    else:
        print(f"Max samples per session: {max_samples_per_session}")
    print(f"{'='*60}\n")

    # Compute normalization from training data only
    print("Computing normalization statistics from training data...")
    mean, std = compute_normalization_stats(
        path, users, train_sessions, frame_size, max_samples_per_session
    )

    # Create generators
    print("Creating training generator...")
    train_gen = EEGDataGenerator(
        path, users, train_sessions, frame_size, batch_size,
        max_samples_per_session, mean, std, shuffle=True
    )

    print("Creating validation generator...")
    val_gen = EEGDataGenerator(
        path, users, val_sessions, frame_size, batch_size,
        max_samples_per_session, mean, std, shuffle=False
    )

    print("Creating test generator...")
    test_gen = EEGDataGenerator(
        path, users, test_sessions, frame_size, batch_size,
        max_samples_per_session, mean, std, shuffle=False
    )

    # Get steps per epoch
    print("Estimating steps per epoch...")
    steps = {
        'train': train_gen.get_steps_per_epoch(),
        'val': val_gen.get_steps_per_epoch(),
        'test': test_gen.get_steps_per_epoch()
    }

    print(f"\nEstimated steps per epoch:")
    print(f"  Train: {steps['train']}")
    print(f"  Val: {steps['val']}")
    print(f"  Test: {steps['test']}")

    return train_gen, val_gen, test_gen, steps


class AugmentedEEGDataGenerator:
    """Generator that applies data augmentation on-the-fly"""

    def __init__(self, base_generator, transformations, sigma_l, ext=False):
        self.base_generator = base_generator
        self.transformations = transformations
        self.sigma_l = sigma_l
        self.ext = ext
        self.n_transforms = len(transformations)

    def __iter__(self):
        """Yields augmented batches"""
        for batch_x, batch_y in self.base_generator:
            for i, (transform, sigma) in enumerate(zip(self.transformations, self.sigma_l)):
                # Original samples (negative)
                yield batch_x, np.zeros(len(batch_x), dtype=bool)

                # Augmented samples (positive)
                augmented = np.array([transform(x, sigma=sigma) for x in batch_x])
                yield augmented, np.ones(len(batch_x), dtype=bool)

                # Extended augmentation
                if self.ext:
                    for j, (other_transform, other_sigma) in enumerate(zip(self.transformations, self.sigma_l)):
                        if i != j:
                            other_augmented = np.array([other_transform(x, sigma=other_sigma) 
                                                       for x in batch_x])
                            yield other_augmented, np.zeros(len(batch_x), dtype=bool)
