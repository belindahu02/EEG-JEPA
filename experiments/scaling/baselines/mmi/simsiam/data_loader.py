import pandas as pd
import numpy as np
import os
import mne
import gc
import tensorflow as tf


def standardize(data):
    """Manual standardization to replace sklearn StandardScaler"""
    mean = np.mean(data, axis=0, dtype=np.float32)
    std = np.std(data, axis=0, dtype=np.float32)
    std[std == 0] = 1
    return (data - mean) / std, mean, std


def apply_standardization(data, mean, std):
    """Apply pre-computed standardization"""
    return (data - mean) / std


def load_edf_file(filepath, max_channels=64):
    """Load a single EDF file efficiently"""
    try:
        raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
        eeg_channels = [ch for ch in raw.ch_names if not ch.startswith('EOG')]
        raw.pick_channels(eeg_channels[:max_channels])
        raw.load_data()
        data = raw.get_data().T.astype(np.float32)
        del raw
        gc.collect()
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def create_windows(data, frame_size=30, overlap=0.5):
    """Create sliding windows efficiently using stride tricks"""
    if data.shape[0] < frame_size:
        return None

    stride = int(frame_size * (1 - overlap))
    n_windows = (data.shape[0] - frame_size) // stride + 1
    shape = (n_windows, frame_size, data.shape[1])
    strides = (stride * data.strides[0], data.strides[0], data.strides[1])
    windowed = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return windowed.copy()


def load_all_data_to_memory(path, users, folders, frame_size=30):
    """
    Load ALL data for specified users into memory at once.
    This is much faster than on-the-fly loading during training.
    
    Returns:
        x_data: numpy array of all windows
        y_data: numpy array of all labels
    """
    train_runs = list(range(1, 11))  # R01-R10
    val_runs = [11, 12]  # R11-R12
    test_runs = [13, 14]  # R13-R14

    if 'TrainingSet' in folders:
        use_runs = train_runs
    elif 'TestingSet' in folders:
        use_runs = val_runs
    elif 'TestingSet_secret' in folders:
        use_runs = test_runs
    else:
        use_runs = train_runs

    all_data = []
    all_labels = []
    
    print(f"Loading data for {len(users)} users...")
    
    for user_id, user in enumerate(users):
        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)
        
        if not os.path.exists(user_path):
            print(f"Warning: User folder {user_path} not found")
            continue
        
        user_windows = []
        file_count = 0
        
        for run in use_runs:
            filename = f"S{user:03d}R{run:02d}.edf"
            filepath = os.path.join(user_path, filename)
            
            if not os.path.exists(filepath):
                continue
            
            # Load file
            data = load_edf_file(filepath)
            if data is None:
                continue
            
            # Create windows
            windowed = create_windows(data, frame_size=frame_size, overlap=0.5)
            del data
            gc.collect()
            
            if windowed is not None and windowed.shape[0] > 0:
                user_windows.append(windowed)
                file_count += 1
            
            del windowed
            gc.collect()
        
        if len(user_windows) > 0:
            # Concatenate all runs for this user
            user_data = np.concatenate(user_windows, axis=0)
            all_data.append(user_data)
            all_labels.extend([user_id] * user_data.shape[0])
            
            print(f"  User {user:03d}: {user_data.shape[0]:6d} samples from {file_count} sessions")
            
            del user_windows, user_data
            gc.collect()
        
        # Periodic garbage collection every 10 users
        if (user_id + 1) % 10 == 0:
            gc.collect()
    
    if len(all_data) == 0:
        raise ValueError("No data loaded!")
    
    # Concatenate all users
    x_data = np.concatenate(all_data, axis=0)
    y_data = np.array(all_labels, dtype=np.int32)
    
    del all_data, all_labels
    gc.collect()
    
    print(f"Total samples loaded: {x_data.shape[0]}")
    print(f"Data shape: {x_data.shape}")
    print(f"Memory usage: ~{x_data.nbytes / 1e9:.2f} GB")
    
    return x_data, y_data


def compute_normalization_params(path, users, folders, frame_size=30, max_files=20):
    """Compute normalization parameters from a subset of data"""
    print("Computing normalization parameters...")

    train_runs = list(range(1, 11))
    if 'TrainingSet' in folders:
        use_runs = train_runs
    else:
        use_runs = train_runs

    all_data = []
    file_count = 0

    for user in users[:min(20, len(users))]:
        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)

        if not os.path.exists(user_path):
            continue

        for run in use_runs[:2]:
            filename = f"S{user:03d}R{run:02d}.edf"
            filepath = os.path.join(user_path, filename)

            if not os.path.exists(filepath):
                continue

            data = load_edf_file(filepath)
            if data is None:
                continue

            windowed = create_windows(data, frame_size=frame_size, overlap=0.5)
            del data
            gc.collect()

            if windowed is not None and windowed.shape[0] > 0:
                sample_size = min(50, windowed.shape[0])
                sample_indices = np.random.choice(windowed.shape[0], sample_size, replace=False)
                all_data.append(windowed[sample_indices])
                file_count += 1

            del windowed
            gc.collect()

            if file_count >= max_files:
                break

        if file_count >= max_files:
            break

    if len(all_data) == 0:
        raise ValueError("No data loaded for normalization")

    all_data = np.concatenate(all_data, axis=0)
    all_data_flat = all_data.reshape(-1, all_data.shape[-1])
    _, mean, std = standardize(all_data_flat)

    del all_data, all_data_flat
    gc.collect()

    print(f"Normalization params computed from {file_count} files")
    return {'mean': mean, 'std': std}


def normalize_data(x_data, mean, std):
    """Normalize data in-place to save memory"""
    print("Normalizing data...")
    original_shape = x_data.shape
    x_flat = x_data.reshape(-1, x_data.shape[-1])
    x_flat[:] = apply_standardization(x_flat, mean, std)
    print("Normalization complete")
    return x_data


def augment_data_in_memory(x_data, y_data):
    """
    Apply ALL augmentations to data in memory during loading (CPU).
    This is done ONCE, so GPU can train fast without augmentation overhead.
    Creates 4x more data with different augmentations.
    
    ALWAYS applies full augmentation - no adaptive logic.
    """
    num_samples = x_data.shape[0]
    
    print("\nApplying augmentations to data (this will take a few minutes)...")
    print(f"Starting with {num_samples} samples...")
    from transformations import DA_Scaling, DA_MagWarp, DA_Negation
    
    augmented_x = []
    augmented_y = []
    
    # Keep original data
    print("  Keeping original data...")
    augmented_x.append(x_data)
    augmented_y.append(y_data)
    
    # Create 3 augmented copies
    print(f"  Applying Scaling augmentation to {num_samples} samples...")
    x_scaled = np.zeros_like(x_data)
    for i in range(num_samples):
        x_scaled[i] = DA_Scaling(x_data[i], sigma=0.1)
        if i % 5000 == 0 and i > 0:
            print(f"    Progress: {i}/{num_samples}")
    augmented_x.append(x_scaled)
    augmented_y.append(y_data.copy())
    print(f"    ✓ Scaling complete: {x_scaled.shape[0]} samples")
    
    print(f"  Applying Magnitude Warping augmentation to {num_samples} samples...")
    x_magwarp = np.zeros_like(x_data)
    for i in range(num_samples):
        x_magwarp[i] = DA_MagWarp(x_data[i], sigma=0.3)
        if i % 5000 == 0 and i > 0:
            print(f"    Progress: {i}/{num_samples}")
    augmented_x.append(x_magwarp)
    augmented_y.append(y_data.copy())
    print(f"    ✓ MagWarp complete: {x_magwarp.shape[0]} samples")
    
    print(f"  Applying Negation augmentation to {num_samples} samples...")
    x_negated = -x_data  # Fast!
    augmented_x.append(x_negated)
    augmented_y.append(y_data.copy())
    print(f"    ✓ Negation complete: {x_negated.shape[0]} samples")
    
    # Concatenate all
    print("  Concatenating all augmented data...")
    x_augmented = np.concatenate(augmented_x, axis=0)
    y_augmented = np.concatenate(augmented_y, axis=0)
    
    print(f"\n  ✓ AUGMENTATION SUMMARY:")
    print(f"    Original samples:   {num_samples}")
    print(f"    Augmented samples:  {x_augmented.shape[0]} (4x)")
    print(f"    Memory usage:       ~{x_augmented.nbytes / 1e9:.2f} GB")
    print(f"  ✓ Augmentation complete!\n")
    
    del augmented_x, augmented_y, x_scaled, x_magwarp, x_negated
    gc.collect()
    
    return x_augmented, y_augmented


def create_tf_dataset(x_data, y_data, batch_size, shuffle=True):
    """
    Create fast TensorFlow dataset from in-memory data
    No runtime augmentation - data is already augmented
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(10000, len(x_data)))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def data_load_with_tf_datasets(path, users, frame_size=30, batch_size=32):
    """
    Load all data into memory once, then create fast TF datasets
    This is MUCH faster than on-the-fly file loading
    
    For large datasets (>80 users), loads and augments in chunks to avoid OOM
    
    Returns:
        train_ds, val_ds, test_ds: TensorFlow datasets ready for training
        steps: Dictionary with step counts
    """
    print("\n" + "="*60)
    print("LOADING DATA INTO MEMORY")
    print("="*60)
    
    # Compute normalization from training data
    norm_params = compute_normalization_params(
        path, users, ['TrainingSet'], frame_size=frame_size
    )
    
    # Determine if we need chunked loading (for large datasets)
    num_users = len(users)
    use_chunked_loading = num_users > 70  # Threshold for chunked loading
    
    if use_chunked_loading:
        print(f"\n⚠ Large dataset ({num_users} users) - using chunked loading to avoid OOM")
        chunk_size = 30  # Load 30 users at a time
        
        # Load training data in chunks
        print("\n1. Loading and augmenting TRAINING data in chunks...")
        train_chunks_x = []
        train_chunks_y = []
        
        for chunk_start in range(0, num_users, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_users)
            chunk_users = users[chunk_start:chunk_end]
            
            print(f"\n  Chunk {chunk_start//chunk_size + 1}: Users {chunk_start+1}-{chunk_end}")
            
            # Load chunk
            x_chunk, y_chunk = load_all_data_to_memory(
                path, chunk_users, ['TrainingSet'], frame_size=frame_size
            )
            x_chunk = normalize_data(x_chunk, norm_params['mean'], norm_params['std'])
            
            # Augment chunk
            x_chunk, y_chunk = augment_data_in_memory(x_chunk, y_chunk)
            
            # Adjust labels for proper user IDs
            y_chunk = y_chunk + chunk_start
            
            # Store chunk
            train_chunks_x.append(x_chunk)
            train_chunks_y.append(y_chunk)
            
            del x_chunk, y_chunk
            gc.collect()
        
        # Concatenate all chunks
        print("\n  Concatenating all training chunks...")
        x_train = np.concatenate(train_chunks_x, axis=0)
        y_train = np.concatenate(train_chunks_y, axis=0)
        
        del train_chunks_x, train_chunks_y
        gc.collect()
        
    else:
        # Normal loading for small/medium datasets
        print("\n1. Loading TRAINING data...")
        x_train, y_train = load_all_data_to_memory(
            path, users, ['TrainingSet'], frame_size=frame_size
        )
        x_train = normalize_data(x_train, norm_params['mean'], norm_params['std'])
        
        # Apply augmentation to training data (CPU, one-time)
        x_train, y_train = augment_data_in_memory(x_train, y_train)
    
    # Aggressive garbage collection
    gc.collect()
    
    # Load validation data (no augmentation)
    print("\n2. Loading VALIDATION data...")
    x_val, y_val = load_all_data_to_memory(
        path, users, ['TestingSet'], frame_size=frame_size
    )
    x_val = normalize_data(x_val, norm_params['mean'], norm_params['std'])
    
    # Aggressive garbage collection
    gc.collect()
    
    # Load test data (no augmentation)
    print("\n3. Loading TEST data...")
    x_test, y_test = load_all_data_to_memory(
        path, users, ['TestingSet_secret'], frame_size=frame_size
    )
    x_test = normalize_data(x_test, norm_params['mean'], norm_params['std'])
    
    # Aggressive garbage collection
    gc.collect()
    
    print("\n" + "="*60)
    print("DATA LOADING COMPLETE")
    print("="*60)
    print(f"Train: {x_train.shape[0]} samples (includes 4x augmentations)")
    print(f"Val:   {x_val.shape[0]} samples")
    print(f"Test:  {x_test.shape[0]} samples")
    print(f"Total memory: ~{(x_train.nbytes + x_val.nbytes + x_test.nbytes) / 1e9:.2f} GB")
    print("="*60 + "\n")
    
    # Create TensorFlow datasets WITHOUT runtime augmentation
    print("Creating TensorFlow datasets...")
    train_ds = create_tf_dataset(x_train, y_train, batch_size, shuffle=True)
    val_ds = create_tf_dataset(x_val, y_val, batch_size, shuffle=False)
    test_ds = create_tf_dataset(x_test, y_test, batch_size, shuffle=False)
    
    # Calculate steps
    steps = {
        'train': int(np.ceil(len(x_train) / batch_size)),
        'val': int(np.ceil(len(x_val) / batch_size)),
        'test': int(np.ceil(len(x_test) / batch_size))
    }
    
    print(f"Steps per epoch - Train: {steps['train']}, Val: {steps['val']}, Test: {steps['test']}")
    
    return train_ds, val_ds, test_ds, steps


# Keep legacy functions for backward compatibility with pre_trainer
def data_load_origin(path, users, folders, frame_size=30, max_samples_per_user=None):
    """Legacy function for pre_trainer"""
    x_data, y_data = load_all_data_to_memory(path, users, folders, frame_size)
    
    if max_samples_per_user is not None:
        # Sample per user
        unique_users = np.unique(y_data)
        sampled_x = []
        sampled_y = []
        
        for user_id in unique_users:
            user_mask = y_data == user_id
            user_indices = np.where(user_mask)[0]
            
            if len(user_indices) > max_samples_per_user:
                selected = np.random.choice(user_indices, max_samples_per_user, replace=False)
            else:
                selected = user_indices
            
            sampled_x.append(x_data[selected])
            sampled_y.append(y_data[selected])
        
        x_data = np.concatenate(sampled_x, axis=0)
        y_data = np.concatenate(sampled_y, axis=0)
    
    sessions = [1] * len(np.unique(y_data))  # Dummy session count
    return x_data, y_data, sessions


def norma_pre(x_all):
    """Normalize data using manual standardization"""
    if x_all.shape[0] == 0:
        return x_all
    original_shape = x_all.shape
    x = x_all.reshape(-1, x_all.shape[-1])
    x, _, _ = standardize(x)
    x_all = x.reshape(original_shape)
    del x
    gc.collect()
    return x_all
