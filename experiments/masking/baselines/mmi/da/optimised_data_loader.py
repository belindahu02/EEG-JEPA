import numpy as np
import tensorflow as tf
import gc


@tf.function
def tf_augment_fast(x):
    """
    Fast TensorFlow-native augmentation (Scaling + Jitter + Drop).
    All operations run on GPU - no Python calls!
    """
    # 1. Scaling: multiply by random factor
    scale_factor = tf.random.normal([1, tf.shape(x)[1]], mean=1.0, stddev=0.55, dtype=tf.float32)
    x = x * scale_factor
    
    # 2. Jitter: add Gaussian noise
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.8, dtype=tf.float32)
    x = x + noise
    
    # 3. Drop: randomly zero out a segment
    drop_start = tf.random.uniform([], minval=0, maxval=tf.shape(x)[0] - 10, dtype=tf.int32)
    indices = tf.range(tf.shape(x)[0])
    drop_mask = tf.cast((indices < drop_start) | (indices >= drop_start + 10), tf.float32)
    drop_mask = tf.expand_dims(drop_mask, axis=1)
    x = x * drop_mask
    
    return x


def create_tf_dataset(file_list, frame_size, mean, std, batch_size=8, 
                      shuffle=True, augment=False, augmentation_fn=None):
    """
    Hybrid approach: Pre-load original data, apply fast TF augmentation on-the-fly.
    
    Memory usage:
    - Pre-loads ALL original data (normalized)
    - Does NOT pre-compute augmented versions
    - Applies augmentation during training (GPU-accelerated)
    
    This uses ~50% less memory than pre-loading augmented versions.
    """
    from data_loader import load_edf_session
    
    # Create sample index mapping
    sample_indices = []
    for file_idx, (filepath, user_id, n_windows) in enumerate(file_list):
        for window_idx in range(n_windows):
            sample_indices.append((file_idx, window_idx, user_id))
    
    n_samples = len(sample_indices)
    
    # Convert to numpy arrays for efficiency
    file_indices = np.array([x[0] for x in sample_indices], dtype=np.int32)
    window_indices = np.array([x[1] for x in sample_indices], dtype=np.int32)
    user_ids = np.array([x[2] for x in sample_indices], dtype=np.int32)
    
    # Pre-load ALL files into memory (ORIGINAL data only, no augmented copies)
    print(f"Pre-loading {len(file_list)} files into memory (original data only)...")
    file_data_cache = {}
    
    for file_idx, (filepath, user_id, n_windows) in enumerate(file_list):
        data = load_edf_session(filepath, frame_size)
        if data is not None:
            # Normalize immediately
            data_flat = data.reshape(-1, data.shape[2])
            data_flat = (data_flat - mean) / std
            data = data_flat.reshape(data.shape).astype(np.float32)
            file_data_cache[file_idx] = data
            
        if (file_idx + 1) % 50 == 0:
            print(f"  Loaded {file_idx + 1}/{len(file_list)} files")
            gc.collect()  # Clean up during loading
    
    print(f"All files loaded! (Augmentation will be applied on-the-fly during training)")
    
    def generator():
        """Generator function that yields samples"""
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        for idx in indices:
            file_idx = file_indices[idx]
            window_idx = window_indices[idx]
            user_id = user_ids[idx]
            
            # Get from pre-loaded cache (fast!)
            X = file_data_cache[file_idx][window_idx]
            y = user_id
            
            yield X, y
    
    # Create dataset
    n_channels = len(mean)
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(frame_size, n_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    # Apply fast TensorFlow augmentation if requested (50% of samples)
    if augment:
        dataset = dataset.map(
            lambda x, y: (
                tf.cond(
                    tf.random.uniform([]) > 0.5,
                    lambda: tf_augment_fast(x),
                    lambda: x
                ),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, n_samples
