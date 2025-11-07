"""
EEG Masking Experiment for Data Augmentation Method - OOM SAFE VERSION
=======================================================================
This version includes multiple OOM prevention strategies:
1. Incremental Cohen's Kappa calculation (no memory accumulation)
2. TensorFlow GPU memory growth enabled
3. Aggressive garbage collection
4. Smaller default batch size
5. Manual cleanup after each configuration
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from backbones import resnetblock_final
from data_loader import load_edf_session


# CRITICAL: Configure TensorFlow for memory safety
def configure_tensorflow_memory():
    """Configure TensorFlow to prevent OOM"""
    # Enable GPU memory growth (allocate as needed, not all at once)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"✓ GPU memory growth enabled for {len(physical_devices)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠ Could not set memory growth: {e}")
    
    # Set per-process GPU memory limit (optional, adjust as needed)
    # Uncomment if you want to hard-limit GPU memory
    # if physical_devices:
    #     tf.config.set_logical_device_configuration(
    #         physical_devices[0],
    #         [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB limit
    #     )


# Call this BEFORE any TensorFlow operations
configure_tensorflow_memory()


def cohen_kappa_score(y_true, y_pred, num_classes):
    """Calculate Cohen's Kappa score"""
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[int(y_true[i]), int(y_pred[i])] += 1
    
    n = len(y_true)
    observed_accuracy = np.trace(confusion_matrix) / n
    
    expected_accuracy = 0
    for i in range(num_classes):
        expected_accuracy += (np.sum(confusion_matrix[i, :]) * np.sum(confusion_matrix[:, i])) / (n * n)
    
    if expected_accuracy == 1.0:
        kappa = 1.0
    else:
        kappa = (observed_accuracy - expected_accuracy) / (1.0 - expected_accuracy)
    
    return kappa


def cohen_kappa_score_incremental(model, dataset, num_classes, batch_size):
    """
    Calculate Cohen's Kappa INCREMENTALLY without loading all predictions.
    This prevents OOM by processing batches one at a time.
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_samples = 0
    
    # Process one batch at a time
    for X_batch, y_batch in dataset:
        # Get predictions for this batch
        preds = model.predict(X_batch, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = y_batch.numpy().flatten()
        
        # Update confusion matrix incrementally
        for true_label, pred_label in zip(true_classes, pred_classes):
            confusion_matrix[int(true_label), int(pred_label)] += 1
            total_samples += 1
        
        # Clean up batch data immediately
        del X_batch, y_batch, preds, pred_classes, true_classes
        gc.collect()
    
    # Calculate Kappa from confusion matrix
    n = total_samples
    observed_accuracy = np.trace(confusion_matrix) / n
    
    expected_accuracy = 0
    for i in range(num_classes):
        expected_accuracy += (np.sum(confusion_matrix[i, :]) * np.sum(confusion_matrix[:, i])) / (n * n)
    
    if expected_accuracy == 1.0:
        kappa = 1.0
    else:
        kappa = (observed_accuracy - expected_accuracy) / (1.0 - expected_accuracy)
    
    return kappa


def mask_eeg_data(data, masking_percentage, num_blocks):
    """
    Mask EEG data by zeroing out specified percentage in blocks.
    data: shape (n_windows, frame_size, n_channels)
    """
    if masking_percentage == 0:
        return data
    
    masked_data = data.copy()
    n_windows, frame_size, n_channels = data.shape
    
    frames_to_mask = int(frame_size * masking_percentage / 100)
    if frames_to_mask == 0:
        return masked_data
    
    block_size = max(1, frames_to_mask // num_blocks)
    
    for window_idx in range(n_windows):
        masked_frames = 0
        attempts = 0
        max_attempts = num_blocks * 3
        
        while masked_frames < frames_to_mask and attempts < max_attempts:
            start_idx = np.random.randint(0, max(1, frame_size - block_size + 1))
            end_idx = min(start_idx + block_size, frame_size)
            masked_data[window_idx, start_idx:end_idx, :] = 0
            masked_frames += (end_idx - start_idx)
            attempts += 1
    
    return masked_data


def create_lightweight_dataset(file_list, frame_size, mean, std, batch_size, 
                                masking_percentage=0, num_blocks=1, shuffle=True):
    """
    Memory-efficient dataset creation using Python generator (no pre-loading).
    REDUCED cache size for OOM safety.
    """
    # Create sample mapping
    sample_indices = []
    for file_idx, (filepath, user_id, n_windows) in enumerate(file_list):
        for window_idx in range(n_windows):
            sample_indices.append((file_idx, window_idx, user_id))
    
    n_samples = len(sample_indices)
    indices = np.arange(n_samples)
    
    def generator():
        """Generator that loads files on-demand"""
        file_cache = {}
        max_cache_size = 3  # REDUCED from 5 to 3 for OOM safety
        
        if shuffle:
            np.random.shuffle(indices)
        
        for idx in indices:
            file_idx, window_idx, user_id = sample_indices[idx]
            filepath, _, _ = file_list[file_idx]
            
            # Load file if not cached
            if file_idx not in file_cache:
                # Clear cache if too large
                if len(file_cache) >= max_cache_size:
                    oldest = next(iter(file_cache))
                    del file_cache[oldest]
                    gc.collect()  # Force cleanup
                
                # Load and process file
                data = load_edf_session(filepath, frame_size)
                if data is not None:
                    # Apply masking
                    if masking_percentage > 0:
                        data = mask_eeg_data(data, masking_percentage, num_blocks)
                    
                    # Normalize
                    data_flat = data.reshape(-1, data.shape[2])
                    data_flat = (data_flat - mean) / std
                    data = data_flat.reshape(data.shape).astype(np.float32)
                    file_cache[file_idx] = data
                else:
                    file_cache[file_idx] = np.zeros((1, frame_size, len(mean)), dtype=np.float32)
            
            X = file_cache[file_idx][window_idx]
            y = user_id
            
            yield X, y
    
    n_channels = len(mean)
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(frame_size, n_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(1)  # Minimal prefetch
    
    return dataset, n_samples


def get_file_lists_and_stats(path, users, frame_size):
    """Get file lists and compute normalization statistics."""
    from data_loader import data_load_eeg_mmi_streaming
    
    print("Loading file lists and computing statistics...")
    train_files, val_files, test_files, mean, std, n_channels, train_samples, val_samples, test_samples = \
        data_load_eeg_mmi_streaming(path, users, frame_size)
    
    return train_files, val_files, test_files, mean, std, n_channels


class MaskingExperimentCheckpoint:
    """Checkpoint manager for saving/loading experiment state"""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        self.results_file = self.checkpoint_dir / "results.npz"
    
    def has_checkpoint(self):
        return self.checkpoint_file.exists()
    
    def save_checkpoint(self, stage, stage_data, accuracy_matrix=None, 
                       kappa_matrix=None, detailed_results=None, metadata=None):
        """Save checkpoint state"""
        checkpoint_data = {
            'stage': stage,
            'stage_data': stage_data,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Save JSON checkpoint
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Save numpy arrays if provided
        if accuracy_matrix is not None and kappa_matrix is not None:
            np.savez(
                self.results_file,
                accuracy_matrix=accuracy_matrix,
                kappa_matrix=kappa_matrix,
                detailed_results=detailed_results if detailed_results else []
            )
        
        print(f"✓ Checkpoint saved: {stage}")
    
    def load_checkpoint(self):
        """Load checkpoint state"""
        if not self.has_checkpoint():
            return None
        
        with open(self.checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Load numpy arrays if they exist
        if self.results_file.exists():
            results_data = np.load(self.results_file, allow_pickle=True)
            checkpoint_data['accuracy_matrix'] = results_data['accuracy_matrix']
            checkpoint_data['kappa_matrix'] = results_data['kappa_matrix']
            checkpoint_data['detailed_results'] = results_data['detailed_results'].tolist()
        
        return checkpoint_data
    
    def clear_checkpoint(self):
        """Remove checkpoint files"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.results_file.exists():
            self.results_file.unlink()


class EEGMaskingExperiment:
    """Complete EEG masking experiment pipeline - OOM SAFE VERSION"""
    
    def __init__(self, config):
        self.config = config
        self.experiment_dir = Path(config['experiment_dir'])
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.experiment_dir / "models"
        self.results_dir = self.experiment_dir / "results"
        self.plots_dir = self.experiment_dir / "plots"
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        
        for d in [self.models_dir, self.results_dir, self.plots_dir, self.checkpoint_dir]:
            d.mkdir(exist_ok=True)
        
        # Initialize checkpoint manager
        self.checkpoint_mgr = MaskingExperimentCheckpoint(self.checkpoint_dir)
        
        # Masking parameter grid
        self.masking_percentages = np.arange(0, 55, 5)
        self.num_blocks_range = [1, 2, 3, 4, 5]
        
        # Store trained model
        self.base_model = None
        self.base_model_path = None
        
        print(f"EEG Masking Experiment (OOM-Safe) initialized")
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Grid size: {len(self.masking_percentages)} × {len(self.num_blocks_range)} = {len(self.masking_percentages) * len(self.num_blocks_range)}")
        print(f"OOM Safety Features:")
        print(f"  - Incremental Kappa calculation (no memory accumulation)")
        print(f"  - TensorFlow GPU memory growth enabled")
        print(f"  - Aggressive garbage collection")
        print(f"  - Reduced cache size (3 files)")
        
        # Check for existing checkpoint
        if self.checkpoint_mgr.has_checkpoint():
            print("⚠️  EXISTING CHECKPOINT FOUND")
            print("Set resume=True in run_complete_experiment() to continue")
    
    def step1_train_base_model(self):
        """Train the base model on unmasked training data"""
        print("\n" + "="*70)
        print("STEP 1: Training Base Model (Unmasked Data)")
        print("="*70)
        
        frame_size = self.config['frame_size']
        path = self.config['raw_eeg_dir']
        users = self.config['user_ids']
        num_classes = len(users)
        batch_size = self.config['batch_size']
        
        # Get file lists and stats
        train_files, val_files, test_files, mean, std, n_channels = get_file_lists_and_stats(
            path, users, frame_size
        )
        
        print(f"Training files: {len(train_files)}")
        print(f"Validation files: {len(val_files)}")
        print(f"Test files: {len(test_files)}")
        
        # Store for later use
        self.test_files = test_files
        self.mean = mean
        self.std = std
        self.n_channels = n_channels
        
        # Create datasets (NO masking for training)
        print("\nCreating training datasets (unmasked)...")
        train_dataset, _ = create_lightweight_dataset(
            train_files, frame_size, mean, std, batch_size,
            masking_percentage=0, num_blocks=1, shuffle=True
        )
        
        val_dataset, _ = create_lightweight_dataset(
            val_files, frame_size, mean, std, batch_size,
            masking_percentage=0, num_blocks=1, shuffle=False
        )
        
        # Build model
        print("\nBuilding model...")
        ks = 3
        con = 3
        inputs = Input(shape=(frame_size, n_channels))
        x = Conv1D(filters=16 * con, kernel_size=ks, strides=1, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling1D(pool_size=4, strides=4)(x)
        x = Dropout(rate=0.1)(x)
        x = resnetblock_final(x, CR=32 * con, KS=ks)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs, outputs)
        
        # Compile
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            restore_best_weights=True,
            patience=5
        )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config['learning_rate'],
            decay_rate=0.95,
            decay_steps=1000
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        print(f"\nTraining model for {self.config['train_epochs']} epochs...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config['train_epochs'],
            callbacks=[callback],
            verbose=1
        )
        
        # Get best validation accuracy
        best_val_acc = max(history.history['val_accuracy'])
        print(f"\nBest validation accuracy: {best_val_acc:.4f}")
        
        # Save model
        self.base_model_path = self.models_dir / "base_model.h5"
        model.save(str(self.base_model_path))
        self.base_model = model
        
        print(f"Model saved to: {self.base_model_path}")
        
        # CLEANUP
        del train_dataset, val_dataset
        gc.collect()
        tf.keras.backend.clear_session()
        
        # Save checkpoint
        self.checkpoint_mgr.save_checkpoint(
            stage='base_model_trained',
            stage_data={
                'model_path': str(self.base_model_path),
                'best_val_acc': float(best_val_acc),
                'num_classes': num_classes,
                'n_channels': int(n_channels)
            },
            metadata={'step': 1, 'description': 'Base model training completed'}
        )
        
        return best_val_acc
    
    def step2_evaluate_masking_grid(self):
        """Evaluate model on all masking configurations - OOM SAFE"""
        print("\n" + "="*70)
        print("STEP 2: Evaluating on Masked Test Data (OOM-Safe)")
        print("="*70)
        
        # Initialize result matrices
        n_mask = len(self.masking_percentages)
        n_blocks = len(self.num_blocks_range)
        accuracy_matrix = np.full((n_mask, n_blocks), np.nan)
        kappa_matrix = np.full((n_mask, n_blocks), np.nan)
        detailed_results = []
        
        # Load model if not already loaded
        if self.base_model is None:
            print("Loading base model...")
            self.base_model = tf.keras.models.load_model(str(self.base_model_path))
        
        frame_size = self.config['frame_size']
        batch_size = self.config['batch_size']
        num_classes = len(self.config['user_ids'])
        
        # Evaluate each configuration
        total_configs = n_mask * n_blocks
        completed = 0
        start_time = time.time()
        
        for mask_idx, mask_pct in enumerate(self.masking_percentages):
            for block_idx, n_blocks_val in enumerate(self.num_blocks_range):
                completed += 1
                
                print(f"\nConfiguration {completed}/{total_configs}: "
                      f"{mask_pct}% masking, {n_blocks_val} blocks")
                
                try:
                    # Create masked test dataset
                    test_dataset, _ = create_lightweight_dataset(
                        self.test_files, frame_size, self.mean, self.std, 
                        batch_size, mask_pct, n_blocks_val, shuffle=False
                    )
                    
                    # Evaluate
                    print("  Evaluating...")
                    results = self.base_model.evaluate(test_dataset, verbose=0)
                    test_acc = results[1]
                    
                    # CRITICAL: Calculate Kappa INCREMENTALLY (OOM-safe)
                    print("  Computing Cohen's Kappa (incremental)...")
                    kappa_score = cohen_kappa_score_incremental(
                        self.base_model, test_dataset, num_classes, batch_size
                    )
                    
                    # Store results
                    accuracy_matrix[mask_idx, block_idx] = test_acc
                    kappa_matrix[mask_idx, block_idx] = kappa_score
                    
                    detailed_results.append({
                        'masking_percentage': int(mask_pct),
                        'num_blocks': int(n_blocks_val),
                        'test_accuracy': float(test_acc),
                        'cohen_kappa': float(kappa_score)
                    })
                    
                    print(f"  ✓ Accuracy: {test_acc:.4f}, Kappa: {kappa_score:.4f}")
                    
                    # CRITICAL: Aggressive cleanup after EACH configuration
                    del test_dataset
                    gc.collect()
                    tf.keras.backend.clear_session()
                    
                    # Reload model (TensorFlow quirk - helps with memory)
                    if completed % 10 == 0:  # Every 10 configs
                        print("  Reloading model for memory cleanup...")
                        del self.base_model
                        gc.collect()
                        self.base_model = tf.keras.models.load_model(str(self.base_model_path))
                    
                    # Save checkpoint after each configuration
                    self.checkpoint_mgr.save_checkpoint(
                        stage='evaluating_grid',
                        stage_data={
                            'completed': completed,
                            'total': total_configs,
                            'last_mask_pct': int(mask_pct),
                            'last_n_blocks': int(n_blocks_val)
                        },
                        accuracy_matrix=accuracy_matrix,
                        kappa_matrix=kappa_matrix,
                        detailed_results=detailed_results,
                        metadata={'step': 2, 'description': f'Evaluated {completed}/{total_configs}'}
                    )
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Try to recover
                    gc.collect()
                    tf.keras.backend.clear_session()
        
        elapsed = time.time() - start_time
        print(f"\nGrid evaluation completed in {elapsed/3600:.2f} hours")
        
        return accuracy_matrix, kappa_matrix, detailed_results
    
    def step3_generate_plots(self, accuracy_matrix, kappa_matrix):
        """Generate all visualization plots"""
        print("\n" + "="*70)
        print("STEP 3: Generating Plots")
        print("="*70)
        
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10
        
        # 1. Accuracy heatmap
        print("Creating accuracy heatmap...")
        plt.figure(figsize=(12, 8))
        im = plt.imshow(accuracy_matrix, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(im, label='Test Accuracy')
        plt.xlabel('Number of Blocks', fontsize=12)
        plt.ylabel('Masking Percentage (%)', fontsize=12)
        plt.title('Test Accuracy: Masking Percentage vs Number of Blocks', 
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(self.num_blocks_range)), self.num_blocks_range)
        plt.yticks(range(len(self.masking_percentages)), self.masking_percentages)
        
        for i in range(len(self.masking_percentages)):
            for j in range(len(self.num_blocks_range)):
                if not np.isnan(accuracy_matrix[i, j]):
                    plt.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                           ha="center", va="center", color="white", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Kappa heatmap
        print("Creating kappa heatmap...")
        plt.figure(figsize=(12, 8))
        im = plt.imshow(kappa_matrix, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(im, label="Cohen's Kappa")
        plt.xlabel('Number of Blocks', fontsize=12)
        plt.ylabel('Masking Percentage (%)', fontsize=12)
        plt.title("Cohen's Kappa: Masking Percentage vs Number of Blocks", 
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(self.num_blocks_range)), self.num_blocks_range)
        plt.yticks(range(len(self.masking_percentages)), self.masking_percentages)
        
        for i in range(len(self.masking_percentages)):
            for j in range(len(self.num_blocks_range)):
                if not np.isnan(kappa_matrix[i, j]):
                    plt.text(j, i, f'{kappa_matrix[i, j]:.3f}',
                           ha="center", va="center", color="white", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'kappa_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3-6. Other plots (same as before)
        # Accuracy vs masking percentage
        plt.figure(figsize=(12, 8))
        for block_idx, n_blocks in enumerate(self.num_blocks_range):
            plt.plot(self.masking_percentages, accuracy_matrix[:, block_idx],
                    marker='o', label=f'{n_blocks} blocks', linewidth=2, markersize=8)
        plt.xlabel('Masking Percentage (%)', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.title('Test Accuracy vs Masking Percentage', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_vs_masking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Kappa vs masking percentage
        plt.figure(figsize=(12, 8))
        for block_idx, n_blocks in enumerate(self.num_blocks_range):
            plt.plot(self.masking_percentages, kappa_matrix[:, block_idx],
                    marker='o', label=f'{n_blocks} blocks', linewidth=2, markersize=8)
        plt.xlabel('Masking Percentage (%)', fontsize=12)
        plt.ylabel("Cohen's Kappa", fontsize=12)
        plt.title("Cohen's Kappa vs Masking Percentage", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'kappa_vs_masking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Accuracy vs blocks
        plt.figure(figsize=(12, 8))
        for mask_idx, mask_pct in enumerate(self.masking_percentages[::2]):
            actual_idx = mask_idx * 2
            plt.plot(self.num_blocks_range, accuracy_matrix[actual_idx, :],
                    marker='o', label=f'{mask_pct}% masking', linewidth=2, markersize=8)
        plt.xlabel('Number of Blocks', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.title('Test Accuracy vs Number of Blocks', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_vs_blocks.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Kappa vs blocks
        plt.figure(figsize=(12, 8))
        for mask_idx, mask_pct in enumerate(self.masking_percentages[::2]):
            actual_idx = mask_idx * 2
            plt.plot(self.num_blocks_range, kappa_matrix[actual_idx, :],
                    marker='o', label=f'{mask_pct}% masking', linewidth=2, markersize=8)
        plt.xlabel('Number of Blocks', fontsize=12)
        plt.ylabel("Cohen's Kappa", fontsize=12)
        plt.title("Cohen's Kappa vs Number of Blocks", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'kappa_vs_blocks.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ All plots saved to {self.plots_dir}")
    
    def step4_save_results(self, accuracy_matrix, kappa_matrix, detailed_results, base_val_acc):
        """Save all results"""
        print("\n" + "="*70)
        print("STEP 4: Saving Results")
        print("="*70)
        
        # Save numpy arrays
        results_file = self.results_dir / "masking_results.npz"
        np.savez(
            results_file,
            accuracy_matrix=accuracy_matrix,
            kappa_matrix=kappa_matrix,
            masking_percentages=self.masking_percentages,
            num_blocks_range=self.num_blocks_range,
            base_val_acc=base_val_acc
        )
        print(f"✓ Saved numpy results to {results_file}")
        
        # Save detailed JSON results
        results_json = {
            'experiment_info': {
                'experiment_dir': str(self.experiment_dir),
                'base_model_path': str(self.base_model_path),
                'base_val_acc': float(base_val_acc) if base_val_acc else None,
                'num_users': len(self.config['user_ids']),
                'frame_size': self.config['frame_size'],
                'batch_size': self.config['batch_size'],
                'train_epochs': self.config['train_epochs'],
                'learning_rate': self.config['learning_rate']
            },
            'masking_grid': {
                'masking_percentages': self.masking_percentages.tolist(),
                'num_blocks_range': self.num_blocks_range
            },
            'results': {
                'accuracy_matrix': accuracy_matrix.tolist(),
                'kappa_matrix': kappa_matrix.tolist(),
                'detailed_results': detailed_results
            },
            'statistics': {
                'baseline_accuracy': float(accuracy_matrix[0, 0]) if not np.isnan(accuracy_matrix[0, 0]) else None,
                'mean_accuracy': float(np.nanmean(accuracy_matrix)),
                'std_accuracy': float(np.nanstd(accuracy_matrix)),
                'mean_kappa': float(np.nanmean(kappa_matrix)),
                'std_kappa': float(np.nanstd(kappa_matrix))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        results_json_file = self.results_dir / "experiment_results.json"
        with open(results_json_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"✓ Saved JSON results to {results_json_file}")
        
        # Save summary text file
        summary_file = self.results_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EEG MASKING EXPERIMENT SUMMARY (OOM-Safe Data Augmentation)\n")
            f.write("="*70 + "\n\n")
            f.write(f"Experiment directory: {self.experiment_dir}\n")
            f.write(f"Base model: {self.base_model_path}\n")
            f.write(f"Base validation accuracy: {base_val_acc:.4f}\n" if base_val_acc else "")
            f.write(f"\nGrid size: {len(self.masking_percentages)} × {len(self.num_blocks_range)}\n")
            f.write(f"Total configurations: {len(self.masking_percentages) * len(self.num_blocks_range)}\n\n")
            
            if not np.isnan(accuracy_matrix[0, 0]):
                f.write(f"Baseline accuracy (0% masking): {accuracy_matrix[0, 0]:.4f}\n")
            
            valid_mask = ~np.isnan(accuracy_matrix[1:])
            if valid_mask.any():
                masked_accuracies = accuracy_matrix[1:][valid_mask]
                f.write(f"Best masked performance: {np.max(masked_accuracies):.4f}\n")
                f.write(f"Worst masked performance: {np.min(masked_accuracies):.4f}\n")
                f.write(f"Mean masked performance: {np.mean(masked_accuracies):.4f}\n")
                f.write(f"Std masked performance: {np.std(masked_accuracies):.4f}\n")
        
        print(f"✓ Saved summary to {summary_file}")
        
        return results_json
    
    def run_complete_experiment(self, resume=False):
        """Run the complete experiment pipeline"""
        experiment_start_time = time.time()
        
        print("\n" + "="*70)
        print("STARTING EEG MASKING EXPERIMENT (OOM-Safe)")
        print("="*70)
        
        try:
            # Check for resume
            if resume and self.checkpoint_mgr.has_checkpoint():
                print("\nResuming from checkpoint...")
                checkpoint = self.checkpoint_mgr.load_checkpoint()
                stage = checkpoint['stage']
                
                if stage == 'base_model_trained':
                    print("Resuming from step 2 (grid evaluation)")
                    self.base_model_path = Path(checkpoint['stage_data']['model_path'])
                    self.base_model = tf.keras.models.load_model(str(self.base_model_path))
                    base_val_acc = checkpoint['stage_data']['best_val_acc']
                    
                    # Reload file info
                    frame_size = self.config['frame_size']
                    path = self.config['raw_eeg_dir']
                    users = self.config['user_ids']
                    _, _, test_files, mean, std, n_channels = get_file_lists_and_stats(path, users, frame_size)
                    self.test_files = test_files
                    self.mean = mean
                    self.std = std
                    self.n_channels = n_channels
                    
                elif stage == 'complete':
                    print("Experiment already complete!")
                    return checkpoint.get('results', None)
            else:
                # Step 1: Train base model
                base_val_acc = self.step1_train_base_model()
            
            # Step 2: Evaluate on masking grid
            accuracy_matrix, kappa_matrix, detailed_results = self.step2_evaluate_masking_grid()
            
            # Step 3: Generate plots
            self.step3_generate_plots(accuracy_matrix, kappa_matrix)
            
            # Step 4: Save results
            results_data = self.step4_save_results(accuracy_matrix, kappa_matrix, 
                                                   detailed_results, base_val_acc)
            
            # Mark complete
            self.checkpoint_mgr.save_checkpoint(
                stage='complete',
                stage_data={
                    'base_model_path': str(self.base_model_path),
                    'base_val_acc': float(base_val_acc) if base_val_acc else None,
                    'experiment_complete': True
                },
                accuracy_matrix=accuracy_matrix,
                kappa_matrix=kappa_matrix,
                detailed_results=detailed_results,
                metadata={'step': 4, 'description': 'Experiment completed'}
            )
            
            # Final summary
            experiment_time = time.time() - experiment_start_time
            
            print("\n" + "="*70)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"Total time: {experiment_time/3600:.2f} hours")
            print(f"Baseline accuracy: {accuracy_matrix[0,0]:.4f}")
            print(f"Results: {self.experiment_dir}")
            
            return results_data
        
        except KeyboardInterrupt:
            print("\n" + "="*70)
            print("EXPERIMENT INTERRUPTED")
            print("="*70)
            print("Progress saved. Resume with resume=True")
            return None
        
        except Exception as e:
            print(f"\nEXPERIMENT FAILED: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main execution function"""
    
    config = {
        'experiment_dir': '/app/data/experiments/da_masking_subset10',
        'raw_eeg_dir': '/app/data/1.0.0',
        'user_ids': list(range(1, 11)),
        'train_sessions': list(range(1, 11)),
        'val_sessions': [11, 12],
        'test_sessions': [13, 14],
        'train_epochs': 100,
        'batch_size': 8,  # REDUCED from 16 for OOM safety
        'learning_rate': 0.001,
        'frame_size': 40,
        'description': 'OOM-safe masking experiment with data augmentation'
    }
    
    print("EEG Masking Experiment (OOM-Safe)")
    print("=" * 70)
    print(f"Users: {len(config['user_ids'])} users")
    print(f"Batch size: {config['batch_size']} (reduced for OOM safety)")
    print(f"Experiment: {config['experiment_dir']}")
    print("=" * 70)
    
    experiment = EEGMaskingExperiment(config)
    results = experiment.run_complete_experiment(resume=False)
    
    if results is not None:
        print("\nExperiment completed successfully!")
        return True
    else:
        print("\nExperiment incomplete")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
