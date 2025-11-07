"""
EEG Masking Experiment - STREAMING VERSION (Memory Safe)
=========================================================
- Uses streaming datasets for training/validation (no OOM)
- Loads test data per-config (safe evaluation)
- Memory efficient while maintaining reasonable speed!
"""

import os
# Force CPU usage if GPU/cuDNN issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


def configure_gpu():
    """Configure TensorFlow to use GPU efficiently"""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            print(f"✓ Using GPU: {physical_devices[0]}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠ No GPU found, using CPU")

configure_gpu()


def cohen_kappa_score_fast(y_true, y_pred, num_classes):
    """Fast Cohen's Kappa calculation"""
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


def mask_eeg_data(data, masking_percentage, num_blocks):
    """Mask EEG data by zeroing out specified percentage in blocks"""
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


def create_streaming_dataset(file_list, frame_size, mean, std, batch_size, 
                             masking_percentage=0, num_blocks=1, shuffle=True,
                             shuffle_buffer=1000):
    """
    Create a streaming dataset that loads files on-demand
    Memory efficient - no pre-loading!
    """
    def data_generator():
        """Generator that yields samples on-the-fly"""
        for file_idx, (filepath, user_id, n_windows) in enumerate(file_list):
            try:
                data = load_edf_session(filepath, frame_size)
                
                if data is not None:
                    # Apply masking
                    if masking_percentage > 0:
                        data = mask_eeg_data(data, masking_percentage, num_blocks)
                    
                    # Normalize
                    data_flat = data.reshape(-1, data.shape[2])
                    data_flat = (data_flat - mean) / std
                    data = data_flat.reshape(data.shape).astype(np.float32)
                    
                    # Yield each window
                    for i in range(n_windows):
                        yield data[i], user_id
                    
                    # Free memory immediately
                    del data
                
            except Exception as e:
                print(f"  Warning: Error loading {filepath}: {e}")
                continue
    
    # Get output signature by loading first file
    sample_data = None
    for filepath, _, _ in file_list[:5]:  # Try first 5 files
        try:
            sample_data = load_edf_session(filepath, frame_size)
            if sample_data is not None:
                break
        except:
            continue
    
    if sample_data is None:
        raise ValueError("Could not load any sample data to determine shape")
    
    output_signature = (
        tf.TensorSpec(shape=(frame_size, sample_data.shape[2]), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def preload_data_fast(file_list, frame_size, mean, std, masking_percentage=0, num_blocks=1, desc="data"):
    """
    Pre-load data into memory (for test evaluation only)
    """
    print(f"  Pre-loading {len(file_list)} files ({desc})...")
    
    all_X = []
    all_y = []
    
    for file_idx, (filepath, user_id, n_windows) in enumerate(file_list):
        data = load_edf_session(filepath, frame_size)
        
        if data is not None:
            # Apply masking if needed
            if masking_percentage > 0:
                data = mask_eeg_data(data, masking_percentage, num_blocks)
            
            # Normalize
            data_flat = data.reshape(-1, data.shape[2])
            data_flat = (data_flat - mean) / std
            data = data_flat.reshape(data.shape).astype(np.float32)
            
            # Add to lists
            all_X.append(data)
            all_y.append(np.full(n_windows, user_id, dtype=np.int32))
        
        if (file_idx + 1) % 50 == 0:
            print(f"    Loaded {file_idx + 1}/{len(file_list)} files")
    
    # Concatenate everything
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"  ✓ Loaded {X.shape[0]} samples, shape: {X.shape}")
    print(f"  ✓ Memory: ~{X.nbytes / 1e9:.2f} GB")
    
    return X, y


def create_fast_dataset(X, y, batch_size, shuffle=True):
    """Create TensorFlow dataset from pre-loaded numpy arrays"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(10000, len(X)))
    
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def get_file_lists_and_stats(path, users, frame_size):
    """Get file lists and compute normalization statistics"""
    from data_loader import data_load_eeg_mmi_streaming
    
    print("Loading file lists and computing statistics...")
    train_files, val_files, test_files, mean, std, n_channels, train_samples, val_samples, test_samples = \
        data_load_eeg_mmi_streaming(path, users, frame_size)
    
    return train_files, val_files, test_files, mean, std, n_channels


class MaskingExperimentCheckpoint:
    """Checkpoint manager"""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        self.results_file = self.checkpoint_dir / "results.npz"
    
    def has_checkpoint(self):
        return self.checkpoint_file.exists()
    
    def save_checkpoint(self, stage, stage_data, accuracy_matrix=None, 
                       kappa_matrix=None, detailed_results=None, metadata=None):
        checkpoint_data = {
            'stage': stage,
            'stage_data': stage_data,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        if accuracy_matrix is not None and kappa_matrix is not None:
            np.savez(
                self.results_file,
                accuracy_matrix=accuracy_matrix,
                kappa_matrix=kappa_matrix,
                detailed_results=detailed_results if detailed_results else []
            )
        
        print(f"✓ Checkpoint saved: {stage}")
    
    def load_checkpoint(self):
        if not self.has_checkpoint():
            return None
        
        with open(self.checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        if self.results_file.exists():
            results_data = np.load(self.results_file, allow_pickle=True)
            checkpoint_data['accuracy_matrix'] = results_data['accuracy_matrix']
            checkpoint_data['kappa_matrix'] = results_data['kappa_matrix']
            checkpoint_data['detailed_results'] = results_data['detailed_results'].tolist()
        
        return checkpoint_data


class EEGMaskingExperiment:
    """Streaming experiment: Memory-safe training + evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.experiment_dir = Path(config['experiment_dir'])
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.experiment_dir / "models"
        self.results_dir = self.experiment_dir / "results"
        self.plots_dir = self.experiment_dir / "plots"
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        
        for d in [self.models_dir, self.results_dir, self.plots_dir, self.checkpoint_dir]:
            d.mkdir(exist_ok=True)
        
        self.checkpoint_mgr = MaskingExperimentCheckpoint(self.checkpoint_dir)
        
        self.masking_percentages = np.arange(0, 55, 5)
        self.num_blocks_range = [1, 2, 3, 4, 5]
        
        self.base_model = None
        self.base_model_path = None
        
        print(f"EEG Masking Experiment (STREAMING - Memory Safe) initialized")
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Grid size: {len(self.masking_percentages)} × {len(self.num_blocks_range)} = {len(self.masking_percentages) * len(self.num_blocks_range)}")
        print(f"Strategy:")
        print(f"  - Training: STREAMING datasets (memory safe)")
        print(f"  - Evaluation: Load test per-config (memory safe)")
    
    def step1_train_base_model(self):
        """Train base model - STREAMING (no pre-loading!)"""
        print("\n" + "="*70)
        print("STEP 1: Training Base Model (Streaming Mode - Memory Safe)")
        print("="*70)
        
        frame_size = self.config['frame_size']
        path = self.config['raw_eeg_dir']
        users = self.config['user_ids']
        num_classes = len(users)
        batch_size = self.config['batch_size']
        shuffle_buffer = self.config.get('shuffle_buffer', 1000)
        
        # Get file lists and stats
        train_files, val_files, test_files, mean, std, n_channels = get_file_lists_and_stats(
            path, users, frame_size
        )
        
        print(f"Training files: {len(train_files)}")
        print(f"Validation files: {len(val_files)}")
        print(f"Test files: {len(test_files)}")
        
        # Store for later
        self.test_files = test_files
        self.mean = mean
        self.std = std
        self.n_channels = n_channels
        
        # CREATE STREAMING DATASETS (memory efficient!)
        print("\nCreating streaming datasets (no pre-loading)...")
        print("This will load data on-demand during training...")
        
        train_dataset = create_streaming_dataset(
            train_files, frame_size, mean, std, batch_size,
            masking_percentage=0, num_blocks=1, shuffle=True,
            shuffle_buffer=shuffle_buffer
        )
        
        val_dataset = create_streaming_dataset(
            val_files, frame_size, mean, std, batch_size,
            masking_percentage=0, num_blocks=1, shuffle=False,
            shuffle_buffer=0
        )
        
        print("✓ Streaming datasets created (memory efficient!)")
        print("  - Data will be loaded file-by-file during training")
        print("  - No large memory spike expected")
        
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
        print("Training with streaming data (memory safe)...")
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config['train_epochs'],
            callbacks=[callback],
            verbose=1
        )
        
        best_val_acc = max(history.history['val_accuracy'])
        print(f"\n✓ Best validation accuracy: {best_val_acc:.4f}")
        
        # Save model
        self.base_model_path = self.models_dir / "base_model.h5"
        model.save(str(self.base_model_path))
        self.base_model = model
        
        print(f"✓ Model saved to: {self.base_model_path}")
        
        # Clean up
        print("\n✓ Cleaning up datasets...")
        del train_dataset, val_dataset
        gc.collect()
        tf.keras.backend.clear_session()
        print("✓ Memory cleaned")
        
        self.checkpoint_mgr.save_checkpoint(
            stage='base_model_trained',
            stage_data={
                'model_path': str(self.base_model_path),
                'best_val_acc': float(best_val_acc),
                'num_classes': num_classes,
                'n_channels': int(n_channels)
            },
            metadata={'step': 1}
        )
        
        return best_val_acc
    
    def step2_evaluate_masking_grid(self):
        """Evaluate masking grid - LOAD TEST PER-CONFIG (memory safe!)"""
        print("\n" + "="*70)
        print("STEP 2: Evaluating Masking Grid (Per-Config Loading)")
        print("="*70)
        
        n_mask = len(self.masking_percentages)
        n_blocks = len(self.num_blocks_range)
        accuracy_matrix = np.full((n_mask, n_blocks), np.nan)
        kappa_matrix = np.full((n_mask, n_blocks), np.nan)
        detailed_results = []
        
        if self.base_model is None:
            print("Loading base model...")
            self.base_model = tf.keras.models.load_model(str(self.base_model_path))
        
        frame_size = self.config['frame_size']
        batch_size = self.config['batch_size']
        num_classes = len(self.config['user_ids'])
        
        total_configs = n_mask * n_blocks
        completed = 0
        start_time = time.time()
        
        for mask_idx, mask_pct in enumerate(self.masking_percentages):
            for block_idx, n_blocks_val in enumerate(self.num_blocks_range):
                completed += 1
                
                print(f"\nConfiguration {completed}/{total_configs}: "
                      f"{mask_pct}% masking, {n_blocks_val} blocks")
                
                try:
                    # Load test data FOR THIS CONFIG ONLY (memory safe!)
                    print(f"  Loading test data with {mask_pct}% masking, {n_blocks_val} blocks...")
                    X_test, y_test = preload_data_fast(
                        self.test_files, frame_size, self.mean, self.std,
                        mask_pct, n_blocks_val, f"test {mask_pct}%/{n_blocks_val}blk"
                    )
                    
                    # Create dataset
                    test_dataset = create_fast_dataset(X_test, y_test, batch_size, shuffle=False)
                    
                    # Evaluate
                    print("  Evaluating...")
                    results = self.base_model.evaluate(test_dataset, verbose=0)
                    test_acc = results[1]
                    
                    # Calculate Kappa
                    print("  Computing Kappa...")
                    y_pred = self.base_model.predict(test_dataset, verbose=0)
                    kappa_score = cohen_kappa_score_fast(y_test, y_pred, num_classes)
                    
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
                    
                    # CRITICAL: Free test data immediately (memory safe!)
                    del X_test, y_test, test_dataset, y_pred
                    gc.collect()
                    print("  ✓ Test data freed from memory")
                    
                    # Save checkpoint
                    self.checkpoint_mgr.save_checkpoint(
                        stage='evaluating_grid',
                        stage_data={
                            'completed': completed,
                            'total': total_configs
                        },
                        accuracy_matrix=accuracy_matrix,
                        kappa_matrix=kappa_matrix,
                        detailed_results=detailed_results,
                        metadata={'step': 2}
                    )
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
        
        elapsed = time.time() - start_time
        print(f"\n✓ Grid evaluation completed in {elapsed/3600:.2f} hours")
        
        return accuracy_matrix, kappa_matrix, detailed_results
    
    def step3_generate_plots(self, accuracy_matrix, kappa_matrix):
        """Generate plots"""
        print("\n" + "="*70)
        print("STEP 3: Generating Plots")
        print("="*70)
        
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10
        
        # Accuracy heatmap
        plt.figure(figsize=(12, 8))
        im = plt.imshow(accuracy_matrix, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(im, label='Test Accuracy')
        plt.xlabel('Number of Blocks', fontsize=12)
        plt.ylabel('Masking Percentage (%)', fontsize=12)
        plt.title('Test Accuracy: Masking Percentage vs Number of Blocks', fontsize=14, fontweight='bold')
        plt.xticks(range(len(self.num_blocks_range)), self.num_blocks_range)
        plt.yticks(range(len(self.masking_percentages)), self.masking_percentages)
        for i in range(len(self.masking_percentages)):
            for j in range(len(self.num_blocks_range)):
                if not np.isnan(accuracy_matrix[i, j]):
                    plt.text(j, i, f'{accuracy_matrix[i, j]:.3f}', ha="center", va="center", color="white", fontsize=8)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Kappa heatmap
        plt.figure(figsize=(12, 8))
        im = plt.imshow(kappa_matrix, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(im, label="Cohen's Kappa")
        plt.xlabel('Number of Blocks', fontsize=12)
        plt.ylabel('Masking Percentage (%)', fontsize=12)
        plt.title("Cohen's Kappa: Masking Percentage vs Number of Blocks", fontsize=14, fontweight='bold')
        plt.xticks(range(len(self.num_blocks_range)), self.num_blocks_range)
        plt.yticks(range(len(self.masking_percentages)), self.masking_percentages)
        for i in range(len(self.masking_percentages)):
            for j in range(len(self.num_blocks_range)):
                if not np.isnan(kappa_matrix[i, j]):
                    plt.text(j, i, f'{kappa_matrix[i, j]:.3f}', ha="center", va="center", color="white", fontsize=8)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'kappa_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Line plots
        plt.figure(figsize=(12, 8))
        for block_idx, n_blocks in enumerate(self.num_blocks_range):
            plt.plot(self.masking_percentages, accuracy_matrix[:, block_idx], marker='o', label=f'{n_blocks} blocks', linewidth=2, markersize=8)
        plt.xlabel('Masking Percentage (%)', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.title('Test Accuracy vs Masking Percentage', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_vs_masking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        for block_idx, n_blocks in enumerate(self.num_blocks_range):
            plt.plot(self.masking_percentages, kappa_matrix[:, block_idx], marker='o', label=f'{n_blocks} blocks', linewidth=2, markersize=8)
        plt.xlabel('Masking Percentage (%)', fontsize=12)
        plt.ylabel("Cohen's Kappa", fontsize=12)
        plt.title("Cohen's Kappa vs Masking Percentage", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'kappa_vs_masking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        for mask_idx, mask_pct in enumerate(self.masking_percentages[::2]):
            actual_idx = mask_idx * 2
            plt.plot(self.num_blocks_range, accuracy_matrix[actual_idx, :], marker='o', label=f'{mask_pct}% masking', linewidth=2, markersize=8)
        plt.xlabel('Number of Blocks', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.title('Test Accuracy vs Number of Blocks', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_vs_blocks.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        for mask_idx, mask_pct in enumerate(self.masking_percentages[::2]):
            actual_idx = mask_idx * 2
            plt.plot(self.num_blocks_range, kappa_matrix[actual_idx, :], marker='o', label=f'{mask_pct}% masking', linewidth=2, markersize=8)
        plt.xlabel('Number of Blocks', fontsize=12)
        plt.ylabel("Cohen's Kappa", fontsize=12)
        plt.title("Cohen's Kappa vs Number of Blocks", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'kappa_vs_blocks.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ All plots saved")
    
    def step4_save_results(self, accuracy_matrix, kappa_matrix, detailed_results, base_val_acc):
        """Save results"""
        print("\n" + "="*70)
        print("STEP 4: Saving Results")
        print("="*70)
        
        results_file = self.results_dir / "masking_results.npz"
        np.savez(results_file, accuracy_matrix=accuracy_matrix, kappa_matrix=kappa_matrix,
                 masking_percentages=self.masking_percentages, num_blocks_range=self.num_blocks_range,
                 base_val_acc=base_val_acc)
        print(f"✓ Saved {results_file}")
        
        results_json = {
            'experiment_info': {
                'experiment_dir': str(self.experiment_dir),
                'base_val_acc': float(base_val_acc) if base_val_acc else None,
                'num_users': len(self.config['user_ids']),
            },
            'results': {
                'accuracy_matrix': accuracy_matrix.tolist(),
                'kappa_matrix': kappa_matrix.tolist(),
                'detailed_results': detailed_results
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.results_dir / "experiment_results.json", 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"✓ Saved JSON results")
        
        with open(self.results_dir / "summary.txt", 'w') as f:
            f.write("="*70 + "\n")
            f.write("EEG MASKING EXPERIMENT SUMMARY (STREAMING)\n")
            f.write("="*70 + "\n\n")
            f.write(f"Baseline accuracy: {accuracy_matrix[0, 0]:.4f}\n" if not np.isnan(accuracy_matrix[0, 0]) else "")
            valid_mask = ~np.isnan(accuracy_matrix[1:])
            if valid_mask.any():
                masked_accuracies = accuracy_matrix[1:][valid_mask]
                f.write(f"Best masked: {np.max(masked_accuracies):.4f}\n")
                f.write(f"Worst masked: {np.min(masked_accuracies):.4f}\n")
        print(f"✓ Saved summary")
        
        return results_json
    
    def run_complete_experiment(self, resume=False):
        """Run complete experiment"""
        experiment_start_time = time.time()
        
        print("\n" + "="*70)
        print("STARTING STREAMING EXPERIMENT (Memory Safe Training + Eval)")
        print("="*70)
        
        try:
            if resume and self.checkpoint_mgr.has_checkpoint():
                checkpoint = self.checkpoint_mgr.load_checkpoint()
                if checkpoint['stage'] == 'complete':
                    print("Experiment already complete!")
                    return checkpoint.get('results', None)
            
            base_val_acc = self.step1_train_base_model()
            accuracy_matrix, kappa_matrix, detailed_results = self.step2_evaluate_masking_grid()
            self.step3_generate_plots(accuracy_matrix, kappa_matrix)
            results_data = self.step4_save_results(accuracy_matrix, kappa_matrix, detailed_results, base_val_acc)
            
            self.checkpoint_mgr.save_checkpoint(
                stage='complete',
                stage_data={'experiment_complete': True},
                accuracy_matrix=accuracy_matrix,
                kappa_matrix=kappa_matrix,
                detailed_results=detailed_results
            )
            
            experiment_time = time.time() - experiment_start_time
            print("\n" + "="*70)
            print("EXPERIMENT COMPLETED")
            print("="*70)
            print(f"Total time: {experiment_time/3600:.2f} hours")
            print(f"Baseline accuracy: {accuracy_matrix[0,0]:.4f}")
            
            return results_data
            
        except Exception as e:
            print(f"\nEXPERIMENT FAILED: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main execution"""
    
    config = {
        'experiment_dir': '/app/data/experiments/masking_da_streaming',
        'raw_eeg_dir': '/app/data/1.0.0',
        'user_ids': list(range(1, 110)),
        'train_epochs': 100,
        'batch_size': 32,  # Can adjust based on memory
        'learning_rate': 0.001,
        'frame_size': 40,
        'shuffle_buffer': 1000,  # Smaller buffer = less memory usage
    }
    
    print("EEG Masking Experiment (STREAMING - Memory Safe)")
    print("=" * 70)
    print(f"Users: {len(config['user_ids'])} users")
    print(f"Strategy: Streaming datasets for train/val, per-config for test")
    print(f"Batch size: {config['batch_size']}")
    print(f"Shuffle buffer: {config['shuffle_buffer']} (smaller = less memory)")
    print("=" * 70)
    
    experiment = EEGMaskingExperiment(config)
    results = experiment.run_complete_experiment(resume=False)
    
    if results is not None:
        print("\n✓ Experiment completed successfully!")
        return True
    else:
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
