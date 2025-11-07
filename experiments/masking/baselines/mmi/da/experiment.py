"""
EEG Masking Experiment for Data Augmentation Method
Matches experiment_new.py structure but works directly with EEG data (.edf files)
No spectrograms or JEPA preprocessing needed - uses data augmentation training approach
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from pathlib import Path
import gc

from backbones import *
from data_loader import load_edf_session, data_load_eeg_mmi_streaming, get_file_list_and_labels
from optimized_data_loader import create_tf_dataset


class MaskingExperimentDA:
    """
    EEG Masking Experiment using Data Augmentation training method.
    Applies masking directly to raw EEG time series data.
    """
    
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
        
        # Masking parameter grid (matches experiment_new.py)
        self.masking_percentages = np.arange(0, 55, 5)  # 0%, 5%, 10%, ..., 50%
        self.num_blocks_range = [1, 2, 3, 4, 5]
        
        # Checkpoint file
        self.checkpoint_file = self.checkpoint_dir / "experiment_checkpoint.json"
        self.results_file = self.checkpoint_dir / "partial_results.npz"
        
        # Model path
        self.base_model_path = None
        
        print(f"EEG Masking Experiment (Data Augmentation Method)")
        print(f"=" * 70)
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Grid size: {len(self.masking_percentages)} × {len(self.num_blocks_range)} = {len(self.masking_percentages) * len(self.num_blocks_range)}")
        print(f"Number of users: {len(config['user_ids'])}")
        print(f"=" * 70)
    
    def load_checkpoint(self):
        """Load checkpoint if it exists"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"\n{'=' * 70}")
            print("RESUMING FROM CHECKPOINT")
            print(f"Stage: {checkpoint.get('stage', 'unknown')}")
            print(f"{'=' * 70}\n")
            return checkpoint
        return None
    
    def save_checkpoint(self, stage, stage_data, accuracy_matrix=None, kappa_matrix=None, detailed_results=None):
        """Save checkpoint"""
        checkpoint = {
            'stage': stage,
            'stage_data': stage_data,
            'timestamp': time.time()
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Save partial results if available
        if accuracy_matrix is not None and kappa_matrix is not None:
            np.savez(
                self.results_file,
                accuracy_matrix=accuracy_matrix,
                kappa_matrix=kappa_matrix,
                detailed_results=detailed_results if detailed_results else []
            )
    
    def step1_train_base_model(self):
        """
        Train the base model on unmasked training data.
        Uses sessions 1-10 for training, 11-12 for validation.
        MEMORY-OPTIMIZED: Reduced batch size and aggressive cleanup.
        """
        print("\n" + "=" * 70)
        print("STEP 1: Training Base Model (Unmasked Data)")
        print("=" * 70)
        
        frame_size = self.config['frame_size']
        batch_size = 32  # REDUCED from 64 to save memory with 109 users
        path = self.config['raw_eeg_dir']
        users = self.config['user_ids']
        
        print(f"Using reduced batch size: {batch_size} (memory-optimized for {len(users)} users)")
        
        # Load data using streaming approach
        train_files, val_files, test_files, mean, std, n_channels, train_samples, val_samples, test_samples = \
            data_load_eeg_mmi_streaming(path, users=users, frame_size=frame_size)
        
        print(f"Training samples: {train_samples}")
        print(f"Validation samples: {val_samples}")
        print(f"Testing samples: {test_samples}")
        
        num_classes = len(users)
        print(f"Number of classes: {num_classes}")
        print(f"Number of channels: {n_channels}")
        
        # Create datasets
        print("\nCreating data pipelines...")
        train_dataset, _ = create_tf_dataset(
            train_files,
            frame_size,
            mean,
            std,
            batch_size=batch_size,
            shuffle=True,
            augment=True
        )
        
        val_dataset, _ = create_tf_dataset(
            val_files,
            frame_size,
            mean,
            std,
            batch_size=batch_size,
            shuffle=False,
            augment=False
        )
        
        # Build model (matches trainers.py architecture)
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
        
        # Enable mixed precision
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision training enabled")
        except:
            print("Mixed precision not available, using float32")
        
        # Compile
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            restore_best_weights=True,
            patience=5
        )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
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
        print(f"\nStarting training for {self.config['train_epochs']} epochs...")
        print("Note: Training with 109 users may take considerable time")
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
        self.base_model_path = self.models_dir / "base_model.keras"
        model.save(self.base_model_path)
        print(f"Model saved to: {self.base_model_path}")
        
        # Save normalization stats
        norm_stats = {
            'mean': mean.tolist(),
            'std': std.tolist(),
            'n_channels': int(n_channels),
            'frame_size': int(frame_size),
            'num_classes': int(num_classes)
        }
        with open(self.models_dir / "norm_stats.json", 'w') as f:
            json.dump(norm_stats, f, indent=2)
        
        # Aggressive cleanup
        del model, history, train_dataset, val_dataset, optimizer, lr_schedule, callback
        del train_files, val_files  # Don't need these anymore
        tf.keras.backend.clear_session()
        gc.collect()
        
        print("Training complete and memory cleared")
        
        return best_val_acc, test_files, mean, std
    
    def apply_time_series_masking(self, data, masking_percentage, num_blocks):
        """
        Apply masking to time series EEG data.
        
        Args:
            data: numpy array of shape (batch_size, frame_size, n_channels) or (frame_size, n_channels)
            masking_percentage: Percentage of time steps to mask (0-100)
            num_blocks: Number of contiguous blocks to mask
        
        Returns:
            Masked data with same shape as input
        """
        if masking_percentage == 0:
            return data
        
        masked_data = data.copy()
        
        # Handle both batched and single sample inputs
        if len(masked_data.shape) == 2:
            masked_data = masked_data[np.newaxis, :, :]  # Add batch dimension
            squeeze_after = True
        else:
            squeeze_after = False
        
        batch_size, frame_size, n_channels = masked_data.shape
        
        # Calculate total time steps to mask
        total_to_mask = int(frame_size * masking_percentage / 100)
        
        if total_to_mask == 0:
            return data
        
        # Calculate block size
        block_size = total_to_mask // num_blocks
        
        if block_size == 0:
            block_size = 1
            num_blocks = total_to_mask
        
        # Apply masking to each sample in batch
        for b in range(batch_size):
            # Randomly select start positions for blocks
            for _ in range(num_blocks):
                if frame_size - block_size > 0:
                    start_pos = np.random.randint(0, frame_size - block_size + 1)
                else:
                    start_pos = 0
                
                # Mask the block (set to zero)
                end_pos = min(start_pos + block_size, frame_size)
                masked_data[b, start_pos:end_pos, :] = 0
        
        if squeeze_after:
            masked_data = masked_data[0]  # Remove batch dimension
        
        return masked_data
    
    def evaluate_masked(self, model, test_files, mean, std, masking_percentage, num_blocks):
        """
        Evaluate model on masked test data.
        MEMORY-OPTIMIZED: Process files one at a time, accumulate metrics incrementally.
        
        Args:
            model: Trained TensorFlow model
            test_files: List of test file info
            mean, std: Normalization statistics
            masking_percentage: Percentage to mask
            num_blocks: Number of blocks
        
        Returns:
            accuracy, kappa_score
        """
        frame_size = self.config['frame_size']
        batch_size = 32  # Reduced batch size for evaluation to save memory
        num_classes = len(self.config['user_ids'])
        
        # For Cohen's Kappa, we need the full confusion matrix
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        total_samples = 0
        correct_predictions = 0
        
        # Process test files ONE AT A TIME
        for file_idx, (filepath, user_id, n_windows) in enumerate(test_files):
            # Load single file
            data = load_edf_session(filepath, frame_size)
            if data is None:
                continue
            
            # Normalize
            data_flat = data.reshape(-1, data.shape[2])
            data_flat = (data_flat - mean) / std
            data = data_flat.reshape(data.shape).astype(np.float32)
            del data_flat
            
            # Apply masking
            data_masked = self.apply_time_series_masking(data, masking_percentage, num_blocks)
            del data
            
            # Predict in small batches
            file_predictions = []
            for i in range(0, len(data_masked), batch_size):
                batch = data_masked[i:i+batch_size]
                predictions = model.predict(batch, verbose=0)
                file_predictions.append(predictions)
            
            # Concatenate predictions for this file only
            file_predictions = np.concatenate(file_predictions, axis=0)
            file_pred_classes = np.argmax(file_predictions, axis=1)
            
            # Update confusion matrix and accuracy
            for pred_class in file_pred_classes:
                confusion_matrix[user_id, pred_class] += 1
                if pred_class == user_id:
                    correct_predictions += 1
                total_samples += 1
            
            # Immediate cleanup after each file
            del data_masked, file_predictions, file_pred_classes
            gc.collect()
            
            # More aggressive cleanup every 10 files
            if (file_idx + 1) % 10 == 0:
                tf.keras.backend.clear_session()
                # Reload model to clear any cached states
                model = tf.keras.models.load_model(self.base_model_path)
                gc.collect()
        
        # Calculate accuracy
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        # Calculate Cohen's Kappa from confusion matrix
        n = total_samples
        observed_accuracy = np.trace(confusion_matrix) / n
        
        expected_accuracy = 0
        for i in range(num_classes):
            expected_accuracy += (np.sum(confusion_matrix[i, :]) * np.sum(confusion_matrix[:, i])) / (n * n)
        
        if expected_accuracy == 1.0:
            kappa = 1.0
        else:
            kappa = (observed_accuracy - expected_accuracy) / (1.0 - expected_accuracy)
        
        return accuracy, kappa
    
    def cohen_kappa_score(self, y_true, y_pred, num_classes):
        """Calculate Cohen's Kappa score (matches trainers.py implementation)"""
        # Convert predictions to class labels if needed
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Ensure arrays are 1D
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Build confusion matrix
        confusion_matrix = np.zeros((num_classes, num_classes))
        for i in range(len(y_true)):
            confusion_matrix[int(y_true[i]), int(y_pred[i])] += 1
        
        # Calculate observed accuracy
        n = len(y_true)
        observed_accuracy = np.trace(confusion_matrix) / n
        
        # Calculate expected accuracy
        expected_accuracy = 0
        for i in range(num_classes):
            expected_accuracy += (np.sum(confusion_matrix[i, :]) * np.sum(confusion_matrix[:, i])) / (n * n)
        
        # Calculate Cohen's Kappa
        if expected_accuracy == 1.0:
            kappa = 1.0
        else:
            kappa = (observed_accuracy - expected_accuracy) / (1.0 - expected_accuracy)
        
        return kappa
    
    def step2_evaluate_all_variants(self, test_files, mean, std):
        """
        Evaluate trained model on all masking variants.
        Matches step 3 from experiment_new.py.
        MEMORY-OPTIMIZED: Reload model periodically to clear memory.
        """
        print("\n" + "=" * 70)
        print("STEP 2: Evaluating All Masking Variants")
        print("=" * 70)
        
        # Initialize results matrices
        n_percentages = len(self.masking_percentages)
        n_blocks = len(self.num_blocks_range)
        accuracy_matrix = np.full((n_percentages, n_blocks), np.nan)
        kappa_matrix = np.full((n_percentages, n_blocks), np.nan)
        detailed_results = []
        
        # Load checkpoint if exists
        checkpoint = self.load_checkpoint()
        completed_variants = []
        if checkpoint and checkpoint.get('stage') == 'evaluation':
            completed_variants = checkpoint['stage_data'].get('completed_variants', [])
            # Load partial results
            if self.results_file.exists():
                data = np.load(self.results_file, allow_pickle=True)
                accuracy_matrix = data['accuracy_matrix']
                kappa_matrix = data['kappa_matrix']
                detailed_results = data['detailed_results'].tolist()
            print(f"Resuming from checkpoint: {len(completed_variants)} variants completed")
        
        # Load trained model initially
        print(f"Loading model from: {self.base_model_path}")
        model = tf.keras.models.load_model(self.base_model_path)
        
        # Iterate through all variants
        total_variants = n_percentages * n_blocks
        completed_count = len(completed_variants)
        
        for p_idx, masking_pct in enumerate(self.masking_percentages):
            for b_idx, num_blocks in enumerate(self.num_blocks_range):
                variant_key = f"{masking_pct}_{num_blocks}"
                
                # Skip if already completed
                if variant_key in completed_variants:
                    continue
                
                completed_count += 1
                print(f"\n[{completed_count}/{total_variants}] Evaluating: {masking_pct}% masking, {num_blocks} blocks")
                
                start_time = time.time()
                
                # Evaluate (note: evaluate_masked will reload model every 10 files internally)
                accuracy, kappa = self.evaluate_masked(
                    model, test_files, mean, std,
                    masking_pct, num_blocks
                )
                
                eval_time = time.time() - start_time
                
                # Store results
                accuracy_matrix[p_idx, b_idx] = accuracy
                kappa_matrix[p_idx, b_idx] = kappa
                
                result = {
                    'masking_percentage': int(masking_pct),
                    'num_blocks': int(num_blocks),
                    'accuracy': float(accuracy),
                    'kappa': float(kappa),
                    'eval_time': float(eval_time)
                }
                detailed_results.append(result)
                
                print(f"  Accuracy: {accuracy:.4f}, Kappa: {kappa:.4f}, Time: {eval_time:.1f}s")
                
                # Update checkpoint
                completed_variants.append(variant_key)
                self.save_checkpoint(
                    stage='evaluation',
                    stage_data={'completed_variants': completed_variants},
                    accuracy_matrix=accuracy_matrix,
                    kappa_matrix=kappa_matrix,
                    detailed_results=detailed_results
                )
                
                # AGGRESSIVE cleanup every 5 variants
                if completed_count % 5 == 0:
                    print("  [Memory cleanup: clearing session and reloading model]")
                    del model
                    tf.keras.backend.clear_session()
                    gc.collect()
                    # Reload model fresh
                    model = tf.keras.models.load_model(self.base_model_path)
        
        print(f"\nEvaluation complete: {total_variants} variants evaluated")
        
        # Final cleanup
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        
        return accuracy_matrix, kappa_matrix, detailed_results
    
    def step3_generate_plots(self, accuracy_matrix, kappa_matrix, base_val_acc):
        """
        Generate visualization plots (matches step 4 from experiment_new.py).
        """
        print("\n" + "=" * 70)
        print("STEP 3: Generating Plots")
        print("=" * 70)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Heatmap: Accuracy vs Masking Configuration
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            accuracy_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=self.num_blocks_range,
            yticklabels=[f"{p}%" for p in self.masking_percentages],
            cbar_kws={'label': 'Accuracy'},
            vmin=0, vmax=1
        )
        plt.title('Test Accuracy vs Masking Configuration', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Blocks', fontsize=14)
        plt.ylabel('Masking Percentage', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Heatmap: Kappa Score vs Masking Configuration
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            kappa_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=self.num_blocks_range,
            yticklabels=[f"{p}%" for p in self.masking_percentages],
            cbar_kws={'label': 'Kappa Score'},
            vmin=0, vmax=1
        )
        plt.title('Kappa Score vs Masking Configuration', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Blocks', fontsize=14)
        plt.ylabel('Masking Percentage', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'kappa_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Line plot: Accuracy vs Masking Percentage (for each num_blocks)
        plt.figure(figsize=(12, 8))
        for b_idx, num_blocks in enumerate(self.num_blocks_range):
            plt.plot(
                self.masking_percentages,
                accuracy_matrix[:, b_idx],
                marker='o',
                linewidth=2,
                markersize=8,
                label=f'{num_blocks} blocks'
            )
        plt.axhline(y=base_val_acc, color='red', linestyle='--', linewidth=2, label='Baseline (0% masking)')
        plt.xlabel('Masking Percentage (%)', fontsize=14)
        plt.ylabel('Test Accuracy', fontsize=14)
        plt.title('Test Accuracy vs Masking Percentage', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_vs_percentage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Line plot: Kappa vs Masking Percentage
        plt.figure(figsize=(12, 8))
        for b_idx, num_blocks in enumerate(self.num_blocks_range):
            plt.plot(
                self.masking_percentages,
                kappa_matrix[:, b_idx],
                marker='o',
                linewidth=2,
                markersize=8,
                label=f'{num_blocks} blocks'
            )
        plt.xlabel('Masking Percentage (%)', fontsize=14)
        plt.ylabel('Kappa Score', fontsize=14)
        plt.title('Kappa Score vs Masking Percentage', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'kappa_vs_percentage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Bar plot: Accuracy vs Number of Blocks (for each masking percentage)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        percentages_to_plot = [0, 10, 20, 30, 40, 50]
        for idx, pct in enumerate(percentages_to_plot):
            if idx >= len(axes):
                break
            p_idx = list(self.masking_percentages).index(pct)
            axes[idx].bar(self.num_blocks_range, accuracy_matrix[p_idx, :])
            axes[idx].set_title(f'{pct}% Masking', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Number of Blocks', fontsize=10)
            axes[idx].set_ylabel('Accuracy', fontsize=10)
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_by_blocks.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"All plots saved to: {self.plots_dir}")
    
    def step4_save_results(self, accuracy_matrix, kappa_matrix, detailed_results, base_val_acc):
        """
        Save all results (matches step 5 from experiment_new.py).
        """
        print("\n" + "=" * 70)
        print("STEP 4: Saving Results")
        print("=" * 70)
        
        # Save matrices
        np.savez(
            self.results_dir / 'results_matrices.npz',
            accuracy_matrix=accuracy_matrix,
            kappa_matrix=kappa_matrix,
            masking_percentages=self.masking_percentages,
            num_blocks_range=self.num_blocks_range
        )
        
        # Save detailed results
        results_data = {
            'experiment_config': self.config,
            'base_validation_accuracy': float(base_val_acc),
            'masking_percentages': self.masking_percentages.tolist(),
            'num_blocks_range': self.num_blocks_range,
            'detailed_results': detailed_results,
            'summary_statistics': {
                'baseline_accuracy': float(accuracy_matrix[0, 0]),
                'baseline_kappa': float(kappa_matrix[0, 0]),
                'min_accuracy': float(np.nanmin(accuracy_matrix[1:])),
                'max_accuracy': float(np.nanmax(accuracy_matrix[1:])),
                'mean_accuracy': float(np.nanmean(accuracy_matrix[1:])),
                'min_kappa': float(np.nanmin(kappa_matrix[1:])),
                'max_kappa': float(np.nanmax(kappa_matrix[1:])),
                'mean_kappa': float(np.nanmean(kappa_matrix[1:]))
            }
        }
        
        with open(self.results_dir / 'experiment_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to: {self.results_dir}")
        
        return results_data
    
    def run_complete_experiment(self, resume=False):
        """
        Run the complete masking experiment pipeline.
        Matches run_complete_experiment from experiment_new.py.
        """
        experiment_start_time = time.time()
        
        try:
            # Check for checkpoint
            checkpoint = self.load_checkpoint()
            
            if checkpoint and not resume:
                print("\n" + "!" * 70)
                print("WARNING: Existing checkpoint found but resume=False")
                print("The experiment will start from scratch and overwrite existing progress.")
                print("Set resume=True to continue from checkpoint.")
                print("!" * 70)
                response = input("\nContinue anyway? (yes/no): ")
                if response.lower() != 'yes':
                    print("Experiment cancelled.")
                    return None
                checkpoint = None
            
            # STEP 1: Train base model (or load if checkpoint exists)
            if checkpoint and checkpoint['stage'] in ['trained', 'evaluation', 'plotting', 'complete']:
                print("\nSkipping training - loading from checkpoint")
                
                # Get base_val_acc (may not exist in older checkpoints)
                base_val_acc = checkpoint['stage_data'].get('base_val_acc', 0.0)
                
                # Get model path - try checkpoint first, then infer from directory
                if 'base_model_path' in checkpoint['stage_data']:
                    self.base_model_path = Path(checkpoint['stage_data']['base_model_path'])
                else:
                    # Infer model path from experiment directory
                    self.base_model_path = self.models_dir / "base_model.keras"
                    print(f"Model path not in checkpoint, using default: {self.base_model_path}")
                
                # Check if model exists
                if not self.base_model_path.exists():
                    print(f"\nERROR: Model file not found at {self.base_model_path}")
                    print("The checkpoint indicates training was done, but the model file is missing.")
                    print("You may need to retrain from scratch.")
                    return None
                
                # Load normalization stats
                norm_stats_path = self.models_dir / "norm_stats.json"
                if not norm_stats_path.exists():
                    print(f"\nERROR: Normalization stats not found at {norm_stats_path}")
                    print("The checkpoint indicates training was done, but norm stats are missing.")
                    print("You may need to retrain from scratch.")
                    return None
                    
                with open(norm_stats_path, 'r') as f:
                    norm_stats = json.load(f)
                mean = np.array(norm_stats['mean'], dtype=np.float32)
                std = np.array(norm_stats['std'], dtype=np.float32)
                
                print(f"Loaded model from: {self.base_model_path}")
                print(f"Loaded normalization stats: mean shape {mean.shape}, std shape {std.shape}")
                
                # Get test files
                path = self.config['raw_eeg_dir']
                users = self.config['user_ids']
                frame_size = self.config['frame_size']
                print("Loading test file list...")
                _, _, test_files, _ = get_file_list_and_labels(path, users, frame_size)
                print(f"Found {len(test_files)} test files")
            else:
                base_val_acc, test_files, mean, std = self.step1_train_base_model()
                
                # Save checkpoint
                self.save_checkpoint(
                    stage='trained',
                    stage_data={
                        'base_model_path': str(self.base_model_path),
                        'base_val_acc': float(base_val_acc)
                    }
                )
            
            # STEP 2: Evaluate all masking variants
            if checkpoint and checkpoint['stage'] in ['plotting', 'complete']:
                print("\nSkipping evaluation - loading results from checkpoint")
                data = np.load(self.results_file, allow_pickle=True)
                accuracy_matrix = data['accuracy_matrix']
                kappa_matrix = data['kappa_matrix']
                detailed_results = data['detailed_results'].tolist()
            else:
                accuracy_matrix, kappa_matrix, detailed_results = self.step2_evaluate_all_variants(
                    test_files, mean, std
                )
                
                # Save checkpoint
                self.save_checkpoint(
                    stage='evaluated',
                    stage_data={
                        'base_model_path': str(self.base_model_path),
                        'base_val_acc': float(base_val_acc)
                    },
                    accuracy_matrix=accuracy_matrix,
                    kappa_matrix=kappa_matrix,
                    detailed_results=detailed_results
                )
            
            # STEP 3: Generate plots
            self.step3_generate_plots(accuracy_matrix, kappa_matrix, base_val_acc)
            
            # STEP 4: Save results
            results_data = self.step4_save_results(accuracy_matrix, kappa_matrix, detailed_results, base_val_acc)
            
            # Mark complete
            self.save_checkpoint(
                stage='complete',
                stage_data={
                    'base_model_path': str(self.base_model_path),
                    'base_val_acc': float(base_val_acc),
                    'experiment_complete': True
                }
            )
            
            # Final summary
            experiment_time = time.time() - experiment_start_time
            
            print("\n" + "=" * 70)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"Total time: {experiment_time / 3600:.2f} hours")
            print(f"Baseline accuracy (0% masking): {accuracy_matrix[0, 0]:.4f}")
            print(f"Results directory: {self.experiment_dir}")
            
            # Key findings
            valid_mask = ~np.isnan(accuracy_matrix[1:])
            if valid_mask.any():
                masked_accuracies = accuracy_matrix[1:][valid_mask]
                best_masked_acc = np.max(masked_accuracies)
                worst_masked_acc = np.min(masked_accuracies)
                
                print(f"Best masked performance: {best_masked_acc:.4f}")
                print(f"Worst masked performance: {worst_masked_acc:.4f}")
                print(f"Performance range: {best_masked_acc - worst_masked_acc:.4f}")
                
                if not np.isnan(accuracy_matrix[0, 0]):
                    degradation_pct = (accuracy_matrix[0, 0] - best_masked_acc) / accuracy_matrix[0, 0] * 100
                    print(f"Maximum degradation: {degradation_pct:.1f}%")
            
            # Clean up checkpoint files
            if self.checkpoint_file.exists():
                os.remove(self.checkpoint_file)
            if self.results_file.exists():
                os.remove(self.results_file)
            
            return results_data
        
        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print("EXPERIMENT INTERRUPTED BY USER")
            print("=" * 70)
            print("Progress saved. Resume with resume=True")
            return None
        
        except Exception as e:
            print(f"\nEXPERIMENT FAILED: {e}")
            import traceback
            traceback.print_exc()
            print("\nProgress saved. Try resuming with resume=True")
            return None


def main():
    """Main execution function"""
    
    # Experiment configuration (matches experiment_new.py structure)
    config = {
        # Paths
        'experiment_dir': '/app/data/experiments/da_masking_new',
        'raw_eeg_dir': '/app/data/1.0.0',
        
        # Local paths (comment out if using server)
        # 'experiment_dir': '/Users/belindahu/Desktop/thesis/masking_da_experiment',
        # 'raw_eeg_dir': '/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0',
        
        # Experiment parameters
        'user_ids': list(range(1, 110)),  # All 109 users
        'frame_size': 40,  # Matches trainers.py
        
        # Training parameters
        # NOTE: batch_size is overridden to 32 in the code for memory efficiency with 109 users
        # For fewer users, you can increase this in step1_train_base_model
        'train_epochs': 100,
        'batch_size': 32,  # Reduced from 64 to handle 109 users
        'learning_rate': 0.001,
        
        # Experiment metadata
        'description': 'EEG masking experiment using data augmentation method (direct EEG processing)'
    }
    
    print("EEG Masking Experiment - Data Augmentation Method")
    print("=" * 70)
    print(f"Users: {len(config['user_ids'])} users")
    print(f"Batch size: {config['batch_size']} (optimized for memory)")
    print(f"Experiment directory: {config['experiment_dir']}")
    print("=" * 70)
    print("\nMemory optimizations enabled:")
    print("  - Reduced batch size (32 for training, 32 for eval)")
    print("  - One-file-at-a-time evaluation")
    print("  - Periodic model reloading")
    print("  - Aggressive garbage collection")
    print("=" * 70)
    
    # Initialize and run experiment
    experiment = MaskingExperimentDA(config)
    
    # Run experiment (set resume=True to continue from checkpoint)
    results = experiment.run_complete_experiment(resume=True)
    
    if results is not None:
        print("\nExperiment completed successfully!")
        print(f"All results saved to: {experiment.experiment_dir}")
        return True
    else:
        print("\nExperiment did not complete.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
