"""
Main script to run the MusicID masking experiment with Data Augmentation
Tests model robustness by masking time series data in different patterns
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import your modules
from backbones import *
from data_loader import *
from transformations import DA_MagWarp


def apply_augmentation(x):
    """Apply data augmentation using numpy function"""
    return DA_MagWarp(x)

@tf.function
def tf_augment_wrapper(input_tensor):
    """TensorFlow wrapper for augmentation that handles float32 input"""
    # Convert to float64 and numpy for augmentation
    result = tf.numpy_function(apply_augmentation, [input_tensor], tf.float64)
    # Convert back to float32
    result = tf.cast(result, tf.float32)
    # Preserve shape
    result.set_shape(input_tensor.shape)
    return result


def apply_time_series_masking(x_data, masking_percentage, num_blocks, mask_value=0):
    """
    Apply masking to time series data
    
    Args:
        x_data: numpy array of shape (num_samples, time_steps, features)
        masking_percentage: percentage of time steps to mask (0-100)
        num_blocks: number of contiguous blocks to mask
        mask_value: value to use for masked regions (default: 0)
    
    Returns:
        masked_data: numpy array with same shape as input
    """
    if masking_percentage == 0:
        return x_data.copy()
    
    masked_data = x_data.copy()
    num_samples, time_steps, features = x_data.shape
    
    # Calculate total time steps to mask
    total_mask_steps = int(time_steps * masking_percentage / 100)
    
    if total_mask_steps == 0:
        return masked_data
    
    # Calculate block size
    block_size = total_mask_steps // num_blocks
    if block_size == 0:
        block_size = 1
        num_blocks = total_mask_steps
    
    # For each sample, randomly place the blocks
    for sample_idx in range(num_samples):
        masked_positions = []
        
        for block_idx in range(num_blocks):
            # Randomly choose start position for this block
            max_start = time_steps - block_size
            if max_start <= 0:
                start_pos = 0
            else:
                start_pos = np.random.randint(0, max_start + 1)
            
            end_pos = min(start_pos + block_size, time_steps)
            
            # Mask this block
            masked_data[sample_idx, start_pos:end_pos, :] = mask_value
            masked_positions.extend(range(start_pos, end_pos))
    
    return masked_data


def cohen_kappa_score(y_true, y_pred, num_classes):
    """
    Calculate Cohen's Kappa score manually
    """
    # Convert predictions to class labels if needed
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Create confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[int(y_true[i]), int(y_pred[i])] += 1
    
    # Calculate observed accuracy
    n = np.sum(confusion_matrix)
    po = np.trace(confusion_matrix) / n
    
    # Calculate expected accuracy
    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)
    
    # Calculate kappa
    if pe == 1:
        return 1.0
    kappa = (po - pe) / (1 - pe)
    
    return kappa


class MusicIDMaskingExperiment:
    """Complete MusicID masking experiment pipeline with data augmentation"""

    def __init__(self, config):
        self.config = config
        self.experiment_dir = Path(config['experiment_dir'])
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.models_dir = self.experiment_dir / "models"
        self.results_dir = self.experiment_dir / "results"
        self.plots_dir = self.experiment_dir / "plots"

        for d in [self.models_dir, self.results_dir, self.plots_dir]:
            d.mkdir(exist_ok=True)

        # Masking parameter grid
        self.masking_percentages = np.arange(0, 55, 5)  # 0%, 5%, 10%, ..., 50%
        self.num_blocks_range = [1, 2, 3, 4, 5]

        # Store trained model
        self.base_model = None

        print(f"MusicID Masking Experiment initialized")
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Grid size: {len(self.masking_percentages)} × {len(self.num_blocks_range)} = {len(self.masking_percentages) * len(self.num_blocks_range)}")
        print(f"Training with data augmentation (scaling)")

    def _load_data_with_validation(self, path, users, folders, frame_size, dataset_name):
        """Load data with proper validation and error handling"""
        sessions = []
        x_data = np.array([])
        y_data = []
        
        for user_id, user in enumerate(users):
            count = 0
            for folder in folders:
                for session in range(1, 6):
                    for typ in ["fav", "same"]:
                        filename = f"user{user}_{typ}_session{session}.csv"
                        filepath = os.path.join(path, folder, filename)
                        try:
                            file = pd.read_csv(filepath)
                            data = np.array(file.iloc[:, 1:25])
                            
                            # Create sliding windows
                            if data.shape[0] < frame_size:
                                continue
                            
                            # Use sliding window with stride
                            windows = np.lib.stride_tricks.sliding_window_view(
                                data, (frame_size, data.shape[1])
                            )
                            # Take every frame_size//2 steps (50% overlap)
                            windows = windows[::frame_size//2, 0, :, :]
                            
                            if windows.shape[2] != 24:
                                continue
                            
                            if x_data.shape[0] == 0:
                                x_data = windows
                                y_data += [user_id] * windows.shape[0]
                            else:
                                x_data = np.concatenate((x_data, windows), axis=0)
                                y_data += [user_id] * windows.shape[0]
                            
                            count += 1
                        except (FileNotFoundError, IndexError, ValueError) as e:
                            continue
            
            sessions.append(count)
        
        # Validate data was loaded
        if x_data.shape[0] == 0:
            print(f"ERROR: No {dataset_name} data loaded!")
            print(f"Checked path: {path}")
            print(f"Folders: {folders}")
            print(f"Users: {users}")
            # List what's actually in the directory
            for folder in folders:
                folder_path = os.path.join(path, folder)
                if os.path.exists(folder_path):
                    files = os.listdir(folder_path)
                    print(f"Files in {folder}: {files[:5]}")  # Show first 5 files
                else:
                    print(f"Folder does not exist: {folder_path}")
            return None
        
        print(f"  Loaded {x_data.shape[0]} samples from {sum(sessions)} files")
        print(f"  Data shape: {x_data.shape}")
        
        return x_data, np.array(y_data), sessions

    def step1_train_base_model(self):
        """Train the base model on unmasked training data with data augmentation"""
        print("\n" + "="*60)
        print("STEP 1: Training Base Model (With Data Augmentation)")
        print("="*60)

        frame_size = self.config['frame_size']
        BATCH_SIZE = self.config['batch_size']
        AUTO = tf.data.AUTOTUNE
        path = self.config['data_path']
        num_classes = self.config['num_classes']
        
        # Select users
        all_users = list(range(1, 21))
        users_to_use = all_users[:num_classes]
        print(f"Using {num_classes} classes (users): {users_to_use}")
        
        folder_train = ["TrainingSet"]
        folder_val = ["TestingSet"]
        
        # Load training data
        print("Loading training data...")
        result = self._load_data_with_validation(
            path, users_to_use, folder_train, frame_size, "Training"
        )
        if result is None:
            raise ValueError("Failed to load training data")
        x_train, y_train, sessions_train = result
        print(f"Training samples: {x_train.shape[0]}")
        
        # Load validation data
        print("Loading validation data...")
        result = self._load_data_with_validation(
            path, users_to_use, folder_val, frame_size, "Validation"
        )
        if result is None:
            raise ValueError("Failed to load validation data")
        x_val, y_val, sessions_val = result
        print(f"Validation samples: {x_val.shape[0]}")
        
        # Check if data was loaded successfully
        if x_train.shape[0] == 0 or x_val.shape[0] == 0:
            raise ValueError(f"No data loaded for {num_classes} classes")
        
        # Normalize data
        x_train, x_val_norm = self._normalize_data(x_train, x_val)
        
        classes, counts = np.unique(y_train, return_counts=True)
        num_classes_actual = len(classes)
        print(f"Number of classes: {num_classes_actual}")
        print(f"Samples per class: {counts}")
        
        # Convert to float32 for TensorFlow compatibility
        x_train = x_train.astype(np.float32)
        x_val_norm = x_val_norm.astype(np.float32)
        
        # Create TensorFlow datasets with data augmentation
        SEED = 34
        ds_x = tf.data.Dataset.from_tensor_slices(x_train)
        ds_x = (
            ds_x.shuffle(1024, seed=SEED)
            .map(tf_augment_wrapper, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
        )
        
        ds_y = tf.data.Dataset.from_tensor_slices(y_train)
        ds_y = (
            ds_y.shuffle(1024, seed=SEED)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
        )
        train_ds = tf.data.Dataset.zip((ds_x, ds_y))
        
        val_ds = tf.data.Dataset.from_tensor_slices((x_val_norm, y_val))
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)
        
        # Build model
        print("Building model...")
        ks = 3
        con = 3
        inputs = tf.keras.Input(shape=(frame_size, x_train.shape[-1]))
        x = tf.keras.layers.Conv1D(filters=16*con, kernel_size=ks, strides=1, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=4, strides=4)(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        x = resnetblock_final(x, CR=32*con, KS=ks)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(num_classes_actual, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)
        
        # Compile model
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
        
        # Train model
        print(f"Training for {self.config['train_epochs']} epochs...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config['train_epochs'],
            callbacks=[callback],
            batch_size=BATCH_SIZE,
            verbose=1
        )
        
        # Evaluate on validation set
        results = model.evaluate(x_val_norm, y_val, verbose=0)
        val_acc = results[1]
        print(f"Validation accuracy: {val_acc:.4f}")
        
        # Calculate validation kappa
        y_pred_val = model.predict(x_val_norm, verbose=0)
        val_kappa = cohen_kappa_score(y_val, y_pred_val, num_classes_actual)
        print(f"Validation kappa: {val_kappa:.4f}")
        
        # Save model
        model_path = self.models_dir / "base_model.keras"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Store model and statistics
        self.base_model = model
        self.train_mean = getattr(self, 'train_mean', None)
        self.train_std = getattr(self, 'train_std', None)
        self.num_classes_actual = num_classes_actual
        
        return val_acc, val_kappa

    def _normalize_data(self, x_train, x_val):
        """Normalize training and validation data"""
        # Reshape for normalization
        x = np.reshape(x_train, (x_train.shape[0]*x_train.shape[1], x_train.shape[2]))
        
        # Standardize
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std[std == 0] = 1
        
        x = (x - mean) / std
        x_train_norm = np.reshape(x, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        
        # Apply same normalization to validation
        x = np.reshape(x_val, (x_val.shape[0]*x_val.shape[1], x_val.shape[2]))
        x = (x - mean) / std
        x_val_norm = np.reshape(x, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
        
        # Store normalization parameters
        self.train_mean = mean
        self.train_std = std
        
        return x_train_norm, x_val_norm

    def step2_load_test_data(self):
        """Load test data (unmasked) - will be masked during evaluation"""
        print("\n" + "="*60)
        print("STEP 2: Loading Test Data")
        print("="*60)

        frame_size = self.config['frame_size']
        path = self.config['data_path']
        num_classes = self.config['num_classes']
        
        all_users = list(range(1, 21))
        users_to_use = all_users[:num_classes]
        
        folder_test = ["TestingSet_secret"]
        
        print("Loading test data...")
        result = self._load_data_with_validation(
            path, users_to_use, folder_test, frame_size, "Test"
        )
        if result is None:
            raise ValueError("Failed to load test data")
        x_test, y_test, sessions_test = result
        print(f"Test samples: {x_test.shape[0]}")
        
        if x_test.shape[0] == 0:
            raise ValueError("No test data loaded")
        
        # Normalize test data using training statistics
        x = np.reshape(x_test, (x_test.shape[0]*x_test.shape[1], x_test.shape[2]))
        x = (x - self.train_mean) / self.train_std
        x_test_norm = np.reshape(x, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
        
        # Convert to float32
        x_test_norm = x_test_norm.astype(np.float32)
        
        self.x_test = x_test_norm
        self.y_test = y_test
        
        print(f"Test data shape: {x_test_norm.shape}")
        
        return x_test_norm, y_test

    def step3_evaluate_masking_variants(self):
        """Evaluate model on all masking variants"""
        print("\n" + "="*60)
        print("STEP 3: Evaluating Masking Variants")
        print("="*60)

        # Initialize result matrices
        accuracy_matrix = np.full((len(self.masking_percentages), len(self.num_blocks_range)), np.nan)
        kappa_matrix = np.full((len(self.masking_percentages), len(self.num_blocks_range)), np.nan)
        
        total_variants = len(self.masking_percentages) * len(self.num_blocks_range)
        current_variant = 0
        
        detailed_results = []
        
        for pct_idx, masking_pct in enumerate(self.masking_percentages):
            for block_idx, num_blocks in enumerate(self.num_blocks_range):
                current_variant += 1
                
                print(f"\n[{current_variant}/{total_variants}] Testing: {masking_pct}% masking, {num_blocks} blocks")
                
                # Apply masking to test data
                x_test_masked = apply_time_series_masking(
                    self.x_test.copy(),
                    masking_percentage=masking_pct,
                    num_blocks=num_blocks,
                    mask_value=0
                )
                
                # Ensure float32 dtype
                x_test_masked = x_test_masked.astype(np.float32)
                
                # Evaluate
                results = self.base_model.evaluate(x_test_masked, self.y_test, verbose=0)
                test_acc = results[1]
                
                # Calculate kappa
                y_pred = self.base_model.predict(x_test_masked, verbose=0)
                kappa_score = cohen_kappa_score(self.y_test, y_pred, self.num_classes_actual)
                
                print(f"  Accuracy: {test_acc:.4f}, Kappa: {kappa_score:.4f}")
                
                # Store results
                accuracy_matrix[pct_idx, block_idx] = test_acc
                kappa_matrix[pct_idx, block_idx] = kappa_score
                
                detailed_results.append({
                    'masking_percentage': int(masking_pct),
                    'num_blocks': int(num_blocks),
                    'accuracy': float(test_acc),
                    'kappa': float(kappa_score)
                })
        
        return accuracy_matrix, kappa_matrix, detailed_results

    def step4_generate_plots(self, accuracy_matrix, kappa_matrix, base_val_acc):
        """Generate heatmap plots"""
        print("\n" + "="*60)
        print("STEP 4: Generating Plots")
        print("="*60)

        # Create heatmaps
        self._create_heatmap(
            accuracy_matrix,
            title='Test Accuracy vs Masking Parameters',
            filename='accuracy_heatmap.png',
            metric_name='Accuracy'
        )
        
        self._create_heatmap(
            kappa_matrix,
            title='Cohen\'s Kappa vs Masking Parameters',
            filename='kappa_heatmap.png',
            metric_name='Kappa'
        )
        
        # Create degradation plot (relative to baseline)
        baseline_acc = accuracy_matrix[0, 0]  # 0% masking, 1 block
        degradation_matrix = (baseline_acc - accuracy_matrix) / baseline_acc * 100
        
        self._create_heatmap(
            degradation_matrix,
            title='Performance Degradation (%) vs Masking Parameters',
            filename='degradation_heatmap.png',
            metric_name='Degradation (%)',
            cmap='Reds'
        )
        
        print(f"Plots saved to {self.plots_dir}")

    def _create_heatmap(self, data_matrix, title, filename, metric_name, cmap='viridis'):
        """Create and save a heatmap"""
        plt.figure(figsize=(10, 8))
        
        # Create annotation matrix with 3 decimal places
        annot_matrix = np.around(data_matrix, decimals=3)
        
        sns.heatmap(
            data_matrix,
            xticklabels=self.num_blocks_range,
            yticklabels=[f"{int(p)}%" for p in self.masking_percentages],
            annot=annot_matrix,
            fmt='.3f',
            cmap=cmap,
            cbar_kws={'label': metric_name},
            square=False
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Number of Blocks', fontsize=12)
        plt.ylabel('Masking Percentage', fontsize=12)
        plt.tight_layout()
        
        save_path = self.plots_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")

    def step5_save_results(self, accuracy_matrix, kappa_matrix, detailed_results, base_val_acc):
        """Save all results to files"""
        print("\n" + "="*60)
        print("STEP 5: Saving Results")
        print("="*60)

        # Save matrices as numpy arrays
        np.save(self.results_dir / 'accuracy_matrix.npy', accuracy_matrix)
        np.save(self.results_dir / 'kappa_matrix.npy', kappa_matrix)
        
        # Save detailed results as JSON
        results_data = {
            'experiment_config': self.config,
            'base_validation_accuracy': float(base_val_acc),
            'masking_percentages': [int(p) for p in self.masking_percentages],
            'num_blocks_range': [int(b) for b in self.num_blocks_range],
            'accuracy_matrix': accuracy_matrix.tolist(),
            'kappa_matrix': kappa_matrix.tolist(),
            'detailed_results': detailed_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.results_dir / 'results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save summary statistics
        valid_mask = ~np.isnan(accuracy_matrix[1:])
        if valid_mask.any():
            masked_accuracies = accuracy_matrix[1:][valid_mask]
            
            summary = {
                'baseline_accuracy': float(accuracy_matrix[0, 0]),
                'baseline_kappa': float(kappa_matrix[0, 0]),
                'best_masked_accuracy': float(np.max(masked_accuracies)),
                'worst_masked_accuracy': float(np.min(masked_accuracies)),
                'mean_masked_accuracy': float(np.mean(masked_accuracies)),
                'std_masked_accuracy': float(np.std(masked_accuracies)),
                'max_degradation_pct': float((accuracy_matrix[0, 0] - np.min(masked_accuracies)) / accuracy_matrix[0, 0] * 100)
            }
            
            with open(self.results_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            print("\nSummary Statistics:")
            print(f"  Baseline accuracy: {summary['baseline_accuracy']:.4f}")
            print(f"  Best masked accuracy: {summary['best_masked_accuracy']:.4f}")
            print(f"  Worst masked accuracy: {summary['worst_masked_accuracy']:.4f}")
            print(f"  Mean masked accuracy: {summary['mean_masked_accuracy']:.4f}")
            print(f"  Max degradation: {summary['max_degradation_pct']:.1f}%")
        
        print(f"All results saved to {self.results_dir}")
        
        return results_data

    def run_complete_experiment(self):
        """Run the complete experiment pipeline"""
        print("\n" + "="*70)
        print("STARTING MUSICID MASKING EXPERIMENT WITH DATA AUGMENTATION")
        print("="*70)
        
        experiment_start_time = time.time()
        
        try:
            # Step 1: Train base model
            base_val_acc, base_val_kappa = self.step1_train_base_model()
            
            # Step 2: Load test data
            self.step2_load_test_data()
            
            # Step 3: Evaluate all masking variants
            accuracy_matrix, kappa_matrix, detailed_results = self.step3_evaluate_masking_variants()
            
            # Step 4: Generate plots
            self.step4_generate_plots(accuracy_matrix, kappa_matrix, base_val_acc)
            
            # Step 5: Save results
            results_data = self.step5_save_results(accuracy_matrix, kappa_matrix, detailed_results, base_val_acc)
            
            # Final summary
            experiment_time = time.time() - experiment_start_time
            
            print("\n" + "="*60)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Total experiment time: {experiment_time/3600:.2f} hours")
            print(f"Baseline accuracy (0% masking): {accuracy_matrix[0,0]:.4f}")
            print(f"Results directory: {self.experiment_dir}")
            
            # Print key findings
            valid_mask = ~np.isnan(accuracy_matrix[1:])
            if valid_mask.any():
                masked_accuracies = accuracy_matrix[1:][valid_mask]
                best_masked_acc = np.max(masked_accuracies)
                worst_masked_acc = np.min(masked_accuracies)
                
                print(f"Best masked performance: {best_masked_acc:.4f}")
                print(f"Worst masked performance: {worst_masked_acc:.4f}")
                print(f"Performance range: {best_masked_acc - worst_masked_acc:.4f}")
                
                if not np.isnan(accuracy_matrix[0,0]):
                    degradation_pct = (accuracy_matrix[0,0] - best_masked_acc) / accuracy_matrix[0,0] * 100
                    print(f"Maximum degradation: {degradation_pct:.1f}%")
            
            return results_data
            
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("EXPERIMENT INTERRUPTED BY USER")
            print("="*60)
            return None
            
        except Exception as e:
            print(f"\nEXPERIMENT FAILED: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main execution function"""
    
    # Experiment configuration
    config = {
        # Paths
        'experiment_dir': '/app/data/da_masking_musicid',
        'data_path': '/app/data/musicid',
        
        # Experiment parameters
        'num_classes': 20,  # Number of users to use
        'frame_size': 30,   # Time window size
        
        # Training parameters
        'train_epochs': 100,
        'batch_size': 8,
        'learning_rate': 0.001,
        
        # Experiment metadata
        'description': 'MusicID masking experiment with data augmentation'
    }
    
    print("MusicID Masking Experiment with Data Augmentation")
    print("=" * 60)
    print(f"Number of classes: {config['num_classes']}")
    print(f"Frame size: {config['frame_size']}")
    print(f"Data path: {config['data_path']}")
    print(f"Experiment directory: {config['experiment_dir']}")
    print("=" * 60)
    
    # Initialize and run experiment
    experiment = MusicIDMaskingExperiment(config)
    
    results = experiment.run_complete_experiment()
    
    if results is not None:
        print("\nExperiment completed successfully!")
        print(f"All results saved to: {experiment.experiment_dir}")
        return True
    else:
        print("\nExperiment failed or was interrupted!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
