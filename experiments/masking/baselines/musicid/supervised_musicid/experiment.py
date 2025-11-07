"""
Main script to run the musicid masking experiment for supervised learning
Matches the structure of experiment_new.py but works with CSV data directly
No spectrograms needed - uses raw sensor data from musicid dataset
"""

import sys
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf

# Import your modules
from trainers import cohen_kappa_score
from data_loader import data_load_origin, norma
from backbones import resnetblock_final

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout


class MusicIDMaskingExperiment:
    """Complete musicid masking experiment pipeline for supervised learning"""

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

        # Masking parameter grid - EXACTLY matching experiment_new.py
        self.masking_percentages = np.arange(0, 55, 5)  # 0%, 5%, 10%, ..., 50%
        self.num_blocks_range = [1, 2, 3, 4, 5]

        # Store trained model
        self.base_model = None
        self.base_model_path = None

        print(f"MusicID Masking Experiment initialized")
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Grid size: {len(self.masking_percentages)} × {len(self.num_blocks_range)} = {len(self.masking_percentages) * len(self.num_blocks_range)}")
        print(f"Model: Supervised ResNet classifier")
        print(f"Training hyperparameters:")
        print(f"  - Batch size: {config['batch_size']}")
        print(f"  - Learning rate: {config['learning_rate']}")
        print(f"  - Epochs: {config['train_epochs']}")

    def mask_data(self, x_data, masking_percentage, num_blocks):
        """
        Apply masking to the data
        Matches the masking strategy from experiment_new.py
        
        Args:
            x_data: Input data of shape (n_samples, time_steps, features)
            masking_percentage: Percentage of data to mask (0-100)
            num_blocks: Number of contiguous blocks to mask
            
        Returns:
            Masked data
        """
        if masking_percentage == 0:
            return x_data.copy()
        
        x_masked = x_data.copy()
        n_samples, time_steps, n_features = x_data.shape
        
        # Calculate total points to mask
        total_points = time_steps * n_features
        points_to_mask = int(total_points * masking_percentage / 100)
        
        # For each sample, apply masking
        for i in range(n_samples):
            if num_blocks == 1:
                # Single contiguous block
                # Decide if we mask along time or features
                if np.random.rand() < 0.5:
                    # Mask along time dimension
                    time_points_to_mask = int(time_steps * masking_percentage / 100)
                    if time_points_to_mask > 0:
                        start_idx = np.random.randint(0, max(1, time_steps - time_points_to_mask + 1))
                        x_masked[i, start_idx:start_idx + time_points_to_mask, :] = 0
                else:
                    # Mask along feature dimension
                    feature_points_to_mask = int(n_features * masking_percentage / 100)
                    if feature_points_to_mask > 0:
                        start_idx = np.random.randint(0, max(1, n_features - feature_points_to_mask + 1))
                        x_masked[i, :, start_idx:start_idx + feature_points_to_mask] = 0
            else:
                # Multiple blocks
                points_per_block = points_to_mask // num_blocks
                
                for _ in range(num_blocks):
                    # Decide if this block masks time or features
                    if np.random.rand() < 0.5:
                        # Mask along time
                        block_time_size = max(1, int(np.sqrt(points_per_block / n_features) * n_features))
                        block_time_size = min(block_time_size, time_steps)
                        if block_time_size > 0:
                            start_t = np.random.randint(0, max(1, time_steps - block_time_size + 1))
                            # Random subset of features
                            n_features_to_mask = max(1, points_per_block // block_time_size)
                            n_features_to_mask = min(n_features_to_mask, n_features)
                            feature_indices = np.random.choice(n_features, n_features_to_mask, replace=False)
                            x_masked[i, start_t:start_t + block_time_size, feature_indices] = 0
                    else:
                        # Mask along features
                        block_feature_size = max(1, int(np.sqrt(points_per_block / time_steps) * time_steps))
                        block_feature_size = min(block_feature_size, n_features)
                        if block_feature_size > 0:
                            start_f = np.random.randint(0, max(1, n_features - block_feature_size + 1))
                            # Random subset of time steps
                            n_time_to_mask = max(1, points_per_block // block_feature_size)
                            n_time_to_mask = min(n_time_to_mask, time_steps)
                            time_indices = np.random.choice(time_steps, n_time_to_mask, replace=False)
                            x_masked[i, time_indices, start_f:start_f + block_feature_size] = 0
        
        return x_masked

    def step1_train_base_model(self):
        """Train the base model on unmasked training data"""
        print("\n" + "="*60)
        print("STEP 1: Training Base Model (Unmasked Data)")
        print("="*60)

        frame_size = self.config['frame_size']
        path = self.config['data_path']
        num_classes = self.config['num_classes']
        
        # Select users
        all_users = list(range(1, 21))
        users_to_use = all_users[:num_classes]
        print(f"Using {num_classes} classes (users): {users_to_use}")
        
        # Load training and validation data
        folder_train = ["TrainingSet"]
        folder_val = ["TestingSet"]
        
        print("Loading training data...")
        x_train, y_train, sessions_train = data_load_origin(
            path, users=users_to_use, folders=folder_train, frame_size=frame_size
        )
        print(f"Training samples: {x_train.shape[0]}")
        
        print("Loading validation data...")
        x_val, y_val, sessions_val = data_load_origin(
            path, users=users_to_use, folders=folder_val, frame_size=frame_size
        )
        print(f"Validation samples: {x_val.shape[0]}")
        
        # Normalize
        print("Normalizing data...")
        # Create dummy test set for normalization (will load real test later)
        x_test_dummy = np.zeros_like(x_val[:10])
        x_train, x_val, _ = norma(x_train, x_val, x_test_dummy)
        
        # Build model
        print("Building model...")
        ks = 3
        con = 3
        inputs = Input(shape=(frame_size, x_train.shape[-1]))
        x = Conv1D(filters=16*con, kernel_size=ks, strides=1, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling1D(pool_size=4, strides=4)(x)
        x = Dropout(rate=0.1)(x)
        x = resnetblock_final(x, CR=32*con, KS=ks)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs, outputs)
        
        # Compile
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', restore_best_weights=True, patience=5
        )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config['learning_rate'],
            decay_rate=0.95,
            decay_steps=1000000
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        print(f"Training for {self.config['train_epochs']} epochs...")
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=self.config['train_epochs'],
            callbacks=[callback],
            batch_size=self.config['batch_size'],
            verbose=1
        )
        
        # Save model
        self.base_model = model
        self.base_model_path = self.models_dir / "base_model.h5"
        model.save(self.base_model_path)
        print(f"Base model saved to {self.base_model_path}")
        
        # Get validation accuracy
        val_results = model.evaluate(x_val, y_val, verbose=0)
        base_val_acc = val_results[1]
        print(f"Base model validation accuracy: {base_val_acc:.4f}")
        
        return base_val_acc

    def step2_evaluate_masking_variants(self, base_val_acc):
        """Evaluate all masking variants on test data"""
        print("\n" + "="*60)
        print("STEP 2: Evaluating Masking Variants")
        print("="*60)

        frame_size = self.config['frame_size']
        path = self.config['data_path']
        num_classes = self.config['num_classes']
        
        # Select users
        all_users = list(range(1, 21))
        users_to_use = all_users[:num_classes]
        
        # Load test data
        folder_test = ["TestingSet_secret"]
        print("Loading test data...")
        x_test, y_test, sessions_test = data_load_origin(
            path, users=users_to_use, folders=folder_test, frame_size=frame_size
        )
        print(f"Test samples: {x_test.shape[0]}")
        
        # Normalize test data using the same normalization as training
        # Load training data again to get normalization parameters
        folder_train = ["TrainingSet"]
        folder_val = ["TestingSet"]
        x_train_norm, y_train_norm, _ = data_load_origin(
            path, users=users_to_use, folders=folder_train, frame_size=frame_size
        )
        x_val_norm, y_val_norm, _ = data_load_origin(
            path, users=users_to_use, folders=folder_val, frame_size=frame_size
        )
        x_train_norm, x_val_norm, x_test = norma(x_train_norm, x_val_norm, x_test)
        
        # Initialize results matrices
        n_percentages = len(self.masking_percentages)
        n_blocks = len(self.num_blocks_range)
        
        accuracy_matrix = np.full((n_percentages, n_blocks), np.nan)
        kappa_matrix = np.full((n_percentages, n_blocks), np.nan)
        detailed_results = []
        
        # Load the trained model
        model = tf.keras.models.load_model(self.base_model_path)
        
        total_variants = n_percentages * n_blocks
        variant_count = 0
        
        print(f"\nEvaluating {total_variants} masking variants...")
        print(f"Test set size: {x_test.shape[0]} samples")
        
        # Evaluate each masking variant
        for p_idx, masking_pct in enumerate(self.masking_percentages):
            for b_idx, num_blocks in enumerate(self.num_blocks_range):
                variant_count += 1
                
                print(f"\n[{variant_count}/{total_variants}] Masking: {masking_pct}%, Blocks: {num_blocks}")
                
                # Run multiple evaluations and average
                acc_runs = []
                kappa_runs = []
                
                for run in range(self.config['eval_runs_per_variant']):
                    # Apply masking
                    x_test_masked = self.mask_data(x_test, masking_pct, num_blocks)
                    
                    # Evaluate
                    results = model.evaluate(x_test_masked, y_test, verbose=0)
                    test_acc = results[1]
                    
                    # Calculate kappa
                    y_pred = model.predict(x_test_masked, verbose=0)
                    kappa_score = cohen_kappa_score(y_test, y_pred, num_classes)
                    
                    acc_runs.append(test_acc)
                    kappa_runs.append(kappa_score)
                
                # Average results
                avg_acc = np.mean(acc_runs)
                avg_kappa = np.mean(kappa_runs)
                
                accuracy_matrix[p_idx, b_idx] = avg_acc
                kappa_matrix[p_idx, b_idx] = avg_kappa
                
                print(f"  Accuracy: {avg_acc:.4f} ± {np.std(acc_runs):.4f}")
                print(f"  Kappa: {avg_kappa:.4f} ± {np.std(kappa_runs):.4f}")
                
                # Store detailed results
                detailed_results.append({
                    'masking_percentage': int(masking_pct),
                    'num_blocks': int(num_blocks),
                    'accuracy': float(avg_acc),
                    'kappa': float(avg_kappa),
                    'accuracy_std': float(np.std(acc_runs)),
                    'kappa_std': float(np.std(kappa_runs)),
                    'runs': len(acc_runs)
                })
        
        return accuracy_matrix, kappa_matrix, detailed_results

    def step3_create_visualizations(self, accuracy_matrix, kappa_matrix, base_val_acc):
        """Create all visualization plots - EXACTLY matching experiment_new.py"""
        print("\n" + "="*60)
        print("STEP 3: Creating Visualizations")
        print("="*60)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # 1. Accuracy Heatmap
        print("Creating accuracy heatmap...")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            accuracy_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=self.num_blocks_range,
            yticklabels=self.masking_percentages,
            cbar_kws={'label': 'Accuracy'},
            vmin=0.0,
            vmax=1.0,
            ax=ax
        )
        ax.set_xlabel('Number of Blocks', fontsize=12)
        ax.set_ylabel('Masking Percentage (%)', fontsize=12)
        ax.set_title('Test Accuracy vs Masking Configuration', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Kappa Heatmap
        print("Creating kappa heatmap...")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            kappa_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=self.num_blocks_range,
            yticklabels=self.masking_percentages,
            cbar_kws={'label': 'Cohen\'s Kappa'},
            vmin=0.0,
            vmax=1.0,
            ax=ax
        )
        ax.set_xlabel('Number of Blocks', fontsize=12)
        ax.set_ylabel('Masking Percentage (%)', fontsize=12)
        ax.set_title('Cohen\'s Kappa vs Masking Configuration', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'kappa_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Accuracy vs Masking Percentage (by num_blocks)
        print("Creating accuracy line plots...")
        fig, ax = plt.subplots(figsize=(12, 8))
        for b_idx, num_blocks in enumerate(self.num_blocks_range):
            ax.plot(
                self.masking_percentages,
                accuracy_matrix[:, b_idx],
                marker='o',
                label=f'{num_blocks} block(s)',
                linewidth=2
            )
        ax.axhline(y=base_val_acc, color='r', linestyle='--', label='Baseline (0% masking)', linewidth=2)
        ax.set_xlabel('Masking Percentage (%)', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Test Accuracy vs Masking Percentage', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_vs_masking.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Kappa vs Masking Percentage (by num_blocks)
        print("Creating kappa line plots...")
        fig, ax = plt.subplots(figsize=(12, 8))
        for b_idx, num_blocks in enumerate(self.num_blocks_range):
            ax.plot(
                self.masking_percentages,
                kappa_matrix[:, b_idx],
                marker='o',
                label=f'{num_blocks} block(s)',
                linewidth=2
            )
        # Baseline kappa (from unmasked evaluation)
        baseline_kappa = kappa_matrix[0, 0]  # 0% masking, 1 block
        ax.axhline(y=baseline_kappa, color='r', linestyle='--', label='Baseline (0% masking)', linewidth=2)
        ax.set_xlabel('Masking Percentage (%)', fontsize=12)
        ax.set_ylabel('Cohen\'s Kappa', fontsize=12)
        ax.set_title('Cohen\'s Kappa vs Masking Percentage', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'kappa_vs_masking.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Performance Degradation Plot
        print("Creating performance degradation plot...")
        fig, ax = plt.subplots(figsize=(12, 8))
        for b_idx, num_blocks in enumerate(self.num_blocks_range):
            degradation = (base_val_acc - accuracy_matrix[:, b_idx]) / base_val_acc * 100
            ax.plot(
                self.masking_percentages,
                degradation,
                marker='o',
                label=f'{num_blocks} block(s)',
                linewidth=2
            )
        ax.set_xlabel('Masking Percentage (%)', fontsize=12)
        ax.set_ylabel('Performance Degradation (%)', fontsize=12)
        ax.set_title('Performance Degradation vs Masking Percentage', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_degradation.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"All plots saved to {self.plots_dir}")

    def step4_save_results(self, accuracy_matrix, kappa_matrix, detailed_results, base_val_acc):
        """Save all results to files"""
        print("\n" + "="*60)
        print("STEP 4: Saving Results")
        print("="*60)

        # Save matrices
        results_file = self.results_dir / 'results_matrices.npz'
        np.savez(
            results_file,
            accuracy_matrix=accuracy_matrix,
            kappa_matrix=kappa_matrix,
            masking_percentages=self.masking_percentages,
            num_blocks_range=self.num_blocks_range,
            base_val_acc=base_val_acc
        )
        print(f"Results matrices saved to {results_file}")

        # Save detailed results as JSON
        results_json = {
            'experiment_info': {
                'date': datetime.now().isoformat(),
                'num_classes': self.config['num_classes'],
                'frame_size': self.config['frame_size'],
                'train_epochs': self.config['train_epochs'],
                'batch_size': self.config['batch_size'],
                'learning_rate': self.config['learning_rate'],
                'base_val_acc': float(base_val_acc)
            },
            'masking_grid': {
                'masking_percentages': self.masking_percentages.tolist(),
                'num_blocks_range': self.num_blocks_range
            },
            'results': detailed_results,
            'summary': {
                'best_masked_accuracy': float(np.nanmax(accuracy_matrix[1:])),
                'worst_masked_accuracy': float(np.nanmin(accuracy_matrix[1:])),
                'best_masked_kappa': float(np.nanmax(kappa_matrix[1:])),
                'worst_masked_kappa': float(np.nanmin(kappa_matrix[1:]))
            }
        }

        json_file = self.results_dir / 'detailed_results.json'
        with open(json_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"Detailed results saved to {json_file}")

        # Save summary text file
        summary_file = self.results_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MUSICID MASKING EXPERIMENT SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment directory: {self.experiment_dir}\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  - Number of classes: {self.config['num_classes']}\n")
            f.write(f"  - Frame size: {self.config['frame_size']}\n")
            f.write(f"  - Training epochs: {self.config['train_epochs']}\n")
            f.write(f"  - Batch size: {self.config['batch_size']}\n")
            f.write(f"  - Learning rate: {self.config['learning_rate']}\n\n")
            
            f.write("Results:\n")
            f.write(f"  - Base validation accuracy: {base_val_acc:.4f}\n")
            f.write(f"  - Best masked accuracy: {np.nanmax(accuracy_matrix[1:]):.4f}\n")
            f.write(f"  - Worst masked accuracy: {np.nanmin(accuracy_matrix[1:]):.4f}\n")
            f.write(f"  - Best masked kappa: {np.nanmax(kappa_matrix[1:]):.4f}\n")
            f.write(f"  - Worst masked kappa: {np.nanmin(kappa_matrix[1:]):.4f}\n")
            
        print(f"Summary saved to {summary_file}")

        return results_json

    def run_complete_experiment(self):
        """Run the complete experiment pipeline"""
        print("\n" + "="*60)
        print("STARTING COMPLETE MASKING EXPERIMENT")
        print("="*60)

        experiment_start_time = time.time()

        try:
            # Step 1: Train base model
            base_val_acc = self.step1_train_base_model()

            # Step 2: Evaluate masking variants
            accuracy_matrix, kappa_matrix, detailed_results = self.step2_evaluate_masking_variants(base_val_acc)

            # Step 3: Create visualizations
            self.step3_create_visualizations(accuracy_matrix, kappa_matrix, base_val_acc)

            # Step 4: Save results
            results_data = self.step4_save_results(accuracy_matrix, kappa_matrix, detailed_results, base_val_acc)

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
        'experiment_dir': '/app/data/supervised_masking_musicid',
        'data_path': '/app/data/musicid',

        # Experiment parameters
        'num_classes': 20,  # Number of users to use
        'frame_size': 30,

        # Training parameters
        'train_epochs': 100,
        'batch_size': 8,
        'learning_rate': 0.001,
        'eval_runs_per_variant': 3,  # Number of evaluation runs per masking variant

        # Experiment metadata
        'description': 'MusicID masking experiment with supervised learning'
    }

    print("MusicID Masking Experiment - Supervised Learning")
    print("=" * 60)
    print(f"Number of classes: {config['num_classes']}")
    print(f"Frame size: {config['frame_size']}")
    print(f"Training epochs: {config['train_epochs']}")
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
