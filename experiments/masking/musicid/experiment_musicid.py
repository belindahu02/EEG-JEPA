"""
Main script to run the MusicID masking experiment
CORRECTED V2: Works with flat directory structure and augmented files
"""

import sys
import numpy as np
import torch
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import fixed MusicID modules
from utils_musicid import (
    MusicIDMaskingProcessor, 
    create_masked_dataset_for_experiment, 
    run_embedding_extraction, 
    run_embedding_grouping,
    parse_musicid_filename
)
from data_loader_musicid import create_session_based_dataloaders
from data_loader import MaskingExperimentTrainer
from backbones import SpectrogramResNet, LightweightSpectrogramResNet
from checkpoint import MaskingExperimentCheckpoint


class MusicIDMaskingExperiment:
    """Complete MusicID masking experiment pipeline with flat CSV structure"""

    def __init__(self, config):
        self.config = config
        self.experiment_dir = Path(config['experiment_dir'])
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(config['experiment_dir']) / "data"

        # Create subdirectories
        self.models_dir = self.experiment_dir / "models"
        self.results_dir = self.experiment_dir / "results"
        self.plots_dir = self.experiment_dir / "plots"

        for d in [self.data_dir, self.models_dir, self.results_dir, self.plots_dir]:
            d.mkdir(exist_ok=True)

        # Initialize checkpoint manager
        checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_mgr = MaskingExperimentCheckpoint(checkpoint_dir)

        # Masking parameter grid
        self.masking_percentages = np.arange(0, 55, 5)  # 0%, 5%, 10%, ..., 50%
        self.num_blocks_range = [1, 2, 3, 4, 5]

        # Store trained model
        self.base_model_path = None

        print(f"MusicID Masking Experiment initialized")
        print(f"Raw CSV directory: {config['raw_csv_dir']}")
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Users: {config['user_ids']}")
        print(f"Session types: {config.get('session_types', 'all')}")
        print(f"Use augmented: {config.get('use_augmented', True)}")
        print(f"Grid size: {len(self.masking_percentages)} × {len(self.num_blocks_range)} = {len(self.masking_percentages) * len(self.num_blocks_range)}")

        # Check for existing checkpoint
        if self.checkpoint_mgr.has_checkpoint():
            print("⚠️  EXISTING CHECKPOINT FOUND")
            print("The experiment can be resumed from the last saved state")

    def step1_train_base_model(self):
        """Train the base model on unmasked training data"""
        print("\n" + "="*60)
        print("STEP 1: Training Base Model (Unmasked Data)")
        print("="*60)

        # Generate unmasked training data
        print("Converting CSV files to spectrograms (unmasked, training/validation)...")
        
        unmasked_dir = self.data_dir / "unmasked_train_val"
        if unmasked_dir.exists() and any(unmasked_dir.glob("**/*.npy")):
           print(f"Dataset already exists at {unmasked_dir}, skipping creation.")
        else:
            create_masked_dataset_for_experiment(
                raw_csv_dir=self.config['raw_csv_dir'],
                output_dir=unmasked_dir,
                user_ids=self.config['user_ids'],
                session_nums=self.config['train_sessions'] + self.config['val_sessions'],
                masking_percentage=0,  # No masking
                num_blocks=1,
                use_augmented=self.config.get('use_augmented', True),
                session_types=self.config.get('session_types', None)
            )

        # Run preprocessing pipeline
        print("Running preprocessing pipeline...")
        embeddings_dir = self.data_dir / "embeddings_train_val"
        grouped_dir = self.data_dir / "grouped_train_val"

        # Check if preprocessing has already been done
        embeddings_exist = embeddings_dir.exists() and any(embeddings_dir.glob("**/*.npy"))
        grouped_exist = grouped_dir.exists() and any(grouped_dir.glob("**/*.npy"))

        # Extract embeddings using pre-trained JEPA encoder
        if not embeddings_exist:
            print("Extracting embeddings...")
            run_embedding_extraction(
                csv_path=unmasked_dir / "files_audioset.csv",
                data_dir=unmasked_dir,
                embeddings_dir=embeddings_dir,
                model_checkpoint_path=self.config['model_checkpoint_path'],
                config_path=self.config['model_config_path'],
                batch_size=self.config['batch_size'],
                num_workers=2,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            print(f"Embeddings already exist in {embeddings_dir}, skipping extraction...")

        # Group embeddings
        if not grouped_exist:
            print("Grouping embeddings...")
            run_embedding_grouping(embeddings_dir, grouped_dir)
        else:
            print(f"Grouped embeddings already exist in {grouped_dir}, skipping grouping...")

        # Create data loaders
        print("Creating data loaders for training...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        train_loader, val_loader, test_loader, num_classes = create_session_based_dataloaders(
            data_path=grouped_dir,
            user_ids=self.config['user_ids'],
            normalization='none',
            batch_size=self.config['batch_size'],
            augment_train=False,
            cache_size=100,
            test_sessions_per_user=len(self.config['test_sessions']),
            val_sessions_per_user=len(self.config['val_sessions'])
        )

        # Get input shape from sample batch
        sample_batch = next(iter(train_loader))
        input_shape = sample_batch[0].shape[1:]  # Remove batch dim
        print(f"Input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")

        # Initialize model
        model_type = self.config.get('model_type', 'lightweight')
        print(f"Initializing {model_type} model...")
        
        if model_type == 'full':
            model = SpectrogramResNet(
                input_channels=input_shape[0],
                num_classes=num_classes,
                channels=[64, 128, 256, 512],
                dropout_rate=self.config.get('dropout_rate', 0.0),
                classifier_dropout=self.config.get('classifier_dropout', 0.0)
            )
        else:  # lightweight
            model = LightweightSpectrogramResNet(
                input_channels=input_shape[0],
                num_classes=num_classes,
                channels=[32, 64, 128]
            )

        # Initialize trainer
        trainer = MaskingExperimentTrainer(
            model, 
            device=device,
            use_cosine_classifier=self.config.get('use_cosine_classifier', True),
            cosine_scale=self.config.get('cosine_scale', 40.0),
            label_smoothing=self.config.get('label_smoothing', 0.1),
            warmup_epochs=self.config.get('warmup_epochs', 5)
        )

        print(f"Training model for {self.config['train_epochs']} epochs...")
        
        best_val_acc = trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config['train_epochs'],
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        # Save trained model
        self.base_model_path = self.models_dir / "base_model.pth"
        trainer.save_model(self.base_model_path)

        print(f"Base model training completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Model saved to: {self.base_model_path}")

        return best_val_acc

    def step2_generate_test_data_variants(self):
        """Generate test data with different masking configurations"""
        print("\n" + "="*60)
        print("STEP 2: Generating Test Data Variants from CSV")
        print("="*60)

        print(f"Checking/generating {len(self.masking_percentages) * len(self.num_blocks_range)} test data variants...")

        test_data_dirs = {}

        for mask_pct in self.masking_percentages:
            for num_blocks in self.num_blocks_range:
                # Create unique identifier
                variant_id = f"test_mask_{mask_pct}pct_{num_blocks}blocks"
                variant_dir = self.data_dir / variant_id
                embeddings_dir = self.data_dir / f"embeddings_{variant_id}"
                grouped_dir = self.data_dir / f"grouped_{variant_id}"

                # Check if this variant already exists (all 3 stages)
                variant_exists = variant_dir.exists() and any(variant_dir.glob("**/*.npy"))
                embeddings_exist = embeddings_dir.exists() and any(embeddings_dir.glob("**/*.npy"))
                grouped_exist = grouped_dir.exists() and any(grouped_dir.glob("**/*.npy"))

                if variant_exists and embeddings_exist and grouped_exist:
                    print(f"✓ Variant {mask_pct}%/{num_blocks}blocks already exists, skipping")
                    test_data_dirs[(mask_pct, num_blocks)] = grouped_dir
                    continue

                print(f"Creating test data: {mask_pct}% masking, {num_blocks} blocks")

                # Generate masked test data from CSV (only if needed)
                if not variant_exists:
                    create_masked_dataset_for_experiment(
                        raw_csv_dir=self.config['raw_csv_dir'],
                        output_dir=variant_dir,
                        user_ids=self.config['user_ids'],
                        session_nums=self.config['test_sessions'],
                        masking_percentage=mask_pct,
                        num_blocks=num_blocks,
                        use_augmented=self.config.get('use_augmented', True),
                        session_types=self.config.get('session_types', None)
                    )
                else:
                    print(f"  Spectrograms exist, skipping CSV conversion")

                # Extract embeddings (only if needed)
                if not embeddings_exist:
                    print(f"  Extracting embeddings...")
                    run_embedding_extraction(
                        csv_path=variant_dir / "files_audioset.csv",
                        data_dir=variant_dir,
                        embeddings_dir=embeddings_dir,
                        model_checkpoint_path=self.config['model_checkpoint_path'],
                        config_path=self.config['model_config_path'],
                        batch_size=self.config['batch_size'],
                        num_workers=2,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                else:
                    print(f"  Embeddings exist, skipping extraction")

                # Group embeddings (only if needed)
                if not grouped_exist:
                    print(f"  Grouping embeddings...")
                    run_embedding_grouping(embeddings_dir, grouped_dir)
                else:
                    print(f"  Grouped data exists, skipping grouping")

                test_data_dirs[(mask_pct, num_blocks)] = grouped_dir

        print(f"All {len(test_data_dirs)} test data variants ready")
        return test_data_dirs

    def step3_evaluate_masking_effects(self, test_data_dirs, resume_state=None):
        """Evaluate the trained model on all masking variants"""
        print("\n" + "="*60)
        print("STEP 3: Evaluating Masking Effects")
        print("="*60)

        if self.base_model_path is None:
            raise ValueError("Base model not trained. Run step1_train_base_model first.")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load trained model
        num_classes = len(self.config['user_ids'])
        model_type = self.config.get('model_type', 'lightweight')

        print(f"Loading {model_type} model for evaluation...")
        
        if model_type == 'full':
            model = SpectrogramResNet(
                input_channels=1,
                num_classes=num_classes,
                channels=[64, 128, 256, 512],
                dropout_rate=self.config.get('dropout_rate', 0.0),
                classifier_dropout=self.config.get('classifier_dropout', 0.0)
            )
        else:
            model = LightweightSpectrogramResNet(
                input_channels=1,
                num_classes=num_classes,
                channels=[32, 64, 128]
            )

        trainer = MaskingExperimentTrainer(
            model, 
            device=device,
            use_cosine_classifier=self.config.get('use_cosine_classifier', True),
            cosine_scale=self.config.get('cosine_scale', 40.0),
            label_smoothing=self.config.get('label_smoothing', 0.1),
            warmup_epochs=self.config.get('warmup_epochs', 5)
        )
        trainer.load_model(self.base_model_path)

        print(f"Loaded trained model from: {self.base_model_path}")

        # Initialize or load existing results
        if resume_state:
            accuracy_matrix, kappa_matrix, detailed_results = resume_state
            completed_combinations = {(r['masking_percentage'], r['num_blocks']) for r in detailed_results}
            print(f"Resuming evaluation: {len(completed_combinations)} combinations already completed")
        else:
            accuracy_matrix = np.zeros((len(self.masking_percentages), len(self.num_blocks_range)))
            kappa_matrix = np.zeros((len(self.masking_percentages), len(self.num_blocks_range)))
            detailed_results = []
            completed_combinations = set()

        total_evaluations = len(self.masking_percentages) * len(self.num_blocks_range)
        current_eval = len(completed_combinations)

        print(f"Total evaluations: {total_evaluations}")
        print(f"Remaining: {total_evaluations - current_eval}")

        for i, mask_pct in enumerate(self.masking_percentages):
            for j, num_blocks in enumerate(self.num_blocks_range):

                # Skip if already completed
                if (mask_pct, num_blocks) in completed_combinations:
                    print(f"Skipping {mask_pct}% / {num_blocks} blocks (already completed)")
                    continue

                current_eval += 1

                print(f"\nEvaluation {current_eval}/{total_evaluations}: {mask_pct}% masking, {num_blocks} blocks")

                try:
                    # Get test data directory
                    grouped_dir = test_data_dirs[(mask_pct, num_blocks)]

                    # Create test data loader
                    test_dataset = MusicIDTestDataset(grouped_dir, self.config['user_ids'])
                    test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=self.config['batch_size'],
                        shuffle=False
                    )

                    if len(test_loader) == 0:
                        print(f"  No test data found for this variant")
                        accuracy_matrix[i, j] = np.nan
                        kappa_matrix[i, j] = np.nan
                        continue

                    # Run evaluations
                    accuracies = []
                    kappa_scores = []

                    for run in range(self.config['eval_runs_per_variant']):
                        acc, kappa = trainer.evaluate_with_kappa(test_loader)
                        accuracies.append(acc)
                        kappa_scores.append(kappa)

                    avg_acc = np.mean(accuracies)
                    avg_kappa = np.mean(kappa_scores)
                    std_acc = np.std(accuracies)
                    std_kappa = np.std(kappa_scores)

                    accuracy_matrix[i, j] = avg_acc
                    kappa_matrix[i, j] = avg_kappa

                    result_record = {
                        'masking_percentage': mask_pct,
                        'num_blocks': num_blocks,
                        'avg_accuracy': avg_acc,
                        'std_accuracy': std_acc,
                        'avg_kappa': avg_kappa,
                        'std_kappa': std_kappa,
                        'accuracies': accuracies,
                        'kappa_scores': kappa_scores,
                        'num_runs': len(accuracies)
                    }
                    detailed_results.append(result_record)
                    completed_combinations.add((mask_pct, num_blocks))

                    print(f"  Results: Acc={avg_acc:.4f}±{std_acc:.4f}, Kappa={avg_kappa:.4f}±{std_kappa:.4f}")

                    # Save checkpoint periodically
                    if current_eval % 5 == 0:
                        self.checkpoint_mgr.save_checkpoint(
                            stage='evaluating',
                            stage_data={
                                'base_model_path': str(self.base_model_path),
                                'completed_variants': list(completed_combinations),
                                'current_progress': f"{current_eval}/{total_evaluations}"
                            },
                            accuracy_matrix=accuracy_matrix,
                            kappa_matrix=kappa_matrix,
                            detailed_results=detailed_results,
                            metadata={'step': 3, 'progress': f"{current_eval}/{total_evaluations}"}
                        )
                        print(f"  Checkpoint saved ({current_eval}/{total_evaluations} completed)")

                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    accuracy_matrix[i, j] = np.nan
                    kappa_matrix[i, j] = np.nan

        return accuracy_matrix, kappa_matrix, detailed_results

    def step4_create_visualizations(self, accuracy_matrix, kappa_matrix, detailed_results):
        """Create heatmaps and other visualizations"""
        print("\n" + "="*60)
        print("STEP 4: Creating Visualizations")
        print("="*60)

        # Accuracy heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            accuracy_matrix,
            xticklabels=self.num_blocks_range,
            yticklabels=[f"{p}%" for p in self.masking_percentages],
            annot=True,
            fmt='.3f',
            cmap='viridis',
            cbar_kws={'label': 'Test Accuracy'},
            vmin=0,
            vmax=1
        )
        plt.title('MusicID Classification Accuracy vs Masking Configuration',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Number of Masking Blocks', fontsize=12)
        plt.ylabel('Total Masking Percentage', fontsize=12)
        plt.tight_layout()

        acc_heatmap_path = self.plots_dir / "accuracy_heatmap.png"
        plt.savefig(acc_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Kappa heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            kappa_matrix,
            xticklabels=self.num_blocks_range,
            yticklabels=[f"{p}%" for p in self.masking_percentages],
            annot=True,
            fmt='.3f',
            cmap='viridis',
            cbar_kws={'label': 'Cohen\'s Kappa'}
        )
        plt.title('MusicID Classification Kappa vs Masking Configuration',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Number of Masking Blocks', fontsize=12)
        plt.ylabel('Total Masking Percentage', fontsize=12)
        plt.tight_layout()

        kappa_heatmap_path = self.plots_dir / "kappa_heatmap.png"
        plt.savefig(kappa_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualizations saved to: {self.plots_dir}")

    def step5_save_results(self, accuracy_matrix, kappa_matrix, detailed_results, base_val_acc):
        """Save all experiment results"""
        print("\n" + "="*60)
        print("STEP 5: Saving Results")
        print("="*60)

        results_data = {
            'experiment_config': self.config,
            'experiment_timestamp': datetime.now().isoformat(),
            'masking_percentages': self.masking_percentages.tolist(),
            'num_blocks_range': self.num_blocks_range,
            'base_model_validation_accuracy': base_val_acc,
            'accuracy_matrix': accuracy_matrix.tolist(),
            'kappa_matrix': kappa_matrix.tolist(),
            'detailed_results': detailed_results
        }

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(x) for x in obj]
            return obj

        # Save as JSON
        json_path = self.results_dir / "experiment_results.json"
        with open(json_path, 'w') as f:
            json.dump(convert_numpy(results_data), f, indent=2)

        # Save matrices as NumPy
        npz_path = self.results_dir / "results_matrices.npz"
        np.savez(
            npz_path,
            accuracy_matrix=accuracy_matrix,
            kappa_matrix=kappa_matrix,
            masking_percentages=self.masking_percentages,
            num_blocks_range=self.num_blocks_range
        )

        print(f"Results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  NumPy: {npz_path}")

        return results_data

    def run_complete_experiment(self, resume=True):
        """Run the complete masking experiment pipeline"""
        print("Starting MusicID Masking Experiment")
        print(f"Raw CSV directory: {self.config['raw_csv_dir']}")

        experiment_start_time = time.time()

        # Check for checkpoint
        checkpoint_state = None
        if resume and self.checkpoint_mgr.has_checkpoint():
            print("\n" + "="*60)
            print("RESUMING FROM CHECKPOINT")
            print("="*60)
            checkpoint_state = self.checkpoint_mgr.load_checkpoint()

        try:
            # Step 1: Train base model
            if checkpoint_state and checkpoint_state['checkpoint_data']['stage'] != 'init':
                print("\n✓ Step 1: Base model already trained, loading...")
                stage_data = checkpoint_state['checkpoint_data']['stage_data']
                self.base_model_path = Path(stage_data.get('base_model_path'))
                base_val_acc = stage_data.get('base_val_acc', 0.0)
            else:
                base_val_acc = self.step1_train_base_model()
                self.checkpoint_mgr.save_checkpoint(
                    stage='train_complete',
                    stage_data={'base_model_path': str(self.base_model_path), 'base_val_acc': float(base_val_acc)},
                    metadata={'step': 1, 'description': 'Base model training completed'}
                )

            # Step 2: Generate test data variants
            if checkpoint_state and checkpoint_state['checkpoint_data']['stage'] in ['variants_complete', 'evaluate_complete', 'complete']:
                print("\n✓ Step 2: Test data variants already generated")
                test_data_dirs = {}
                for k, v in checkpoint_state['checkpoint_data']['stage_data'].get('test_data_dirs', {}).items():
                    mask_pct, num_blocks = eval(k)
                    test_data_dirs[(mask_pct, num_blocks)] = Path(v)
            else:
                test_data_dirs = self.step2_generate_test_data_variants()
                test_data_dirs_serializable = {str(k): str(v) for k, v in test_data_dirs.items()}
                self.checkpoint_mgr.save_checkpoint(
                    stage='variants_complete',
                    stage_data={'base_model_path': str(self.base_model_path), 'test_data_dirs': test_data_dirs_serializable},
                    metadata={'step': 2}
                )

            # Step 3: Evaluate
            if checkpoint_state and checkpoint_state['checkpoint_data']['stage'] in ['evaluate_complete', 'complete']:
                print("\n✓ Step 3: Evaluation already complete")
                accuracy_matrix = checkpoint_state['accuracy_matrix']
                kappa_matrix = checkpoint_state['kappa_matrix']
                detailed_results = checkpoint_state.get('detailed_results', [])
            else:
                resume_state = None
                if checkpoint_state and 'accuracy_matrix' in checkpoint_state:
                    accuracy_matrix = checkpoint_state['accuracy_matrix']
                    kappa_matrix = checkpoint_state['kappa_matrix']
                    detailed_results = checkpoint_state.get('detailed_results', [])
                    resume_state = (accuracy_matrix, kappa_matrix, detailed_results)

                accuracy_matrix, kappa_matrix, detailed_results = self.step3_evaluate_masking_effects(
                    test_data_dirs, resume_state
                )
                self.checkpoint_mgr.save_checkpoint(
                    stage='evaluate_complete',
                    stage_data={'base_model_path': str(self.base_model_path)},
                    accuracy_matrix=accuracy_matrix,
                    kappa_matrix=kappa_matrix,
                    detailed_results=detailed_results,
                    metadata={'step': 3}
                )

            # Step 4: Visualizations
            self.step4_create_visualizations(accuracy_matrix, kappa_matrix, detailed_results)

            # Step 5: Save results
            results_data = self.step5_save_results(accuracy_matrix, kappa_matrix, detailed_results, base_val_acc)

            # Mark complete
            self.checkpoint_mgr.save_checkpoint(
                stage='complete',
                stage_data={'experiment_complete': True},
                accuracy_matrix=accuracy_matrix,
                kappa_matrix=kappa_matrix,
                detailed_results=detailed_results,
                metadata={'step': 5}
            )

            experiment_time = time.time() - experiment_start_time

            print("\n" + "="*60)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Total time: {experiment_time/3600:.2f} hours")
            print(f"Results directory: {self.experiment_dir}")

            return results_data

        except KeyboardInterrupt:
            print("\nEXPERIMENT INTERRUPTED - Progress saved")
            return None
        except Exception as e:
            print(f"\nEXPERIMENT FAILED: {e}")
            import traceback
            traceback.print_exc()
            return None


class MusicIDTestDataset(torch.utils.data.Dataset):
    """Dataset that loads test session data for MusicID"""

    def __init__(self, data_dir, user_ids):
        self.data_dir = Path(data_dir)
        self.user_ids = user_ids
        self.file_paths = []
        self.labels = []

        for user_idx, user_id in enumerate(user_ids):
            user_folder = f"user{user_id}"
            user_path = self.data_dir / user_folder

            if not user_path.exists():
                continue

            # Find all npy files in user directory
            for npy_file in user_path.rglob("*_stacked.npy"):
                self.file_paths.append(npy_file)
                self.labels.append(user_idx)

        print(f"MusicIDTestDataset: {len(self.file_paths)} test files")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def analyze_available_data(csv_dir):
    """Analyze what data is available in the directory"""
    csv_dir = Path(csv_dir)
    csv_files = list(csv_dir.glob("*.csv"))
    
    print(f"\nAnalyzing {len(csv_files)} CSV files in {csv_dir}...")
    
    from collections import defaultdict
    from utils_musicid import parse_musicid_filename
    
    users = set()
    session_types = set()
    sessions_per_user = defaultdict(set)
    types_per_user = defaultdict(set)
    augmented_count = 0
    original_count = 0
    
    for csv_file in csv_files:
        info = parse_musicid_filename(csv_file.name)
        if info:
            users.add(info['user_id'])
            session_types.add(info['session_type'])
            sessions_per_user[info['user_id']].add(info['session_num'])
            types_per_user[info['user_id']].add(info['session_type'])
            if info['is_augmented']:
                augmented_count += 1
            else:
                original_count += 1
    
    print(f"\nAvailable users: {sorted(users)}")
    print(f"Session types: {sorted(session_types)}")
    print(f"Original files: {original_count}")
    print(f"Augmented files: {augmented_count}")
    print(f"\nSessions per user:")
    for user_id in sorted(users):
        print(f"  User {user_id}: Sessions {sorted(sessions_per_user[user_id])}, Types {sorted(types_per_user[user_id])}")


def main():
    """Main execution function"""

    # First, analyze available data
    csv_dir = '/app/data/musicid_augmented'
    if Path(csv_dir).exists():
        analyze_available_data(csv_dir)
    
    config = {
        # CORRECTED PATHS - flat directory
        'raw_csv_dir': csv_dir,
        'experiment_dir': '/app/data/experiments/masking_musicid_unstructured',

        # Pre-trained model paths
        'model_checkpoint_path': '/app/data/jepa_logs_musicid_unstructured/xps/97d170e1/checkpoints/last.ckpt',
        'model_config_path': '/app/configs',

        # Experiment parameters - UPDATE THESE based on analysis above
        'user_ids': [1, 2],  # Users available in your data
        'train_sessions': [1, 2, 3],  # Sessions for training
        'val_sessions': [4],          # Sessions for validation
        'test_sessions': [5],         # Sessions for testing
        'session_types': ['fav', 'same'],  # Which session types to use, or None for all
        'use_augmented': True,  # Whether to include augmented files

        # Training parameters
        'train_epochs': 100,
        'batch_size': 8,
        'learning_rate': 0.001,
        'eval_runs_per_variant': 1,
        'weight_decay': 1e-4,

        # Advanced training
        'use_cosine_classifier': True,
        'cosine_scale': 40.0,
        'label_smoothing': 0.1,
        'warmup_epochs': 5,
        'dropout_rate': 0.2,
        'classifier_dropout': 0.0,

        # Model
        'model_type': 'lightweight',

        'description': 'MusicID masking experiment with flat CSV structure'
    }

    print("\nMusicID Masking Experiment (Flat Directory Structure)")
    print("=" * 60)
    print(f"Raw CSV directory: {config['raw_csv_dir']}")
    print(f"Users: {config['user_ids']}")
    print(f"Session types: {config['session_types']}")
    print(f"Experiment directory: {config['experiment_dir']}")
    print("=" * 60)

    experiment = MusicIDMaskingExperiment(config)
    results = experiment.run_complete_experiment()

    return results is not None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
