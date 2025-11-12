"""
EEG Masking Experiment for Supervised Learning (Direct EEG Data)
Tests how masking affects model performance on biometric identification

Operates directly with raw EEG data (.edf files) without spectrograms
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from pathlib import Path
import gc
import torch 

from trainers_masking import train_base_model, evaluate_on_masked_test



class SupervisedMaskingExperiment:
    """
    Masking experiment for supervised EEG biometric identification.
    Tests different masking configurations on test data.
    """

    def __init__(self, config):
        self.config = config
        self.experiment_dir = Path(config['experiment_dir'])
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.results_dir = self.experiment_dir / "results"
        self.plots_dir = self.experiment_dir / "plots"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"

        for d in [self.results_dir, self.plots_dir, self.checkpoints_dir]:
            d.mkdir(exist_ok=True)

        # Masking parameter grid
        self.masking_percentages = np.arange(0, 55, 5)  # 0%, 5%, 10%, ..., 50%
        self.num_blocks_range = [1, 2, 3, 4, 5]

        # Checkpoint files
        self.checkpoint_file = self.checkpoints_dir / "checkpoint.json"
        self.results_file = self.checkpoints_dir / "partial_results.npz"
        self.model_path = self.checkpoints_dir / "base_model.h5"

        print(f"\n{'='*70}")
        print("EEG Supervised Masking Experiment")
        print(f"{'='*70}")
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Number of users: {len(config['user_ids'])}")
        print(f"Grid size: {len(self.masking_percentages)} × {len(self.num_blocks_range)} = {len(self.masking_percentages) * len(self.num_blocks_range)}")
        print(f"Eval runs per variant: {config['eval_runs_per_variant']}")

    def load_checkpoint(self):
        """Load checkpoint if it exists"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"\n{'='*70}")
            print("RESUMING FROM CHECKPOINT")
            print(f"Stage: {checkpoint['stage']}")
#            print(f"Completed variants: {checkpoint['completed_variants']}/{checkpoint['total_variants']}")
            print(f"{'='*70}\n")
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

        # Save partial results if provided
        if accuracy_matrix is not None and kappa_matrix is not None:
            np.savez(
                self.results_file,
                accuracy_matrix=accuracy_matrix,
                kappa_matrix=kappa_matrix,
                detailed_results=detailed_results if detailed_results is not None else []
            )

    def step1_train_base_model(self):
        """Train base model on unmasked data"""
        print(f"\n{'='*70}")
        print("STEP 1: Training Base Model on Unmasked Data")
        print(f"{'='*70}")

        # Check if model already exists
        if self.model_path.exists():
            print(f"Base model already exists at {self.model_path}")
            print("Loading existing model...")
            import tensorflow as tf
            model = tf.keras.models.load_model(self.model_path)
            
            # Need to get normalization stats
            from data_loader_masking import calculate_normalization_stats
            normalization_stats = calculate_normalization_stats(
                path=self.config['data_path'],
                users=self.config['user_ids'],
                frame_size=40,
                max_samples=10000
            )
            
            # Get validation accuracy from checkpoint
            checkpoint = self.load_checkpoint()
            if checkpoint and 'stage_data' in checkpoint:
                val_acc = checkpoint['stage_data'].get('val_acc', 0.0)
            else:
                val_acc = 0.0
                
            return model, normalization_stats, val_acc

        # Train new model
        model, normalization_stats, val_acc = train_base_model(
            num_users=len(self.config['user_ids']),
            cache_size_gb=self.config.get('cache_size_gb', None)
        )

        # Save model
        model.save(self.model_path)
        print(f"Base model saved to {self.model_path}")

        # Save checkpoint
        self.save_checkpoint(
            stage='base_model_trained',
            stage_data={
                'val_acc': float(val_acc),
                'model_path': str(self.model_path)
            }
        )

        return model, normalization_stats, val_acc

    def step2_evaluate_masked_variants(self, model, normalization_stats):
        """Evaluate model on all masked test variants"""
        print(f"\n{'='*70}")
        print("STEP 2: Evaluating on Masked Test Data")
        print(f"{'='*70}")

        # Initialize result matrices
        n_percentages = len(self.masking_percentages)
        n_blocks = len(self.num_blocks_range)
        n_runs = self.config['eval_runs_per_variant']

        accuracy_matrix = np.zeros((n_percentages, n_blocks, n_runs))
        kappa_matrix = np.zeros((n_percentages, n_blocks, n_runs))
        detailed_results = []

        # Load checkpoint to resume if needed
        checkpoint = self.load_checkpoint()
        completed_variants = set()
        
        if checkpoint and checkpoint['stage'] == 'evaluating':
            if self.results_file.exists():
                data = np.load(self.results_file, allow_pickle=True)
                accuracy_matrix = data['accuracy_matrix']
                kappa_matrix = data['kappa_matrix']
                detailed_results = data['detailed_results'].tolist()
                
            # Rebuild completed variants set
            for result in detailed_results:
                variant_key = (result['masking_percentage'], result['num_blocks'], result['run'])
                completed_variants.add(variant_key)

        total_variants = n_percentages * n_blocks * n_runs
        completed_count = len(completed_variants)

        print(f"Total variants to evaluate: {total_variants}")
        print(f"Already completed: {completed_count}")
        print(f"Remaining: {total_variants - completed_count}\n")

        # Evaluate each configuration
        variant_num = 0
        for i, mask_pct in enumerate(self.masking_percentages):
            for j, n_blocks in enumerate(self.num_blocks_range):
                for run in range(n_runs):
                    variant_num += 1
                    
                    # Skip if already completed
                    if (mask_pct, n_blocks, run) in completed_variants:
                        print(f"[{variant_num}/{total_variants}] Skipping (already completed): "
                              f"{mask_pct}% masking, {n_blocks} blocks, run {run+1}")
                        continue

                    print(f"\n[{variant_num}/{total_variants}] Evaluating: "
                          f"{mask_pct}% masking, {n_blocks} blocks, run {run+1}/{n_runs}")

                    try:
                        # Evaluate on this configuration
                        test_acc, kappa_score = evaluate_on_masked_test(
                            model=model,
                            num_users=len(self.config['user_ids']),
                            normalization_stats=normalization_stats,
                            masking_percentage=mask_pct,
                            num_blocks=n_blocks,
                            cache_size_gb=self.config.get('cache_size_gb', 4)
                        )

                        # Store results
                        accuracy_matrix[i, j, run] = test_acc
                        kappa_matrix[i, j, run] = kappa_score

                        result_entry = {
                            'masking_percentage': int(mask_pct),
                            'num_blocks': int(n_blocks),
                            'run': int(run),
                            'test_accuracy': float(test_acc),
                            'kappa_score': float(kappa_score)
                        }
                        detailed_results.append(result_entry)

                        print(f"  Accuracy: {test_acc:.4f}, Kappa: {kappa_score:.4f}")

                        # Save checkpoint after each variant
                        self.save_checkpoint(
                            stage='evaluating',
                            stage_data={
                                'completed_variants': len(detailed_results),
                                'total_variants': total_variants,
                                'last_completed': result_entry
                            },
                            accuracy_matrix=accuracy_matrix,
                            kappa_matrix=kappa_matrix,
                            detailed_results=detailed_results
                        )

                        # Periodic cleanup
                        if variant_num % 10 == 0:
                            gc.collect()

                    except Exception as e:
                        print(f"  ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        print("  Saving checkpoint and continuing...")
                        
                        # Save checkpoint even on error
                        self.save_checkpoint(
                            stage='evaluating',
                            stage_data={
                                'completed_variants': len(detailed_results),
                                'total_variants': total_variants,
                                'error': str(e)
                            },
                            accuracy_matrix=accuracy_matrix,
                            kappa_matrix=kappa_matrix,
                            detailed_results=detailed_results
                        )
                   
                        raise
                    finally:
                        # Always clean up GPU + Python references
                        del test_acc, kappa_score
                        torch.cuda.empty_cache()
                        gc.collect()                        
                         # Re-raise to allow outer handler to catch

        print(f"\n{'='*70}")
        print("Evaluation completed!")
        print(f"{'='*70}\n")

        return accuracy_matrix, kappa_matrix, detailed_results

    def step3_analyze_results(self, accuracy_matrix, kappa_matrix):
        """Analyze and summarize results"""
        print(f"\n{'='*70}")
        print("STEP 3: Analyzing Results")
        print(f"{'='*70}\n")

        # Average over runs
        avg_accuracy = np.mean(accuracy_matrix, axis=2)
        std_accuracy = np.std(accuracy_matrix, axis=2)
        avg_kappa = np.mean(kappa_matrix, axis=2)
        std_kappa = np.std(kappa_matrix, axis=2)

        # Print summary table
        print("\nAverage Test Accuracy (over all runs):")
        print(f"{'Masking %':<12}", end='')
        for n_blocks in self.num_blocks_range:
            print(f"{n_blocks} blocks{' '*4}", end='')
        print()
        print("-" * 70)

        for i, mask_pct in enumerate(self.masking_percentages):
            print(f"{mask_pct:>10}%  ", end='')
            for j in range(len(self.num_blocks_range)):
                acc = avg_accuracy[i, j]
                std = std_accuracy[i, j]
                print(f"{acc:.4f}±{std:.4f}  ", end='')
            print()

        print("\n\nAverage Kappa Score (over all runs):")
        print(f"{'Masking %':<12}", end='')
        for n_blocks in self.num_blocks_range:
            print(f"{n_blocks} blocks{' '*4}", end='')
        print()
        print("-" * 70)

        for i, mask_pct in enumerate(self.masking_percentages):
            print(f"{mask_pct:>10}%  ", end='')
            for j in range(len(self.num_blocks_range)):
                kappa = avg_kappa[i, j]
                std = std_kappa[i, j]
                print(f"{kappa:.4f}±{std:.4f}  ", end='')
            print()

        # Find best and worst configurations
        baseline_acc = avg_accuracy[0, 0]  # 0% masking, 1 block
        print(f"\n\nBaseline (0% masking): {baseline_acc:.4f}")

        # Exclude baseline from min/max
        masked_accuracies = avg_accuracy[1:].flatten()
        if len(masked_accuracies) > 0:
            best_masked = np.max(masked_accuracies)
            worst_masked = np.min(masked_accuracies)
            
            print(f"Best masked performance: {best_masked:.4f}")
            print(f"Worst masked performance: {worst_masked:.4f}")
            print(f"Performance degradation: {baseline_acc - worst_masked:.4f} ({(baseline_acc - worst_masked)/baseline_acc*100:.1f}%)")

        return avg_accuracy, std_accuracy, avg_kappa, std_kappa

    def step4_create_visualizations(self, accuracy_matrix, kappa_matrix):
        """Create visualization plots"""
        print(f"\n{'='*70}")
        print("STEP 4: Creating Visualizations")
        print(f"{'='*70}\n")

        # Average over runs
        avg_accuracy = np.mean(accuracy_matrix, axis=2)
        avg_kappa = np.mean(kappa_matrix, axis=2)

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Accuracy Heatmap
        im1 = axes[0].imshow(avg_accuracy, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0].set_xlabel('Number of Blocks', fontsize=12)
        axes[0].set_ylabel('Masking Percentage (%)', fontsize=12)
        axes[0].set_title('Test Accuracy vs Masking Configuration', fontsize=14, fontweight='bold')
        axes[0].set_xticks(range(len(self.num_blocks_range)))
        axes[0].set_xticklabels(self.num_blocks_range)
        axes[0].set_yticks(range(len(self.masking_percentages)))
        axes[0].set_yticklabels(self.masking_percentages)
        
        # Add text annotations
        for i in range(len(self.masking_percentages)):
            for j in range(len(self.num_blocks_range)):
                text = axes[0].text(j, i, f'{avg_accuracy[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im1, ax=axes[0], label='Accuracy')

        # Plot 2: Kappa Score Heatmap
        im2 = axes[1].imshow(avg_kappa, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1].set_xlabel('Number of Blocks', fontsize=12)
        axes[1].set_ylabel('Masking Percentage (%)', fontsize=12)
        axes[1].set_title('Kappa Score vs Masking Configuration', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(len(self.num_blocks_range)))
        axes[1].set_xticklabels(self.num_blocks_range)
        axes[1].set_yticks(range(len(self.masking_percentages)))
        axes[1].set_yticklabels(self.masking_percentages)
        
        # Add text annotations
        for i in range(len(self.masking_percentages)):
            for j in range(len(self.num_blocks_range)):
                text = axes[1].text(j, i, f'{avg_kappa[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im2, ax=axes[1], label='Kappa Score')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'masking_heatmaps.png', dpi=300, bbox_inches='tight')
        print(f"Saved heatmaps to {self.plots_dir / 'masking_heatmaps.png'}")
        plt.close()

        # Create line plots showing degradation
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 3: Accuracy vs Masking Percentage (for each num_blocks)
        for j, n_blocks in enumerate(self.num_blocks_range):
            axes[0].plot(self.masking_percentages, avg_accuracy[:, j], 
                        marker='o', label=f'{n_blocks} blocks', linewidth=2)
        
        axes[0].set_xlabel('Masking Percentage (%)', fontsize=12)
        axes[0].set_ylabel('Test Accuracy', fontsize=12)
        axes[0].set_title('Accuracy Degradation with Masking', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 4: Kappa vs Masking Percentage
        for j, n_blocks in enumerate(self.num_blocks_range):
            axes[1].plot(self.masking_percentages, avg_kappa[:, j], 
                        marker='o', label=f'{n_blocks} blocks', linewidth=2)
        
        axes[1].set_xlabel('Masking Percentage (%)', fontsize=12)
        axes[1].set_ylabel('Kappa Score', fontsize=12)
        axes[1].set_title('Kappa Score Degradation with Masking', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'masking_lines.png', dpi=300, bbox_inches='tight')
        print(f"Saved line plots to {self.plots_dir / 'masking_lines.png'}")
        plt.close()

    def step5_save_results(self, accuracy_matrix, kappa_matrix, detailed_results, base_val_acc):
        """Save all results to disk"""
        print(f"\n{'='*70}")
        print("STEP 5: Saving Results")
        print(f"{'='*70}\n")

        # Save matrices
        np.savez(
            self.results_dir / 'results_matrices.npz',
            accuracy_matrix=accuracy_matrix,
            kappa_matrix=kappa_matrix,
            masking_percentages=self.masking_percentages,
            num_blocks_range=self.num_blocks_range
        )
        print(f"Saved result matrices to {self.results_dir / 'results_matrices.npz'}")

        # Save detailed results as JSON
        results_dict = {
            'experiment_config': {
                'num_users': len(self.config['user_ids']),
                'user_ids': self.config['user_ids'],
                'eval_runs_per_variant': self.config['eval_runs_per_variant'],
                'masking_percentages': self.masking_percentages.tolist(),
                'num_blocks_range': self.num_blocks_range
            },
            'base_model': {
                'validation_accuracy': float(base_val_acc)
            },
            'detailed_results': detailed_results,
            'summary_statistics': {
                'avg_accuracy': np.mean(accuracy_matrix, axis=2).tolist(),
                'std_accuracy': np.std(accuracy_matrix, axis=2).tolist(),
                'avg_kappa': np.mean(kappa_matrix, axis=2).tolist(),
                'std_kappa': np.std(kappa_matrix, axis=2).tolist()
            }
        }

        with open(self.results_dir / 'detailed_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Saved detailed results to {self.results_dir / 'detailed_results.json'}")

        return results_dict

    def run_complete_experiment(self, resume=False):
        """Run the complete experiment pipeline"""
        print(f"\n{'='*70}")
        print("STARTING COMPLETE EXPERIMENT")
        print(f"{'='*70}\n")

        experiment_start_time = time.time()

        try:
            # Check for checkpoint
            checkpoint = None
            if resume:
                checkpoint = self.load_checkpoint()

            # Step 1: Train base model
            if checkpoint is None or checkpoint['stage'] == 'base_model_trained':
                model, normalization_stats, base_val_acc = self.step1_train_base_model()
            else:
                print("Base model already trained, loading...")
                import tensorflow as tf
                model = tf.keras.models.load_model(self.model_path)
                from data_loader_masking import calculate_normalization_stats
                normalization_stats = calculate_normalization_stats(
                    path=self.config['data_path'],
                    users=self.config['user_ids'],
                    frame_size=40,
                    max_samples=10000
                )
                base_val_acc = checkpoint['stage_data'].get('val_acc', 0.0)

            # Step 2: Evaluate on masked variants
            accuracy_matrix, kappa_matrix, detailed_results = self.step2_evaluate_masked_variants(
                model, normalization_stats
            )

            # Step 3: Analyze results
            self.step3_analyze_results(accuracy_matrix, kappa_matrix)

            # Step 4: Create visualizations
            self.step4_create_visualizations(accuracy_matrix, kappa_matrix)

            # Step 5: Save results
            results_data = self.step5_save_results(
                accuracy_matrix, kappa_matrix, detailed_results, base_val_acc
            )

            # Mark as complete
            self.save_checkpoint(
                stage='complete',
                stage_data={
                    'experiment_complete': True,
                    'total_variants_evaluated': len(detailed_results)
                },
                accuracy_matrix=accuracy_matrix,
                kappa_matrix=kappa_matrix,
                detailed_results=detailed_results
            )

            # Final summary
            experiment_time = time.time() - experiment_start_time

            print(f"\n{'='*70}")
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print(f"{'='*70}")
            print(f"Total experiment time: {experiment_time/3600:.2f} hours")
            print(f"Baseline accuracy (0% masking): {np.mean(accuracy_matrix[0,0,:]):.4f}")
            print(f"Results directory: {self.experiment_dir}")

            # Clean up checkpoint files
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.results_file.exists():
                self.results_file.unlink()
            print("Checkpoint files removed (experiment completed)")

            return results_data

        except KeyboardInterrupt:
            print(f"\n{'='*70}")
            print("EXPERIMENT INTERRUPTED BY USER")
            print(f"{'='*70}")
            print("Progress has been saved. Resume with resume=True")
            return None

        except Exception as e:
            print(f"\n{'='*70}")
            print(f"EXPERIMENT FAILED: {e}")
            print(f"{'='*70}")
            import traceback
            traceback.print_exc()
            print("\nProgress has been saved. You can try to resume with resume=True")
            return None


def main():
    """Main execution function"""

    config = {
        # Paths
        'experiment_dir': '/app/data/experiments/supervised_masking',
        'data_path': '/app/data/1.0.0',

        # For local testing (uncomment these and comment out above):
        # 'experiment_dir': '/path/to/local/experiments/supervised_masking',
        # 'data_path': '/path/to/local/eegmmidb/1.0.0',

        # Experiment parameters
        'user_ids': list(range(1, 110)),  # S001-S010
        'eval_runs_per_variant': 1,  # Number of evaluation runs per masking config

        # Training parameters
        'cache_size_gb': 8,  # Memory cache size
    }

    print("="*70)
    print("EEG Supervised Masking Experiment")
    print("="*70)
    print(f"Users: {len(config['user_ids'])} (S001-S{config['user_ids'][-1]:03d})")
    print(f"Experiment directory: {config['experiment_dir']}")
    print(f"Evaluation runs per variant: {config['eval_runs_per_variant']}")
    print("="*70)

    # Initialize and run experiment
    experiment = SupervisedMaskingExperiment(config)

    # Run with resume=True to continue from checkpoint
    results = experiment.run_complete_experiment(resume=False)

    if results is not None:
        print("\nExperiment completed successfully!")
        print(f"All results saved to: {experiment.experiment_dir}")
        return True
    else:
        print("\nExperiment interrupted or failed.")
        print("You can resume by running again with resume=True")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
