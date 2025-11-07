"""
Diagnostic script to check experiment state and provide instructions
"""

import json
from pathlib import Path
import sys


def diagnose_experiment(experiment_dir):
    """
    Check experiment state and provide clear instructions
    
    Args:
        experiment_dir: Path to experiment directory
    """
    experiment_dir = Path(experiment_dir)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT STATE DIAGNOSIS")
    print("=" * 70)
    print(f"Directory: {experiment_dir}\n")
    
    # Check if directory exists
    if not experiment_dir.exists():
        print("❌ Experiment directory does not exist")
        print(f"   Create it by running the experiment for the first time")
        return
    
    # Check for checkpoint
    checkpoint_file = experiment_dir / "checkpoints" / "experiment_checkpoint.json"
    has_checkpoint = checkpoint_file.exists()
    
    print(f"{'✓' if has_checkpoint else '✗'} Checkpoint file: {checkpoint_file}")
    
    if has_checkpoint:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f"   Stage: {checkpoint.get('stage', 'unknown')}")
        stage_data = checkpoint.get('stage_data', {})
        print(f"   Keys: {list(stage_data.keys())}")
        if 'completed_variants' in stage_data:
            print(f"   Completed variants: {len(stage_data['completed_variants'])}/55")
    
    # Check for model
    model_file = experiment_dir / "models" / "base_model.keras"
    has_model = model_file.exists()
    print(f"\n{'✓' if has_model else '✗'} Model file: {model_file}")
    if has_model:
        model_size_mb = model_file.stat().st_size / 1024 / 1024
        print(f"   Size: {model_size_mb:.1f} MB")
    
    # Check for normalization stats
    norm_stats_file = experiment_dir / "models" / "norm_stats.json"
    has_norm_stats = norm_stats_file.exists()
    print(f"{'✓' if has_norm_stats else '✗'} Norm stats: {norm_stats_file}")
    
    if has_norm_stats:
        with open(norm_stats_file, 'r') as f:
            norm_stats = json.load(f)
        print(f"   Channels: {norm_stats.get('n_channels', 'unknown')}")
        print(f"   Classes: {norm_stats.get('num_classes', 'unknown')}")
    
    # Check for partial results
    results_file = experiment_dir / "checkpoints" / "partial_results.npz"
    has_results = results_file.exists()
    print(f"\n{'✓' if has_results else '✗'} Partial results: {results_file}")
    
    # Check for final results
    final_results = experiment_dir / "results" / "experiment_results.json"
    has_final = final_results.exists()
    print(f"{'✓' if has_final else '✗'} Final results: {final_results}")
    
    # Check for plots
    plots_dir = experiment_dir / "plots"
    plot_files = list(plots_dir.glob("*.png")) if plots_dir.exists() else []
    print(f"\n{'✓' if plot_files else '✗'} Plots directory: {plots_dir}")
    if plot_files:
        print(f"   Found {len(plot_files)} plot files")
    
    # Provide recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if has_final:
        print("✓ Experiment is COMPLETE!")
        print(f"  View results in: {experiment_dir / 'results'}")
        print(f"  View plots in: {experiment_dir / 'plots'}")
        
    elif has_checkpoint and has_model and has_norm_stats:
        stage = checkpoint.get('stage', 'unknown')
        
        if stage == 'complete':
            print("✓ Experiment is marked complete in checkpoint")
            if not has_final:
                print("⚠  But final results are missing - may need to regenerate")
        
        elif stage in ['trained', 'evaluation']:
            completed = len(stage_data.get('completed_variants', []))
            remaining = 55 - completed
            
            print(f"⏸  Experiment is IN PROGRESS")
            print(f"   Stage: {stage}")
            print(f"   Completed: {completed}/55 variants")
            print(f"   Remaining: {remaining} variants")
            print(f"\n   To continue, run:")
            print(f"   python experiment_masking_da.py")
            print(f"   (with resume=True in the code)")
            print(f"\n   Or use:")
            print(f"   python resume_experiment.py")
            
            # Check if checkpoint needs fixing
            if 'base_model_path' not in stage_data:
                print(f"\n⚠  WARNING: Checkpoint is missing 'base_model_path'")
                print(f"   The updated code will handle this automatically")
                print(f"   Or run: python fix_checkpoint.py {experiment_dir}")
        
        else:
            print(f"⚠  Experiment is in an unknown state: {stage}")
            print(f"   You may need to restart from scratch")
    
    elif has_model and has_norm_stats:
        print("⚠  Training completed but no checkpoint found")
        print("   The model and normalization stats exist")
        print("   You can continue by running the experiment with resume=True")
        print("   It will automatically find the model")
    
    else:
        print("▶  Experiment has NOT been started")
        print("   Run: python experiment_masking_da.py")
        print("   (with resume=False in the code)")
    
    print("=" * 70)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python diagnose_experiment.py <experiment_dir>")
        print("\nExample:")
        print("  python diagnose_experiment.py /app/data/experiments/da_masking_new")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    diagnose_experiment(experiment_dir)


if __name__ == "__main__":
    main()
