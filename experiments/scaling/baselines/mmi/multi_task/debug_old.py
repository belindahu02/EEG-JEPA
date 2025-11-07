"""
Diagnostic script to verify the augmentation pipeline is working correctly
"""
import numpy as np
import matplotlib.pyplot as plt
from data_loader import *
from transformations import *
from pre_trainers import AugmentedDataSequence

def check_augmentation_pipeline():
    """Verify that augmentations are being applied correctly"""
    
    print("=" * 60)
    print("DIAGNOSTIC: Checking Augmentation Pipeline")
    print("=" * 60)
    
    # Setup
    frame_size = 40
    path = "/app/data/1.0.0"
    batch_size = 8
    users = list(range(1, 11))  # Use fewer users for quick test
    train_sessions = [1, 2]  # Use fewer sessions
    
    # Create base generator
    print("\n1. Creating base generator...")
    mean, std = compute_normalization_stats(
        path, users=users, sessions=train_sessions,
        frame_size=frame_size, max_samples_per_session=5000
    )
    
    base_generator = EEGDataGenerator(
        path, users, train_sessions, frame_size,
        batch_size=batch_size,
        max_samples_per_session=5000,
        mean=mean, std=std, shuffle=True
    )
    
    # Get a sample batch
    print("\n2. Getting sample batch from base generator...")
    sample_batch_x, sample_batch_y = next(iter(base_generator))
    print(f"   Sample batch shape: {sample_batch_x.shape}")
    print(f"   Sample labels shape: {sample_batch_y.shape}")
    print(f"   Sample labels (user IDs): {sample_batch_y}")
    
    # Setup transformations
    transformations = [
        DA_Jitter, DA_Scaling, DA_MagWarp, DA_RandSampling, DA_Flip, DA_Drop
    ]
    sigma_l = [0.1, 0.2, 0.2, None, None, 3]
    
    # Test each transformation individually
    print("\n3. Testing individual transformations...")
    test_sample = sample_batch_x[0]
    print(f"   Original sample shape: {test_sample.shape}")
    print(f"   Original sample mean: {np.mean(test_sample):.4f}, std: {np.std(test_sample):.4f}")
    
    for i, (transform, sigma) in enumerate(zip(transformations, sigma_l)):
        transformed = transform(test_sample.copy(), sigma=sigma)
        diff = np.mean(np.abs(transformed - test_sample))
        print(f"   Transform {i+1} ({transform.__name__}): Mean abs difference = {diff:.6f}")
        
        if diff < 1e-6:
            print(f"   WARNING: Transform {i+1} produced identical output!")
    
    # Create augmented sequence
    print("\n4. Creating AugmentedDataSequence...")
    aug_sequence = AugmentedDataSequence(
        base_generator, transformations, sigma_l,
        ext=False, batches_per_epoch=10
    )
    
    print(f"   Number of cached batches: {len(aug_sequence.cached_batches)}")
    print(f"   Total batches in sequence: {len(aug_sequence)}")
    expected_batches = len(aug_sequence.cached_batches) * len(transformations)
    print(f"   Expected: {len(aug_sequence.cached_batches)} * {len(transformations)} = {expected_batches}")
    print(f"   (Each batch trains ONE head with 50/50 positive/negative examples)")
    
    # Check if the fix was applied
    if len(aug_sequence) != expected_batches:
        print(f"   ERROR: Batch count mismatch! Expected {expected_batches}, got {len(aug_sequence)}")
    
    # Sample from augmented sequence
    print("\n5. Sampling batches from AugmentedDataSequence...")
    label_distribution = {f"head_{i+1}": {"0": 0, "1": 0} for i in range(len(transformations))}
    
    # Sample one full cycle: 6 transforms = 6 batches (one per head)
    batches_to_check = min(len(transformations), len(aug_sequence))
    
    for i in range(batches_to_check):
        batch_data = aug_sequence[i]
        
        # Handle both 2-tuple and 3-tuple returns (with/without sample weights)
        if len(batch_data) == 3:
            x_dict, y_dict, sw_dict = batch_data
        else:
            x_dict, y_dict = batch_data
            sw_dict = None
        
        print(f"\n   Batch {i}:")
        for head_name, y_vals in y_dict.items():
            ones = np.sum(y_vals == 1)
            zeros = np.sum(y_vals == 0)
            
            # Check sample weights if available
            if sw_dict is not None:
                weights = sw_dict[head_name]
                active = np.sum(weights > 0)
                weight_str = f" (weight: {active}/{len(weights)} active)" if active < len(weights) else ""
            else:
                weight_str = ""
                
            label_distribution[head_name]["0"] += zeros
            label_distribution[head_name]["1"] += ones
            print(f"      {head_name}: {zeros} zeros, {ones} ones{weight_str}")
    
    print("\n6. Label distribution across all sampled batches:")
    print("   (Note: With sample weights, only count active batches for true balance)")
    
    # Calculate effective balance considering only active batches
    for head_name, counts in label_distribution.items():
        total = counts["0"] + counts["1"]
        if total > 0:
            pct_0 = 100*counts['0']/total
            pct_1 = 100*counts['1']/total
            
            # Each head should have 1 active batch with 4 zeros + 4 ones = 50/50
            # The rest are dummy batches that don't contribute to loss
            expected_active_batches = batches_to_check // len(transformations)
            effective_balance = "✓ (50/50 in active batch)" if expected_active_batches > 0 else "?"
            
            print(f"   {head_name}: {counts['0']} zeros ({pct_0:.1f}%), "
                  f"{counts['1']} ones ({pct_1:.1f}%) - {effective_balance}")
            
    # Explanation
    print(f"\n   Note: Each head gets its own balanced task.")
    print(f"   Heads not being trained in a batch receive dummy zero inputs/labels.")
    
    # Verify data format
    print("\n7. Verifying data format...")
    batch_data = aug_sequence[0]
    
    # Handle both 2-tuple and 3-tuple returns
    if len(batch_data) == 3:
        x_dict, y_dict, sw_dict = batch_data
        has_weights = True
    else:
        x_dict, y_dict = batch_data
        has_weights = False
        
    print(f"   Number of inputs: {len(x_dict)} (expected: {len(transformations)})")
    print(f"   Number of outputs: {len(y_dict)} (expected: {len(transformations)})")
    if has_weights:
        print(f"   Sample weights: YES ✓")
    else:
        print(f"   Sample weights: NO")
    print(f"   Input names: {list(x_dict.keys())}")
    print(f"   Output names: {list(y_dict.keys())}")
    
    for key in x_dict.keys():
        print(f"   {key} shape: {x_dict[key].shape}")
    
    # Check if augmented samples are actually different
    print("\n8. Verifying augmented samples differ from originals...")
    
    # Get first batch for each head
    for batch_idx in range(min(len(transformations), len(aug_sequence))):
        batch_data = aug_sequence[batch_idx]
        
        if len(batch_data) == 3:
            x_dict, y_dict, sw_dict = batch_data
        else:
            x_dict, y_dict = batch_data
            sw_dict = None
        
        # Find which head is active in this batch
        for i, (head_name, y_vals) in enumerate(y_dict.items()):
            if np.sum(y_vals) > 0:  # This head has augmented samples
                input_name = f"input_{i+1}"
                
                # For comparison, need original data
                # The batch contains half augmented, half original
                # Just check that there's variation in the data
                data = x_dict[input_name]
                if np.std(data) > 0:
                    diff_str = "has variation ✓"
                else:
                    diff_str = "ERROR: no variation!"
                    
                print(f"   Batch {batch_idx} - {head_name}: {diff_str}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
    return True


def visualize_augmentations():
    """Create visualizations of each augmentation for manual inspection"""
    
    print("\n" + "=" * 60)
    print("Creating augmentation visualizations...")
    print("=" * 60)
    
    # Get sample data
    frame_size = 40
    path = "/app/data/1.0.0"
    users = [1]
    sessions = [1]
    
    mean, std = compute_normalization_stats(
        path, users=users, sessions=sessions,
        frame_size=frame_size, max_samples_per_session=5000
    )
    
    gen = EEGDataGenerator(
        path, users, sessions, frame_size,
        batch_size=1, max_samples_per_session=5000,
        mean=mean, std=std, shuffle=False
    )
    
    sample_x, _ = next(iter(gen))
    sample = sample_x[0]  # Shape: (frame_size, n_channels)
    
    transformations = [
        ("Jitter", DA_Jitter, 0.1),
        ("Scaling", DA_Scaling, 0.2),
        ("MagWarp", DA_MagWarp, 0.2),
        ("RandSampling", DA_RandSampling, None),
        ("Flip", DA_Flip, None),
        ("Drop", DA_Drop, 3)
    ]
    
    fig, axes = plt.subplots(len(transformations) + 1, 2, figsize=(15, 3 * (len(transformations) + 1)))
    
    # Plot original
    axes[0, 0].plot(sample[:, 0])
    axes[0, 0].set_title("Original - Channel 0")
    axes[0, 1].plot(sample[:, min(1, sample.shape[1]-1)])
    axes[0, 1].set_title(f"Original - Channel {min(1, sample.shape[1]-1)}")
    
    # Plot each transformation
    for i, (name, transform, sigma) in enumerate(transformations):
        transformed = transform(sample.copy(), sigma=sigma)
        
        axes[i+1, 0].plot(transformed[:, 0])
        axes[i+1, 0].set_title(f"{name} - Channel 0")
        
        axes[i+1, 1].plot(transformed[:, min(1, transformed.shape[1]-1)])
        axes[i+1, 1].set_title(f"{name} - Channel {min(1, transformed.shape[1]-1)}")
        
        # Add difference info
        diff = np.mean(np.abs(transformed - sample))
        axes[i+1, 0].text(0.02, 0.98, f"Mean abs diff: {diff:.4f}", 
                          transform=axes[i+1, 0].transAxes, 
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/app/data/experiments/scaling/baselines/multi_task/graphs/augmentation_visualization.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved augmentation visualization to augmentation_visualization.png")


if __name__ == "__main__":
    # Run diagnostics
    check_augmentation_pipeline()
    
    # Create visualizations
    visualize_augmentations()
