# =============================================
# CRITICAL FIXES for Dataset Loading Issues
# =============================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from collections import Counter


class FixedSpectrogramDataset(Dataset):
    """
    Fixed dataset with proper normalization and data validation
    """

    def __init__(self, file_paths, labels, normalization='log_scale', add_channel_dim=True,
                 augment=False, cache_size=100):
        self.file_paths = file_paths
        self.labels = labels
        self.normalization = normalization
        self.add_channel_dim = add_channel_dim
        self.augment = augment

        # Cache
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

        # CRITICAL: Pre-compute normalization statistics from a sample of data
        self._compute_global_stats()

    def _compute_global_stats(self):
        """
        CRITICAL FIX: Compute global statistics for consistent normalization
        """
        print("Computing global normalization statistics...")

        # Sample files to compute statistics (use every 10th file to be efficient)
        sample_indices = np.arange(0, len(self.file_paths), max(1, len(self.file_paths) // 50))
        sample_specs = []

        for idx in sample_indices[:20]:  # Limit to 20 samples for efficiency
            try:
                spec = np.load(self.file_paths[idx])
                if not np.isfinite(spec).all():
                    print(f"Warning: Non-finite values in {self.file_paths[idx]}")
                    continue
                sample_specs.append(spec.flatten())
            except Exception as e:
                print(f"Error loading {self.file_paths[idx]}: {e}")
                continue

        if not sample_specs:
            raise ValueError("No valid spectrograms found for computing statistics!")

        all_values = np.concatenate(sample_specs)

        # Global statistics
        self.global_mean = np.mean(all_values)
        self.global_std = np.std(all_values)
        self.global_min = np.min(all_values)
        self.global_max = np.max(all_values)

        print(f"Global stats - Mean: {self.global_mean:.4f}, Std: {self.global_std:.4f}")
        print(f"Global range: [{self.global_min:.4f}, {self.global_max:.4f}]")

        # For log scaling - compute safe bounds
        if self.normalization == 'log_scale':
            # Find minimum positive value for safe log scaling
            positive_values = all_values[all_values > 0]
            if len(positive_values) > 0:
                self.min_positive = np.min(positive_values)
            else:
                self.min_positive = 1e-8
            print(f"Minimum positive value for log scaling: {self.min_positive}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Check cache first
        if file_path in self.cache:
            self.cache_order.remove(file_path)
            self.cache_order.append(file_path)
            spec = self.cache[file_path].copy()
        else:
            # Load from disk
            spec = np.load(file_path)

            # CRITICAL: Validate loaded data
            if not np.isfinite(spec).all():
                print(f"Warning: Non-finite values in {file_path}, replacing with zeros")
                spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

            # Add to cache
            if len(self.cache) >= self.cache_size:
                oldest = self.cache_order.pop(0)
                del self.cache[oldest]

            self.cache[file_path] = spec.copy()
            self.cache_order.append(file_path)

        # CRITICAL: Apply FIXED normalization
        if self.normalization:
            spec = self._normalize_single_fixed(spec, self.normalization)

        # Add channel dimension if needed
        if self.add_channel_dim:
            spec = spec[np.newaxis, :, :]

        # Apply augmentation if training
        if self.augment:
            spec = self._augment_single(spec)

        return torch.tensor(spec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _normalize_single_fixed(self, spec, method):
        """
        FIXED normalization that prevents gradient explosion
        """
        if method == 'log_scale':
            # Use global minimum positive value for consistent scaling
            spec_positive = np.abs(spec) + self.min_positive
            result = np.log(spec_positive)

            # CRITICAL: Apply global z-score normalization after log
            result = (result - np.mean(result)) / (np.std(result) + 1e-8)

            # Clip extreme values to prevent gradient explosion
            result = np.clip(result, -10.0, 10.0)
            return result

        elif method == 'global_z_score':
            # RECOMMENDED: Use global statistics for consistent normalization
            result = (spec - self.global_mean) / (self.global_std + 1e-8)
            result = np.clip(result, -5.0, 5.0)  # Clip to reasonable range
            return result

        elif method == 'robust_minmax':
            # Use global min/max with clipping
            if self.global_max - self.global_min < 1e-8:
                return np.zeros_like(spec)
            else:
                result = (spec - self.global_min) / (self.global_max - self.global_min)
                return np.clip(result, 0.0, 1.0)

        elif method == 'local_z_score':
            # Local normalization but with clipping
            mean = np.mean(spec)
            std = np.std(spec)
            if std < 1e-8:
                return np.zeros_like(spec)
            else:
                result = (spec - mean) / std
                return np.clip(result, -5.0, 5.0)  # CRITICAL: Clip extreme values

        return spec

    def _augment_single(self, spec):
        """Apply light augmentation"""
        # Reduce augmentation intensity to prevent issues
        if np.random.rand() > 0.7:  # Reduced probability
            if spec.ndim == 3:  # (channel, time, freq)
                t_max = spec.shape[1]
                mask_size = np.random.randint(1, min(t_max // 20, 25))  # Smaller masks
                start = np.random.randint(0, max(1, t_max - mask_size))
                spec[:, start:start + mask_size, :] *= 0.1  # Reduce instead of zero
            else:
                t_max = spec.shape[0]
                mask_size = np.random.randint(1, min(t_max // 20, 25))
                start = np.random.randint(0, max(1, t_max - mask_size))
                spec[start:start + mask_size, :] *= 0.1

        if np.random.rand() > 0.7:  # Frequency masking
            if spec.ndim == 3:
                f_max = spec.shape[2]
                mask_size = np.random.randint(1, min(f_max // 20, 25))
                start = np.random.randint(0, max(1, f_max - mask_size))
                spec[:, :, start:start + mask_size] *= 0.1
            else:
                f_max = spec.shape[1]
                mask_size = np.random.randint(1, min(f_max // 20, 25))
                start = np.random.randint(0, max(1, f_max - mask_size))
                spec[:, start:start + mask_size] *= 0.1

        return spec


def create_fixed_dataloaders(data_path, user_ids, samples_per_user,
                             normalization='global_z_score', batch_size=16,
                             augment_train=False, cache_size=100):
    """
    FIXED data loader creation with proper validation
    """
    print("Creating FIXED memory-efficient data loaders...")

    # Get all file paths and labels
    from data_loader_2d_lazy import get_file_paths_and_labels
    file_paths, labels, sessions = get_file_paths_and_labels(data_path, user_ids)

    if len(file_paths) == 0:
        raise ValueError("No valid spectrogram files found")

    print(f"Total files found: {len(file_paths)}")
    print(f"User sessions: {dict(zip(user_ids, sessions))}")

    # Convert to numpy arrays
    file_paths = np.array(file_paths)
    labels = np.array(labels)

    # FIXED: Better sample limiting logic
    if samples_per_user is not None:
        limited_paths = []
        limited_labels = []

        for user_idx in range(len(user_ids)):
            user_mask = labels == user_idx
            user_paths = file_paths[user_mask]
            user_labels = labels[user_mask]

            print(f"User {user_ids[user_idx]} (idx {user_idx}): {len(user_paths)} files available")

            if len(user_paths) == 0:
                print(f"WARNING: No files found for user {user_ids[user_idx]}")
                continue

            # Sample randomly if we have more than requested
            if len(user_paths) > samples_per_user:
                indices = np.random.choice(len(user_paths), samples_per_user, replace=False)
                user_paths = user_paths[indices]
                user_labels = user_labels[indices]

            limited_paths.extend(user_paths)
            limited_labels.extend(user_labels)
            print(f"  -> Using {len(user_paths)} files")

        file_paths = np.array(limited_paths)
        labels = np.array(limited_labels)

    print(f"Final dataset: {len(file_paths)} files")
    print(f"Label distribution after limiting: {dict(Counter(labels))}")

    # FIXED: Improved data splitting per user
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for user_idx in range(len(user_ids)):
        user_mask = labels == user_idx
        user_paths = file_paths[user_mask]
        user_labels = labels[user_mask]

        n_samples = len(user_paths)
        if n_samples == 0:
            continue

        print(f"Splitting {n_samples} samples for user {user_ids[user_idx]}")

        if n_samples < 3:
            # Put all samples in training if too few
            train_paths.extend(user_paths)
            train_labels.extend(user_labels)
            print(f"  -> All {n_samples} samples to train (too few to split)")
        else:
            # Proper splitting
            n_train = max(int(n_samples * 0.7), 1)
            n_val = max(int(n_samples * 0.15), 1)
            n_test = n_samples - n_train - n_val

            # Ensure we have at least 1 test sample if possible
            if n_test < 1 and n_samples >= 3:
                n_test = 1
                n_val = min(n_val, n_samples - n_train - n_test)

            print(f"  -> Train: {n_train}, Val: {n_val}, Test: {n_test}")

            # Random permutation for splitting
            indices = np.random.permutation(n_samples)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val] if n_val > 0 else []
            test_idx = indices[n_train + n_val:] if n_test > 0 else []

            train_paths.extend(user_paths[train_idx])
            train_labels.extend(user_labels[train_idx])

            if len(val_idx) > 0:
                val_paths.extend(user_paths[val_idx])
                val_labels.extend(user_labels[val_idx])

            if len(test_idx) > 0:
                test_paths.extend(user_paths[test_idx])
                test_labels.extend(user_labels[test_idx])

    print(f"\nFinal splits:")
    print(f"  Train: {len(train_paths)} samples")
    print(f"  Val: {len(val_paths)} samples")
    print(f"  Test: {len(test_paths)} samples")

    # Verify all classes are present in training
    train_classes = set(train_labels)
    expected_classes = set(range(len(user_ids)))
    missing_classes = expected_classes - train_classes

    if missing_classes:
        print(f"CRITICAL ERROR: Missing classes in training set: {missing_classes}")
        # Add at least one sample from each missing class
        for missing_class in missing_classes:
            user_mask = labels == missing_class
            if np.any(user_mask):
                # Take one sample from val or test and move to train
                available_paths = file_paths[user_mask]
                available_labels = labels[user_mask]
                if len(available_paths) > 0:
                    train_paths.append(available_paths[0])
                    train_labels.append(available_labels[0])
                    print(f"  -> Added emergency sample for class {missing_class}")

    # Create fixed datasets
    train_dataset = FixedSpectrogramDataset(
        train_paths, train_labels, normalization,
        add_channel_dim=True, augment=augment_train,
        cache_size=cache_size
    )

    val_dataset = FixedSpectrogramDataset(
        val_paths, val_labels, normalization,
        add_channel_dim=True, augment=False,
        cache_size=cache_size // 2
    )

    test_dataset = FixedSpectrogramDataset(
        test_paths, test_labels, normalization,
        add_channel_dim=True, augment=False,
        cache_size=cache_size // 2
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader, sessions


def diagnose_gradient_explosion(data_path, user_ids, samples_per_user=50):
    """
    Specific test for gradient explosion issues
    """
    print("=" * 60)
    print("GRADIENT EXPLOSION DIAGNOSIS")
    print("=" * 60)

    # Test different normalization methods
    normalizations = ['global_z_score', 'robust_minmax', 'log_scale', 'local_z_score']

    for norm_method in normalizations:
        print(f"\nTesting normalization: {norm_method}")

        try:
            train_loader, val_loader, test_loader, _ = create_fixed_dataloaders(
                data_path, user_ids, samples_per_user,
                normalization=norm_method, batch_size=8
            )

            # Get a batch and check its properties
            X_batch, y_batch = next(iter(train_loader))

            print(f"  Batch shape: {X_batch.shape}")
            print(f"  Data range: [{X_batch.min().item():.4f}, {X_batch.max().item():.4f}]")
            print(f"  Data mean: {X_batch.mean().item():.4f}")
            print(f"  Data std: {X_batch.std().item():.4f}")

            # Check for problematic values
            has_nan = torch.isnan(X_batch).any()
            has_inf = torch.isinf(X_batch).any()
            extreme_values = (torch.abs(X_batch) > 100).any()

            print(f"  Has NaN: {has_nan}")
            print(f"  Has Inf: {has_inf}")
            print(f"  Has extreme values (>100): {extreme_values}")

            # Quick gradient test
            if not (has_nan or has_inf or extreme_values):
                gradient_stable = test_gradient_stability(X_batch, y_batch, len(user_ids))
                print(f"  Gradient stable: {gradient_stable}")
            else:
                print(f"  ❌ Skipping gradient test due to data issues")

        except Exception as e:
            print(f"  ❌ ERROR with {norm_method}: {e}")

    return True


def test_gradient_stability(X_batch, y_batch, num_classes):
    """
    Test if gradients explode with this data
    """
    # Create tiny test model
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(np.prod(X_batch.shape[1:]), 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, num_classes)
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Forward pass
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)

    # Check loss value
    loss_val = loss.item()
    print(f"    Initial loss: {loss_val:.4f}")

    if loss_val > 1000 or not np.isfinite(loss_val):
        return False

    # Backward pass
    loss.backward()

    # Check gradients
    total_grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    print(f"    Gradient norm: {total_grad_norm:.4f}")

    return total_grad_norm < 100.0  # Reasonable gradient norm


def create_data_visualization(data_path, user_ids, samples_per_user=20):
    """
    Create visualizations to understand your data better
    """
    print("Creating data visualizations...")

    # Test with fixed loader
    train_loader, _, _, _ = create_fixed_dataloaders(
        data_path, user_ids, samples_per_user,
        normalization='global_z_score', batch_size=len(user_ids) * 2
    )

    # Get one batch with samples from each user
    X_batch, y_batch = next(iter(train_loader))

    # Create visualization
    fig, axes = plt.subplots(2, len(user_ids), figsize=(4 * len(user_ids), 8))
    if len(user_ids) == 1:
        axes = axes.reshape(2, 1)

    for user_idx in range(len(user_ids)):
        # Find samples for this user
        user_mask = y_batch == user_idx
        if user_mask.any():
            user_sample = X_batch[user_mask][0, 0].numpy()  # First sample, remove channel dim

            # Plot spectrogram
            im1 = axes[0, user_idx].imshow(user_sample.T, aspect='auto', origin='lower', cmap='viridis')
            axes[0, user_idx].set_title(f'User {user_ids[user_idx]} Spectrogram')
            axes[0, user_idx].set_xlabel('Time')
            axes[0, user_idx].set_ylabel('Frequency')
            plt.colorbar(im1, ax=axes[0, user_idx])

            # Plot histogram of values
            axes[1, user_idx].hist(user_sample.flatten(), bins=50, alpha=0.7)
            axes[1, user_idx].set_title(f'User {user_ids[user_idx]} Value Distribution')
            axes[1, user_idx].set_xlabel('Value')
            axes[1, user_idx].set_ylabel('Count')
            axes[1, user_idx].grid(True, alpha=0.3)

            # Add statistics as text
            stats_text = f"Mean: {np.mean(user_sample):.3f}\nStd: {np.std(user_sample):.3f}\nRange: [{np.min(user_sample):.3f}, {np.max(user_sample):.3f}]"
            axes[1, user_idx].text(0.05, 0.95, stats_text, transform=axes[1, user_idx].transAxes,
                                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[0, user_idx].text(0.5, 0.5, f'No sample\nfor User {user_ids[user_idx]}',
                                   ha='center', va='center', transform=axes[0, user_idx].transAxes)
            axes[1, user_idx].text(0.5, 0.5, f'No sample\nfor User {user_ids[user_idx]}',
                                   ha='center', va='center', transform=axes[1, user_idx].transAxes)

    plt.tight_layout()
    plt.savefig('fixed_dataset_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to 'fixed_dataset_visualization.png'")


# Usage example
if __name__ == "__main__":
    # Replace with your actual values
    DATA_PATH = "/app/data/grouped_embeddings"
    USER_IDS = list(range(1, 110))  # Increased to more users for better evaluation
    SAMPLES_PER_USER = 133

    print("Step 1: Diagnosing gradient explosion...")
    diagnose_gradient_explosion(DATA_PATH, USER_IDS)

    print("\nStep 2: Creating visualizations...")
    create_data_visualization(DATA_PATH, USER_IDS)

    print("\nStep 3: Testing improved data loader...")
    train_loader, val_loader, test_loader, _ = create_fixed_dataloaders(
        DATA_PATH, USER_IDS, samples_per_user=SAMPLES_PER_USER,
        normalization='global_z_score',  # Try this instead of log_scale
        batch_size=16
    )

