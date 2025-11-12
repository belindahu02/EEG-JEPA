import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import numpy as np
from collections import defaultdict, Counter


class SpectrogramDataset(Dataset):
    """Dataset for pre-normalised spectrograms"""
    
    def __init__(self, file_paths, labels, normalization='none', add_channel_dim=True,
                 augment=False, cache_size=100):
        self.file_paths = file_paths
        self.labels = labels
        self.normalization = normalization
        self.add_channel_dim = add_channel_dim
        self.augment = augment
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        if file_path in self.cache:
            self.cache_order.remove(file_path)
            self.cache_order.append(file_path)
            spec = self.cache[file_path].copy()
        else:
            spec = np.load(file_path)
            if not np.isfinite(spec).all():
                spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
            
            if len(self.cache) >= self.cache_size:
                oldest = self.cache_order.pop(0)
                del self.cache[oldest]
            
            self.cache[file_path] = spec.copy()
            self.cache_order.append(file_path)

        if self.add_channel_dim:
            spec = spec[np.newaxis, :, :]

        if self.augment:
            spec = self._augment_single(spec)

        return torch.tensor(spec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _augment_single(self, spec):
        if np.random.rand() > 0.8:
            if spec.ndim == 3:
                t_max = spec.shape[1]
                mask_size = np.random.randint(1, min(t_max // 30, 20))
                start = np.random.randint(0, max(1, t_max - mask_size))
                spec[:, start:start + mask_size, :] *= np.random.uniform(0.1, 0.3)
        return spec


def get_session_based_file_paths_and_labels(data_path, user_ids):
    """Get file paths and labels organised by sessions"""
    user_session_data = {}

    for user_idx, user_id in enumerate(user_ids):
        user_folder = f"user{user_id}"
        user_path = os.path.join(data_path, user_folder)

        if not os.path.exists(user_path):
            print(f"Warning: Path {user_path} does not exist")
            user_session_data[user_idx] = {}
            continue

        sessions = {}

        for session_folder in sorted(os.listdir(user_path)):
            session_path = os.path.join(user_path, session_folder)
            
            if os.path.isdir(session_path) and session_folder.startswith(f"user{user_id}_"):
                session_files = []

                for npy_file in sorted(os.listdir(session_path)):
                    if npy_file.endswith("_stacked.npy"):
                        file_path = os.path.join(session_path, npy_file)
                        session_files.append(file_path)

                if session_files:
                    sessions[session_folder] = session_files

        user_session_data[user_idx] = sessions

    return user_session_data


def create_session_based_splits(user_session_data, user_ids, test_sessions_per_user=1, 
                                 val_sessions_per_user=1):
    """Create train/val/test splits at session level"""
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    test_paths = []
    test_labels = []

    print("\n" + "=" * 80)
    print("Creating session-based splits")
    print("=" * 80)

    for user_idx, user_id in enumerate(user_ids):
        if user_idx not in user_session_data or not user_session_data[user_idx]:
            continue

        sessions = user_session_data[user_idx]
        session_names = sorted(sessions.keys())
        total_sessions = len(session_names)

        if total_sessions < (test_sessions_per_user + val_sessions_per_user + 1):
            if total_sessions == 1:
                train_session_names = session_names
                val_session_names = []
                test_session_names = []
            elif total_sessions == 2:
                train_session_names = session_names[:1]
                val_session_names = []
                test_session_names = session_names[-1:]
            elif total_sessions == 3:
                train_session_names = session_names[:1]
                val_session_names = session_names[1:2]
                test_session_names = session_names[-1:]
            else:
                test_start = total_sessions - test_sessions_per_user
                val_start = test_start - val_sessions_per_user
                train_session_names = session_names[:val_start]
                val_session_names = session_names[val_start:test_start]
                test_session_names = session_names[test_start:]
        else:
            test_start = total_sessions - test_sessions_per_user
            val_start = test_start - val_sessions_per_user
            train_session_names = session_names[:val_start]
            val_session_names = session_names[val_start:test_start]
            test_session_names = session_names[test_start:]

        train_count = 0
        for session_name in train_session_names:
            if session_name in sessions:
                session_files = sessions[session_name]
                train_paths.extend(session_files)
                train_labels.extend([user_idx] * len(session_files))
                train_count += len(session_files)

        val_count = 0
        for session_name in val_session_names:
            if session_name in sessions:
                session_files = sessions[session_name]
                val_paths.extend(session_files)
                val_labels.extend([user_idx] * len(session_files))
                val_count += len(session_files)

        test_count = 0
        for session_name in test_session_names:
            if session_name in sessions:
                session_files = sessions[session_name]
                test_paths.extend(session_files)
                test_labels.extend([user_idx] * len(session_files))
                test_count += len(session_files)

    print(f"\nFinal split: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
    print("=" * 80 + "\n")

    # Verify all classes in training
    train_classes = set(train_labels)
    expected_classes = set(range(len(user_ids)))
    missing_classes = expected_classes - train_classes

    if missing_classes:
        print(f"⚠  ERROR: Missing classes in training: {[user_ids[i] for i in missing_classes]}")
        for missing_class in missing_classes:
            val_indices = [i for i, label in enumerate(val_labels) if label == missing_class]
            test_indices = [i for i, label in enumerate(test_labels) if label == missing_class]

            if val_indices:
                idx = val_indices[0]
                train_paths.append(val_paths[idx])
                train_labels.append(val_labels[idx])
                val_paths.pop(idx)
                val_labels.pop(idx)
                print(f"  ✓ Moved class {user_ids[missing_class]} from val to train")
            elif test_indices:
                idx = test_indices[0]
                train_paths.append(test_paths[idx])
                train_labels.append(test_labels[idx])
                test_paths.pop(idx)
                test_labels.pop(idx)
                print(f"  ✓ Moved class {user_ids[missing_class]} from test to train")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def create_weighted_sampler(labels):
    """
    Create weighted sampler to balance classes
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    
    # Oversample minority classes
    weights_per_class = 1.0 / class_counts
    weights = weights_per_class[labels]
    
    print(f"\n{'='*80}")
    print("WEIGHTED RANDOM SAMPLER")
    print(f"{'='*80}")
    print(f"Class counts: {class_counts.tolist()}")
    print(f"Weights per class: {[f'{w:.4f}' for w in weights_per_class]}")
    print(f"Effective samples after weighting: {weights.sum():.0f}")
    print(f"{'='*80}\n")
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def create_session_based_dataloaders(data_path, user_ids, normalization='none', batch_size=8,
                                     augment_train=False, cache_size=100,
                                     test_sessions_per_user=1, val_sessions_per_user=1):
    """
    \Create data loaders with weighted sampling for class balance
    """
    print("\n" + "=" * 80)
    print("Creating FIXED session-based data loaders with class balancing")
    print("=" * 80)

    user_session_data = get_session_based_file_paths_and_labels(data_path, user_ids)

    if not user_session_data:
        raise ValueError("No valid user session data found")

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = create_session_based_splits(
        user_session_data, user_ids, test_sessions_per_user, val_sessions_per_user)

    if not train_paths:
        raise ValueError("No training data found")

    print(f"\nCreating PyTorch datasets...")

    train_dataset = SpectrogramDataset(train_paths, train_labels, normalization,
                                       add_channel_dim=True, augment=augment_train,
                                       cache_size=cache_size)
    val_dataset = SpectrogramDataset(val_paths, val_labels, normalization,
                                     add_channel_dim=True, augment=False,
                                     cache_size=cache_size // 2)
    test_dataset = SpectrogramDataset(test_paths, test_labels, normalization,
                                      add_channel_dim=True, augment=False,
                                      cache_size=cache_size // 2)

    train_sampler = create_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,  # Use sampler instead of shuffle
        num_workers=0, 
        pin_memory=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)

    print("\nValidating data loaders...")
    train_batch = next(iter(train_loader))
    X_sample, y_sample = train_batch
    print(f"Sample batch - Shape: {X_sample.shape}")

    print("=" * 80 + "\n")

    return train_loader, val_loader, test_loader, len(user_ids)
