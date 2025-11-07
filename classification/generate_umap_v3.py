import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import umap
from datetime import datetime
import math
from sklearn.neighbors import NearestNeighbors
import gc

# =============================
# Hardcoded paths and settings
# =============================
EMBEDDINGS_DIR = '/app/data/data/embeddings_musicid_unstructured'
OUTPUT_DIR = '/app/data/data/umap_musicid_unstructured_patched'
LABEL_STRATEGY = 'user'  # options: 'prefix', 'directory', 'user', 'none'
USERS_PER_PLOT = 10
NUM_PLOTS = 2
FRACTION_TO_PLOT = 1/3  # Plot only 1/3 of points per user

# Memory optimization settings
UMAP_FIT_SAMPLES = 10000  # Fit UMAP on this many samples, then transform rest
TRANSFORM_BATCH_SIZE = 2000  # Transform embeddings in batches
USE_FLOAT32 = True  # Use float32 instead of float64 (halves memory)


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# =============================
# Select spatially-connected patch
# =============================
def select_random_patch(coords, fraction=1/3, random_state=None):
    """
    Select a spatially-connected patch of points.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_points = len(coords)
    n_select = max(1, int(n_points * fraction))
    
    if n_select >= n_points:
        return np.arange(n_points)
    
    # Randomly pick a starting point
    start_idx = np.random.randint(0, n_points)
    
    # Build a nearest neighbor tree
    n_neighbors = min(10, n_points)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(coords)
    
    # Grow the patch by adding nearest neighbors
    selected = set([start_idx])
    candidates = set(range(n_points)) - selected
    
    while len(selected) < n_select and candidates:
        border_points = list(selected)
        distances, indices = nbrs.kneighbors(coords[border_points])
        
        new_candidates = []
        for neighbor_list in indices:
            for neighbor_idx in neighbor_list:
                if neighbor_idx in candidates:
                    new_candidates.append(neighbor_idx)
        
        if not new_candidates:
            new_point = np.random.choice(list(candidates))
        else:
            new_point = np.random.choice(new_candidates)
        
        selected.add(new_point)
        candidates.discard(new_point)
    
    return np.array(sorted(list(selected)))


# =============================
# Load filenames only
# =============================
def load_filenames(embeddings_dir):
    embeddings_dir = Path(embeddings_dir)
    embedding_files = list(embeddings_dir.glob("**/*_emb.npy"))
    log(f"Found {len(embedding_files)} embedding files")
    if len(embedding_files) == 0:
        raise ValueError(f"No embedding files found in {embeddings_dir}")
    filenames = [str(f.relative_to(embeddings_dir)) for f in embedding_files]
    return filenames


# =============================
# Create labels from filenames
# =============================
def create_labels_from_filenames(filenames, label_strategy='prefix'):
    if label_strategy == 'none':
        return ['data'] * len(filenames)
    elif label_strategy == 'prefix':
        labels = []
        for fname in filenames:
            base = Path(fname).stem
            if '_' in base:
                prefix = base.split('_')[0]
            else:
                prefix = base[:3]
            labels.append(prefix)
        return labels
    elif label_strategy == 'directory':
        labels = [Path(fname).parts[0] if len(Path(fname).parts) > 1 else 'root' for fname in filenames]
        return labels
    elif label_strategy == 'user':
        return [Path(fname).parts[0] for fname in filenames]
    else:
        return ['data'] * len(filenames)


# =============================
# Create user subsets (filenames only)
# =============================
def create_user_subsets(labels, filenames, num_plots=10, users_per_plot=10):
    unique_users = sorted(set(labels))
    log(f"Found {len(unique_users)} unique users: {unique_users[:10]}...")

    total_users = len(unique_users)
    if total_users < num_plots * users_per_plot:
        log(f"Warning: Only {total_users} users available, adjusting plot parameters")
        users_per_plot = max(1, total_users // num_plots)
        num_plots = math.ceil(total_users / users_per_plot)

    user_subsets = []
    for i in range(num_plots):
        start_idx = i * users_per_plot
        end_idx = min(start_idx + users_per_plot, total_users)
        subset_users = unique_users[start_idx:end_idx]
        if not subset_users:
            break

        subset_filenames = [f for j, f in enumerate(filenames) if labels[j] in subset_users]
        subset_labels = [labels[j] for j, f in enumerate(filenames) if labels[j] in subset_users]

        user_subsets.append({
            'filenames': subset_filenames,
            'labels': subset_labels,
            'users': subset_users,
            'plot_id': i + 1
        })
        log(f"Subset {i+1}: {len(subset_users)} users, {len(subset_filenames)} samples")

    return user_subsets


# =============================
# Load single embedding
# =============================
def load_single_embedding(fname, embeddings_dir):
    """Load a single embedding file."""
    emb_path = Path(embeddings_dir) / fname
    emb = np.load(emb_path)
    if emb.ndim > 2:
        emb = emb.reshape(emb.shape[0], -1) if emb.shape[0] > 1 else emb.flatten()
    elif emb.ndim == 2:
        emb = emb.flatten()
    if USE_FLOAT32:
        emb = emb.astype(np.float32)
    return emb


# =============================
# Compute statistics in one pass
# =============================
def compute_statistics(subset_filenames, embeddings_dir):
    """Compute mean and std in one pass through data."""
    log("Computing statistics (mean/std) in one pass...")
    n = 0
    sum_x = None
    sum_x2 = None
    
    for i, fname in enumerate(subset_filenames):
        emb = load_single_embedding(fname, embeddings_dir)
        
        if sum_x is None:
            sum_x = np.zeros_like(emb, dtype=np.float64)
            sum_x2 = np.zeros_like(emb, dtype=np.float64)
        
        sum_x += emb.astype(np.float64)
        sum_x2 += (emb.astype(np.float64) ** 2)
        n += 1
        
        if (i + 1) % 5000 == 0:
            log(f"  Processed {i+1}/{len(subset_filenames)} files for statistics")
    
    mean = sum_x / n
    variance = (sum_x2 / n) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 0))
    std = np.where(std == 0, 1, std)
    
    if USE_FLOAT32:
        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
    
    log(f"Statistics computed: mean shape={mean.shape}, std shape={std.shape}")
    return mean, std


# =============================
# Standardize embedding using precomputed stats
# =============================
def standardize_embedding(emb, mean, std):
    """Standardize a single embedding."""
    return (emb - mean) / std


# =============================
# Memory-efficient UMAP with fit-transform strategy
# =============================
def compute_umap_memory_efficient(subset_filenames, labels, embeddings_dir, mean, std):
    """
    Fit UMAP on a subset, then transform all data in batches.
    """
    n_total = len(subset_filenames)
    n_fit = min(UMAP_FIT_SAMPLES, n_total)
    
    log(f"Memory-efficient UMAP: fitting on {n_fit} samples, transforming {n_total} total")
    
    # Step 1: Select stratified sample for fitting (ensure all users represented)
    log("Selecting stratified sample for UMAP fitting...")
    unique_labels = sorted(set(labels))
    samples_per_user = max(1, n_fit // len(unique_labels))
    
    fit_indices = []
    for label in unique_labels:
        user_indices = [i for i, l in enumerate(labels) if l == label]
        n_select = min(samples_per_user, len(user_indices))
        selected = np.random.choice(user_indices, n_select, replace=False)
        fit_indices.extend(selected)
    
    # If we need more samples, add random ones
    if len(fit_indices) < n_fit:
        remaining = list(set(range(n_total)) - set(fit_indices))
        n_more = min(n_fit - len(fit_indices), len(remaining))
        fit_indices.extend(np.random.choice(remaining, n_more, replace=False))
    
    fit_indices = fit_indices[:n_fit]
    log(f"Selected {len(fit_indices)} samples for fitting")
    
    # Step 2: Load and standardize fitting data
    log("Loading embeddings for UMAP fitting...")
    fit_embeddings = []
    for idx in fit_indices:
        emb = load_single_embedding(subset_filenames[idx], embeddings_dir)
        emb_std = standardize_embedding(emb, mean, std)
        fit_embeddings.append(emb_std)
    
    fit_embeddings = np.array(fit_embeddings)
    log(f"Loaded {len(fit_embeddings)} embeddings for fitting, shape: {fit_embeddings.shape}")
    
    # Step 3: Fit UMAP
    log("Fitting UMAP...")
    reducer = umap.UMAP(
        n_neighbors=min(15, len(fit_embeddings) - 1),
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42,
        low_memory=True,
        verbose=True
    )
    reducer.fit(fit_embeddings)
    log("UMAP fitting complete")
    
    # Free memory
    del fit_embeddings
    gc.collect()
    
    # Step 4: Transform all data in batches
    log(f"Transforming all {n_total} embeddings in batches of {TRANSFORM_BATCH_SIZE}...")
    all_coords = np.zeros((n_total, 2), dtype=np.float32)
    
    for batch_start in range(0, n_total, TRANSFORM_BATCH_SIZE):
        batch_end = min(batch_start + TRANSFORM_BATCH_SIZE, n_total)
        batch_size = batch_end - batch_start
        
        # Load batch
        batch_embeddings = []
        for idx in range(batch_start, batch_end):
            emb = load_single_embedding(subset_filenames[idx], embeddings_dir)
            emb_std = standardize_embedding(emb, mean, std)
            batch_embeddings.append(emb_std)
        
        batch_embeddings = np.array(batch_embeddings)
        
        # Transform batch
        batch_coords = reducer.transform(batch_embeddings)
        all_coords[batch_start:batch_end] = batch_coords.astype(np.float32)
        
        # Free memory
        del batch_embeddings, batch_coords
        gc.collect()
        
        log(f"  Transformed {batch_end}/{n_total} embeddings")
    
    log("All embeddings transformed")
    return all_coords, reducer


# =============================
# Plot UMAP for a subset
# =============================
def plot_umap_subset(subset_data, output_dir, title_prefix="UMAP Visualization of JEPA Embeddings"):
    labels = subset_data['labels']
    filenames = subset_data['filenames']
    users = subset_data['users']
    plot_id = subset_data['plot_id']

    save_path = os.path.join(output_dir, f'jepa_embeddings_umap_subset_{plot_id:02d}.png')
    coords_path = os.path.join(output_dir, f'umap_coordinates_subset_{plot_id:02d}.csv')
    selected_coords_path = os.path.join(output_dir, f'umap_coordinates_selected_subset_{plot_id:02d}.csv')

    log(f"\n{'='*60}")
    log(f"Processing subset {plot_id}: {len(filenames)} samples from {len(users)} users")
    log(f"{'='*60}")
    
    # Step 1: Compute statistics
    mean, std = compute_statistics(filenames, EMBEDDINGS_DIR)
    
    # Step 2: Memory-efficient UMAP
    embedding_2d, reducer = compute_umap_memory_efficient(filenames, labels, EMBEDDINGS_DIR, mean, std)
    
    # Free memory
    del mean, std
    gc.collect()
    
    # Step 3: Save all coordinates
    log("Saving all UMAP coordinates...")
    all_coords_df = pd.DataFrame({
        'filename': filenames,
        'label': labels,
        'umap_1': embedding_2d[:, 0],
        'umap_2': embedding_2d[:, 1]
    })
    all_coords_df.to_csv(coords_path, index=False)
    log(f"All UMAP coordinates saved to {coords_path}")

    # Step 4: Select patches per user
    log(f"Selecting {FRACTION_TO_PLOT:.1%} of points per user as spatially-connected patches...")
    selected_indices = []
    
    unique_labels = sorted(set(labels))
    for user_label in unique_labels:
        user_mask = np.array([l == user_label for l in labels])
        user_indices = np.where(user_mask)[0]
        user_coords = embedding_2d[user_indices]
        
        patch_indices = select_random_patch(user_coords, fraction=FRACTION_TO_PLOT, 
                                           random_state=42 + plot_id + hash(user_label) % 1000)
        
        selected_global = user_indices[patch_indices]
        selected_indices.extend(selected_global)
        
        log(f"  User {user_label}: selected {len(patch_indices)}/{len(user_indices)} points")
    
    selected_indices = np.array(selected_indices)
    
    # Step 5: Save selected coordinates
    selected_coords_df = all_coords_df.iloc[selected_indices].copy()
    selected_coords_df.to_csv(selected_coords_path, index=False)
    log(f"Selected coordinates saved to {selected_coords_path}")
    
    # Step 6: Plot
    log(f"Plotting {len(selected_indices)} selected points...")
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = selected_coords_df['label'] == label
        plot_coords = selected_coords_df[mask]
        plt.scatter(plot_coords['umap_1'], plot_coords['umap_2'],
                    c=[colors[i]], label=label, alpha=0.7, s=30, rasterized=True)
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f"{title_prefix} - Subset {plot_id} (1/3 samples per user)")
    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.clf()
    
    log(f"Plot saved to {save_path}")
    
    # Free memory
    del reducer, embedding_2d
    gc.collect()

    return selected_coords_df


# =============================
# Main
# =============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

log("Scanning filenames...")
filenames = load_filenames(EMBEDDINGS_DIR)

log(f"Creating labels using strategy: {LABEL_STRATEGY}")
labels = create_labels_from_filenames(filenames, LABEL_STRATEGY)
log(f"Created {len(set(labels))} unique labels")

log(f"Creating {NUM_PLOTS} subsets with {USERS_PER_PLOT} users each...")
user_subsets = create_user_subsets(labels, filenames, NUM_PLOTS, USERS_PER_PLOT)

all_coordinates = []
for subset_data in user_subsets:
    coords_df = plot_umap_subset(subset_data, OUTPUT_DIR)
    coords_df['subset_id'] = subset_data['plot_id']
    all_coordinates.append(coords_df)

if all_coordinates:
    combined_coords = pd.concat(all_coordinates, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, 'umap_coordinates_all_subsets_selected.csv')
    combined_coords.to_csv(combined_path, index=False)
    log(f"Combined selected coordinates saved to {combined_path}")

log("\n" + "="*60)
log("=== UMAP Subset Visualization Statistics ===")
log("="*60)
log(f"Total unique labels: {len(set(labels))}")
log(f"Number of subsets created: {len(user_subsets)}")
log(f"Fraction plotted per user: {FRACTION_TO_PLOT:.1%}")
for i, subset in enumerate(user_subsets):
    log(f"Subset {i+1}: {len(subset['users'])} users, {len(subset['filenames'])} total samples")
