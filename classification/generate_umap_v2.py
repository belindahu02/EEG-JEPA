import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import umap
from datetime import datetime
import math
from sklearn.neighbors import NearestNeighbors

# =============================
# Hardcoded paths and settings
# =============================
EMBEDDINGS_DIR = '/app/data/data/embeddings_musicid_unstructured'
OUTPUT_DIR = '/app/data/data/umap_musicid_unstructured_patched'
LABEL_STRATEGY = 'user'  # options: 'prefix', 'directory', 'user', 'none'
USERS_PER_PLOT = 10
NUM_PLOTS = 2
FRACTION_TO_PLOT = 1/3  # Plot only 1/3 of points per user


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# =============================
# Select spatially-connected patch
# =============================
def select_random_patch(coords, fraction=1/3, random_state=None):
    """
    Select a spatially-connected patch of points.
    
    Args:
        coords: Nx2 array of 2D coordinates
        fraction: Fraction of points to select
        random_state: Random seed for reproducibility
    
    Returns:
        indices: Array of selected point indices
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
        # For each point in the selected set, find its neighbors
        border_points = list(selected)
        
        # Find neighbors of border points
        distances, indices = nbrs.kneighbors(coords[border_points])
        
        # Collect all neighbors that aren't already selected
        new_candidates = []
        for neighbor_list in indices:
            for neighbor_idx in neighbor_list:
                if neighbor_idx in candidates:
                    new_candidates.append(neighbor_idx)
        
        if not new_candidates:
            # If no neighbors found, pick random point from candidates
            new_point = np.random.choice(list(candidates))
        else:
            # Pick one of the neighboring candidates
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
# Load embeddings for subset only
# =============================
def load_subset_embeddings(subset_filenames, embeddings_dir):
    embeddings = []
    for fname in subset_filenames:
        emb_path = Path(embeddings_dir) / fname
        emb = np.load(emb_path)
        if emb.ndim > 2:
            emb = emb.reshape(emb.shape[0], -1) if emb.shape[0] > 1 else emb.flatten()
        elif emb.ndim == 2:
            emb = emb.flatten()
        embeddings.append(emb)
    return np.array(embeddings)


# =============================
# Standardize embeddings
# =============================
def standardize_embeddings(embeddings):
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    std = np.where(std == 0, 1, std)
    return (embeddings - mean) / std


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

    # Load only embeddings needed for this subset
    log(f"Loading embeddings for subset {plot_id}...")
    embeddings = load_subset_embeddings(filenames, EMBEDDINGS_DIR)
    embeddings_scaled = standardize_embeddings(embeddings)

    # Generate UMAP coordinates for ALL points
    log(f"Generating UMAP coordinates for all {len(embeddings)} points...")
    reducer = umap.UMAP(
        n_neighbors=min(15, len(embeddings) - 1),
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42,
        low_memory=True
    )
    embedding_2d = reducer.fit_transform(embeddings_scaled)
    del embeddings, embeddings_scaled  # free memory

    # Save all coordinates
    all_coords_df = pd.DataFrame({
        'filename': filenames,
        'label': labels,
        'umap_1': embedding_2d[:, 0],
        'umap_2': embedding_2d[:, 1]
    })
    all_coords_df.to_csv(coords_path, index=False)
    log(f"All UMAP coordinates saved to {coords_path}")

    # Now select 1/3 of points per user as spatially-connected patches
    log(f"Selecting {FRACTION_TO_PLOT:.1%} of points per user as spatially-connected patches...")
    selected_indices = []
    
    unique_labels = sorted(set(labels))
    for user_label in unique_labels:
        # Get indices for this user
        user_mask = np.array([l == user_label for l in labels])
        user_indices = np.where(user_mask)[0]
        user_coords = embedding_2d[user_indices]
        
        # Select a random patch
        patch_indices = select_random_patch(user_coords, fraction=FRACTION_TO_PLOT, 
                                           random_state=42 + plot_id + hash(user_label) % 1000)
        
        # Map back to global indices
        selected_global = user_indices[patch_indices]
        selected_indices.extend(selected_global)
        
        log(f"  User {user_label}: selected {len(patch_indices)}/{len(user_indices)} points")
    
    selected_indices = np.array(selected_indices)
    
    # Create dataframe with selected points
    selected_coords_df = all_coords_df.iloc[selected_indices].copy()
    selected_coords_df.to_csv(selected_coords_path, index=False)
    log(f"Selected coordinates saved to {selected_coords_path}")
    
    # Plot only the selected points
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
    del reducer

    return embedding_2d, None, selected_coords_df


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
    embedding_2d, reducer, coords_df = plot_umap_subset(subset_data, OUTPUT_DIR)
    coords_df['subset_id'] = subset_data['plot_id']
    all_coordinates.append(coords_df)

if all_coordinates:
    combined_coords = pd.concat(all_coordinates, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, 'umap_coordinates_all_subsets_selected.csv')
    combined_coords.to_csv(combined_path, index=False)
    log(f"Combined selected coordinates saved to {combined_path}")

log("=== UMAP Subset Visualization Statistics ===")
log(f"Total unique labels: {len(set(labels))}")
log(f"Number of subsets created: {len(user_subsets)}")
log(f"Fraction plotted per user: {FRACTION_TO_PLOT:.1%}")
for i, subset in enumerate(user_subsets):
    log(f"Subset {i+1}: {len(subset['users'])} users, {len(subset['filenames'])} total samples")
