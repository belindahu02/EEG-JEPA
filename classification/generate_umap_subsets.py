import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import umap
from datetime import datetime
import math

# =============================
# Hardcoded paths and settings
# =============================
EMBEDDINGS_DIR = '/app/data/embeddings_full'
OUTPUT_DIR = '/app/data/umap_full'
LABEL_STRATEGY = 'user'  # options: 'prefix', 'directory', 'user', 'none'
USERS_PER_PLOT = 10
NUM_PLOTS = 10

# Saved cache files
CACHE_EMBEDDINGS = os.path.join(OUTPUT_DIR, "embeddings.npy")
CACHE_FILENAMES = os.path.join(OUTPUT_DIR, "filenames.csv")


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_embeddings(embeddings_dir, output_dir):
    # If cached versions exist, load them instead of re-parsing
    if os.path.exists(CACHE_EMBEDDINGS) and os.path.exists(CACHE_FILENAMES):
        log("Loading cached embeddings and filenames...")
        embeddings = np.load(CACHE_EMBEDDINGS)
        filenames = pd.read_csv(CACHE_FILENAMES)["filename"].tolist()
        log(f"Loaded {len(embeddings)} cached embeddings with shape {embeddings.shape}")
        return embeddings, filenames

    # Otherwise, parse from disk
    embeddings_dir = Path(embeddings_dir)
    embedding_files = list(embeddings_dir.glob("**/*_emb.npy"))
    log(f"Found {len(embedding_files)} embedding files")

    if len(embedding_files) == 0:
        raise ValueError(f"No embedding files found in {embeddings_dir}")

    embeddings, filenames = [], []

    for emb_file in embedding_files:
        try:
            emb = np.load(emb_file)
            if emb.ndim > 2:
                emb = emb.reshape(emb.shape[0], -1) if emb.shape[0] > 1 else emb.flatten()
            elif emb.ndim == 2:
                emb = emb.flatten()

            embeddings.append(emb)
            # keep relative path so we can extract user (S001 etc.)
            filenames.append(str(emb_file.relative_to(embeddings_dir)))

        except Exception as e:
            log(f"Error loading {emb_file}: {e}")
            continue

    if len(embeddings) == 0:
        raise ValueError("No embeddings could be loaded successfully")

    embeddings = np.array(embeddings)
    log(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")

    # Save cache for future runs
    os.makedirs(output_dir, exist_ok=True)
    np.save(CACHE_EMBEDDINGS, embeddings)
    pd.DataFrame({"filename": filenames}).to_csv(CACHE_FILENAMES, index=False)
    log(f"Cached embeddings to {CACHE_EMBEDDINGS}")
    log(f"Cached filenames to {CACHE_FILENAMES}")

    return embeddings, filenames


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
        labels = []
        for fname in filenames:
            parts = Path(fname).parts
            labels.append(parts[0] if len(parts) > 1 else 'root')
        return labels

    elif label_strategy == 'user':
        # top-level folder name, e.g. S001
        return [Path(fname).parts[0] for fname in filenames]

    else:
        return ['data'] * len(filenames)


def standardize_embeddings(embeddings):
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    std = np.where(std == 0, 1, std)
    return (embeddings - mean) / std


def create_user_subsets(labels, filenames, embeddings, num_plots=10, users_per_plot=10):
    """Create subsets of users for plotting"""
    unique_users = sorted(set(labels))
    log(f"Found {len(unique_users)} unique users: {unique_users[:10]}...")
    
    # If we have fewer users than expected, adjust the parameters
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
        
        if len(subset_users) == 0:
            break
            
        # Get all samples for these users
        subset_mask = [label in subset_users for label in labels]
        subset_embeddings = embeddings[subset_mask]
        subset_labels = [labels[j] for j in range(len(labels)) if subset_mask[j]]
        subset_filenames = [filenames[j] for j in range(len(filenames)) if subset_mask[j]]
        
        user_subsets.append({
            'embeddings': subset_embeddings,
            'labels': subset_labels,
            'filenames': subset_filenames,
            'users': subset_users,
            'plot_id': i + 1
        })
        
        log(f"Subset {i+1}: {len(subset_users)} users, {len(subset_embeddings)} samples")
    
    return user_subsets


def plot_umap_subset(subset_data, output_dir, title_prefix="UMAP Visualization of JEPA Embeddings"):
    """Plot UMAP for a subset of users"""
    embeddings = subset_data['embeddings']
    labels = subset_data['labels']
    filenames = subset_data['filenames']
    users = subset_data['users']
    plot_id = subset_data['plot_id']
    
    log(f"Computing UMAP projection for subset {plot_id}...")
    embeddings_scaled = standardize_embeddings(embeddings)

    reducer = umap.UMAP(
        n_neighbors=min(15, len(embeddings) - 1),  # Adjust neighbors if we have few samples
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )

    embedding_2d = reducer.fit_transform(embeddings_scaled)
    log(f"UMAP completed for subset {plot_id}. Embedding shape: {embedding_2d.shape}")

    plt.figure(figsize=(12, 8))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = [l == label for l in labels]
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[i]],
            label=label,
            alpha=0.7,
            s=50
        )

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f"{title_prefix} - Subset {plot_id} (Users: {', '.join(users)})")

    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(output_dir, f'jepa_embeddings_umap_subset_{plot_id:02d}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    log(f"Plot saved to {save_path}")

    plt.close()  # Close plot to avoid display issues in SSH
    
    # Save coordinates for this subset
    coords_df = pd.DataFrame({
        'filename': filenames,
        'label': labels,
        'umap_1': embedding_2d[:, 0],
        'umap_2': embedding_2d[:, 1]
    })
    coords_path = os.path.join(output_dir, f'umap_coordinates_subset_{plot_id:02d}.csv')
    coords_df.to_csv(coords_path, index=False)
    log(f"UMAP coordinates for subset {plot_id} saved to {coords_path}")
    
    return embedding_2d, reducer, coords_df


# =============================
# Main script
# =============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

log("Loading embeddings...")
embeddings, filenames = load_embeddings(EMBEDDINGS_DIR, OUTPUT_DIR)

log(f"Creating labels using strategy: {LABEL_STRATEGY}")
labels = create_labels_from_filenames(filenames, LABEL_STRATEGY)
log(f"Created {len(set(labels))} unique labels")

# Create user subsets
log(f"Creating {NUM_PLOTS} subsets with {USERS_PER_PLOT} users each...")
user_subsets = create_user_subsets(labels, filenames, embeddings, NUM_PLOTS, USERS_PER_PLOT)

# Plot each subset
all_coordinates = []
for subset_data in user_subsets:
    embedding_2d, reducer, coords_df = plot_umap_subset(subset_data, OUTPUT_DIR)
    coords_df['subset_id'] = subset_data['plot_id']
    all_coordinates.append(coords_df)

# Combine all coordinates into one file
if all_coordinates:
    combined_coords = pd.concat(all_coordinates, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, 'umap_coordinates_all_subsets.csv')
    combined_coords.to_csv(combined_path, index=False)
    log(f"Combined coordinates saved to {combined_path}")

log("=== UMAP Subset Visualization Statistics ===")
log(f"Total samples: {len(embeddings)}")
log(f"Embedding dimension: {embeddings.shape[1]}")
log(f"Total unique labels: {len(set(labels))}")
log(f"Number of subsets created: {len(user_subsets)}")
for i, subset in enumerate(user_subsets):
    log(f"Subset {i+1}: {len(subset['users'])} users, {len(subset['embeddings'])} samples")
log(f"Label distribution: {dict(pd.Series(labels).value_counts())}")
