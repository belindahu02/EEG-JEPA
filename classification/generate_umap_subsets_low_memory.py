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
EMBEDDINGS_DIR = '/app/data/data/embeddings_musicid_unstructured'
OUTPUT_DIR = '/app/data/data/umap_musicid_unstructured'
LABEL_STRATEGY = 'user'  # options: 'prefix', 'directory', 'user', 'none'
USERS_PER_PLOT = 5
NUM_PLOTS = 4


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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

    if os.path.exists(save_path) and os.path.exists(coords_path):
        log(f"Subset {plot_id} already exists, loading existing coordinates...")
        coords_df = pd.read_csv(coords_path)
        embedding_2d = coords_df[['umap_1', 'umap_2']].values
        return embedding_2d, None, coords_df

    # Load only embeddings needed for this subset
    embeddings = load_subset_embeddings(filenames, EMBEDDINGS_DIR)
    embeddings_scaled = standardize_embeddings(embeddings)

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

    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = np.array([l == label for l in labels])
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                    c=[colors[i]], label=label, alpha=0.7, s=30, rasterized=True)
    plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
    plt.title(f"{title_prefix} - Subset {plot_id}")
    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(); plt.clf()

    # Save coordinates
    coords_df = pd.DataFrame({
        'filename': filenames,
        'label': labels,
        'umap_1': embedding_2d[:, 0],
        'umap_2': embedding_2d[:, 1]
    })
    coords_df.to_csv(coords_path, index=False)
    log(f"UMAP coordinates for subset {plot_id} saved to {coords_path}")
    del reducer

    return embedding_2d, None, coords_df


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
    combined_path = os.path.join(OUTPUT_DIR, 'umap_coordinates_all_subsets.csv')
    combined_coords.to_csv(combined_path, index=False)
    log(f"Combined coordinates saved to {combined_path}")

log("=== UMAP Subset Visualization Statistics ===")
log(f"Total unique labels: {len(set(labels))}")
log(f"Number of subsets created: {len(user_subsets)}")
for i, subset in enumerate(user_subsets):
    log(f"Subset {i+1}: {len(subset['users'])} users, {len(subset['filenames'])} samples")
