
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import layers
import gc
import os

# Disable XLA at the start of this module
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
tf.config.optimizer.set_jit(False)

from backbones import *
from data_loader import *

base_dir = "/app/data/experiments/baselines"

# Make sure these exist
graph_data_dir = os.path.join(base_dir, "multi_task/graph_data")
graphs_dir = os.path.join(base_dir, "multi_task/graphs")

os.makedirs(graph_data_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)


class AugmentedDataSequence(tf.keras.utils.Sequence):
    """Keras Sequence for on-the-fly data augmentation"""

    def __init__(self, base_generator, transformations, sigma_l, ext=False,
                 batches_per_epoch=100):
        self.base_generator = base_generator
        self.transformations = transformations
        self.sigma_l = sigma_l
        self.ext = ext
        self.n_transforms = len(transformations)
        self.batches_per_epoch = batches_per_epoch
        self.cached_batches = []
        self.cache_epoch_data()

    def cache_epoch_data(self):
        self.cached_batches = []
        batch_count = 0
        for batch_x, batch_y in self.base_generator:
            self.cached_batches.append((batch_x, batch_y))
            batch_count += 1
            if batch_count >= self.batches_per_epoch:
                break

    def __len__(self):
        return len(self.cached_batches) * self.n_transforms

    def __getitem__(self, idx):
        base_batch_idx = idx // self.n_transforms
        transform_idx = idx % self.n_transforms
        batch_x, batch_y = self.cached_batches[base_batch_idx]
        batch_size = len(batch_x)
        half = batch_size // 2
        
        transform = self.transformations[transform_idx]
        sigma = self.sigma_l[transform_idx]
        
        augmented = np.array([transform(x, sigma=sigma) for x in batch_x[:half]], dtype=np.float32)
        original = np.array(batch_x[half:], dtype=np.float32)
        
        combined_data = np.concatenate([augmented, original], axis=0)
        combined_labels = np.concatenate([
            np.ones(half, dtype=np.float32),
            np.zeros(batch_size - half, dtype=np.float32)
        ])
        
        shuffle_idx = np.random.permutation(batch_size)
        combined_data = combined_data[shuffle_idx]
        combined_labels = combined_labels[shuffle_idx]
        
        x_dict = {}
        y_dict = {}
        sample_weight_dict = {}
        
        for i in range(self.n_transforms):
            if i == transform_idx:
                x_dict[f"input_{i + 1}"] = combined_data
                y_dict[f"head_{i + 1}"] = combined_labels
                sample_weight_dict[f"head_{i + 1}"] = np.ones(batch_size, dtype=np.float32)
            else:
                x_dict[f"input_{i + 1}"] = np.zeros_like(combined_data)
                y_dict[f"head_{i + 1}"] = np.zeros(batch_size, dtype=np.float32)
                sample_weight_dict[f"head_{i + 1}"] = np.zeros(batch_size, dtype=np.float32)

        return x_dict, y_dict, sample_weight_dict

    def on_epoch_end(self):
        self.cache_epoch_data()
        gc.collect()


def pre_trainer(scen):
    """Pre-train feature extractor with multi-GPU support and no sample limits"""
    
    # Create multi-GPU strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    
    frame_size = 40
    path = "/app/data/1.0.0"
    
    # CRITICAL: Very small batch size to avoid cuDNN memory issues
    batch_size_per_replica = 4  # 2 per GPU = 4 total on 2 GPUs
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    users = list(range(1, 110))
    train_sessions = list(range(1, 15))

    print(f"\n{'='*60}")
    print(f"PRE-TRAINING WITH ALL DATA (NO SAMPLE LIMITS)")
    print(f"Users: {len(users)}, Sessions: {train_sessions}")
    print(f"Batch size per replica: {batch_size_per_replica}")
    print(f"Effective batch size: {batch_size}")
    print(f"{'='*60}\n")

    # Compute normalization WITHOUT sample limits
    print("Computing normalization statistics from ALL training data...")
    mean, std = compute_normalization_stats(
        path, users=users, sessions=train_sessions,
        frame_size=frame_size, max_samples_per_session=None  # USE ALL DATA
    )

    print("Creating data generator with NO sample limits...")
    base_generator = EEGDataGenerator(
        path, users, train_sessions, frame_size,
        batch_size=batch_size_per_replica,  # Batch per replica
        max_samples_per_session=None,  # USE ALL DATA
        mean=mean, std=std, shuffle=True
    )

    steps_per_epoch = base_generator.get_steps_per_epoch()
    batches_per_epoch = min(steps_per_epoch, 300)  # More batches since we have all data
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Using {batches_per_epoch} batches per epoch")

    first_batch_x, first_batch_y = next(iter(base_generator))
    n_channels = first_batch_x.shape[-1]
    print(f"Data shape: (batch_size={batch_size_per_replica}, {frame_size}, {n_channels})")

    # REQUESTED AUGMENTATIONS: Scaling, MagWarp, TimeWarp, Negation
    transformations = [
        DA_Scaling,      # Random Scaling
        DA_MagWarp,      # Magnitude Warping
        DA_TimeWarp,     # Time Warping
        DA_Negation      # Negation
    ]
    sigma_l = [1.0, 1.5, 1.0, None]  # DOUBLED - was [0.5, 0.5, 0.2, None]
#    sigma_l = [0.5, 0.5, 0.2, None]  # Optimized sigma values

    print(f"\nAugmentations: {[t.__name__ for t in transformations]}")
    print(f"Sigma values: {sigma_l}")

    # Optimized architecture for memory efficiency
    con = 3  # Reduced from 4 to save memory
    ks = 3

    def trunk():
        """
        Memory-optimized trunk for 109-user classification
        REDUCED capacity to avoid cuDNN memory issues
        """
        input_ = Input(shape=(frame_size, n_channels), name='input_')
        
        # Initial conv with SMALLER pooling
        x = Conv1D(filters=20 * con, kernel_size=ks, strides=1, padding='same')(input_)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling1D(pool_size=2, strides=2)(x)
        x = Dropout(rate=0.2)(x)
        
        # ResNet Block 1
        x = resnetblock(x, CR=80 * con, KS=ks)
        x = Dropout(rate=0.2)(x)
        
        # Final ResNet Block with GlobalMaxPooling
        x = resnetblock_final(x, CR=160 * con, KS=ks)
        
        return tf.keras.models.Model(input_, x, name='trunk_')

    # Build model within strategy scope for multi-GPU
    with strategy.scope():
        inputs = []
        for i in range(len(transformations)):
            inputs.append(Input(shape=(frame_size, n_channels), name=f'input_{i+1}'))

        trunk = trunk()
        trunk.summary()

        fets = [trunk(inp) for inp in inputs]

        # SIMPLER classification heads to reduce memory
        heads = []
        for i, fet in enumerate(fets):
            head_name = 'head_' + str(i + 1)
            
            # Simplified: 128 → 1 (removed one layer)
            x = Dense(128, activation='relu', 
                     kernel_regularizer=tf.keras.regularizers.l2(0.001))(fet)
            x = Dropout(0.5)(x)
            head = Dense(1, activation='sigmoid', name=head_name, dtype='float32')(x)
            heads.append(head)

        model = tf.keras.models.Model(inputs, heads, name='multi-task_self-supervised')

        # Compile
        loss = ['binary_crossentropy'] * len(transformations)
        loss_weights = [1 / len(transformations)] * len(transformations)
        metrics = [[tf.keras.metrics.BinaryAccuracy(name=f'binary_accuracy')] 
                   for _ in transformations]

        # Lower learning rate for stability with larger dataset
        opt = tf.keras.optimizers.Adam(learning_rate=0.0003)
        
        model.compile(
            loss=loss,
            loss_weights=loss_weights,
            optimizer=opt,
            metrics=metrics,
            run_eagerly=False
        )

    model.summary()

    class Logger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            acc = []
            for i in range(len(transformations)):
                acc.append(logs.get('head_' + str(i + 1) + '_binary_accuracy'))
            print('=' * 30, epoch + 1, '=' * 30)
            print('Head accuracies:', [f'{a:.4f}' if a is not None else 'N/A' for a in acc])
            print(f'Total loss: {logs.get("loss", 0):.4f}')
            
            avg_acc = np.mean([a for a in acc if a is not None])
            if epoch > 15 and avg_acc < 0.70:
                print(f"⚠️  WARNING: Average accuracy {avg_acc:.4f} is low after {epoch+1} epochs")

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0.003, patience=20, restore_best_weights=True
    )
    
    callback_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1
    )

    print("\nPreparing augmented data sequence...")
    aug_sequence = AugmentedDataSequence(
        base_generator, transformations, sigma_l,
        ext=False, batches_per_epoch=batches_per_epoch
    )

    print(f"Cached batches: {len(aug_sequence.cached_batches)}")
    print(f"Total batches per epoch: {len(aug_sequence)}")

    print("\nStarting pre-training with enhanced architecture and ALL data...")
    
    history = model.fit(
        aug_sequence,
        epochs=60,
        callbacks=[Logger(), callback, callback_lr],
        verbose=1
    )

    # Check final accuracies
    final_accs = [history.history[f'head_{i+1}_binary_accuracy'][-1] 
                  for i in range(len(transformations))]
    avg_acc = np.mean(final_accs)
    print(f"\n{'='*60}")
    print(f"FINAL PRE-TRAINING ACCURACIES:")
    for i, (transform, acc) in enumerate(zip(transformations, final_accs)):
        status = "✓" if acc > 0.75 else "⚠️" if acc > 0.65 else "✗"
        print(f"  {status} {transform.__name__}: {acc:.4f}")
    print(f"\nAverage accuracy: {avg_acc:.4f}")
    if avg_acc < 0.70:
        print("⚠️  Pre-training accuracy is moderate")
    else:
        print("✓ Pre-training successful!")
    print(f"{'='*60}\n")

    fet_extrct = model.layers[len(transformations)]
    print(f"Extracted feature extractor: {fet_extrct.name}")

    del model, aug_sequence, base_generator
    gc.collect()

    # Visualize latent space
    print("\nGenerating latent space visualization...")
    viz_generator = EEGDataGenerator(
        path, users[:20], train_sessions[:3], frame_size,
        batch_size=32,
        max_samples_per_session=2000,
        mean=mean, std=std, shuffle=False
    )

    x_train_viz = []
    y_train_viz = []
    samples_collected = 0
    max_samples = 1000

    for batch_x, batch_y in viz_generator:
        x_train_viz.append(batch_x)
        y_train_viz.append(batch_y)
        samples_collected += len(batch_x)
        if samples_collected >= max_samples:
            break

    x_train_viz = np.concatenate(x_train_viz, axis=0)[:max_samples]
    y_train_viz = np.concatenate(y_train_viz, axis=0)[:max_samples]

    enc_results = fet_extrct.predict(x_train_viz, batch_size=32, verbose=0)
    enc_results = np.array(enc_results)

    if len(enc_results.shape) > 2:
        enc_results = enc_results.reshape(enc_results.shape[0], -1)

    enc_centered = enc_results - np.mean(enc_results, axis=0)
    cov_matrix = np.cov(enc_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    X_embedded = enc_centered @ eigenvectors[:, :2].real

    fig4 = plt.figure(figsize=(18, 12))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train_viz, alpha=0.6, cmap='tab20')
    plt.title(f'Latent Space (Enhanced Multi-GPU) - Avg Acc: {avg_acc:.3f}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(label='User ID')
    plt.savefig(os.path.join(graphs_dir, 'latentspace_scen_' + str(scen) + '.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig4)

    del x_train_viz, y_train_viz, enc_results, viz_generator
    gc.collect()

    print("\nPre-training complete!")
    
    # CRITICAL: Return BOTH the feature extractor AND the strategy
    return fet_extrct, strategy
