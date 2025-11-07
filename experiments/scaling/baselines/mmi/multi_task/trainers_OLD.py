import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import layers
import gc
import os

from backbones import *
from data_loader import *


def compute_cohen_kappa(y_true, y_pred, num_classes):
    """Compute Cohen's Kappa score manually."""
    if len(y_pred.shape) == 2:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true, pred in zip(y_true, y_pred_classes):
        confusion_matrix[int(true), int(pred)] += 1

    n = np.sum(confusion_matrix)
    observed_agreement = np.trace(confusion_matrix) / n

    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    expected_agreement = np.sum(row_sums * col_sums) / (n * n)

    if expected_agreement == 1.0:
        return 1.0

    kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    return kappa


def trainer(num_users, fet_extrct, scen, ft, config=None, strategy=None):
    """
    Train classifier with ALL available data and multi-GPU support
    MEMORY OPTIMIZED to avoid cuDNN allocation failures
    
    Args:
        num_users: Number of users (up to 109)
        fet_extrct: Pre-trained feature extractor
        scen: Scenario number
        ft: Fine-tuning configuration (0-5)
        config: Configuration dict
        strategy: Existing MirroredStrategy to reuse (CRITICAL for pre-trained models)
    """
    if config is None:
        config = {
            'frame_size': 40,
            'max_samples_per_session': None  # USE ALL DATA
        }

    # CRITICAL: Reuse strategy from pre-training or create new one
    if strategy is None:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Created new strategy with {strategy.num_replicas_in_sync} devices")
    else:
        print(f"Reusing existing strategy with {strategy.num_replicas_in_sync} devices")

    ft_dict = {0: 17, 1: 12, 2: 11, 3: 8, 4: 5, 5: 0}
    ft_layers = ft_dict[ft]

    # Set feature extractor trainability
    for i in range(1, ft_layers + 1):
        if i < len(fet_extrct.layers):
            fet_extrct.layers[i].trainable = False

    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTOR CONFIG:")
    print(f"ft parameter: {ft}, freezing first {ft_layers} layers")
    print(f"Feature extractor has {len(fet_extrct.layers)} layers")
    trainable_count = sum([1 for layer in fet_extrct.layers if layer.trainable])
    print(f"Trainable layers: {trainable_count}/{len(fet_extrct.layers)}")
    print(f"{'='*60}\n")

    frame_size = config['frame_size']
    path = "/app/data/1.0.0"

    users = list(range(1, num_users + 1))
    num_classes = num_users

    print(f"\n{'=' * 60}")
    print(f"Training with {num_users} users (ALL DATA)")
    print(f"{'=' * 60}")

    # CRITICAL: Very small batch size per replica to avoid cuDNN issues
    batch_size_per_replica = 2  # 2 per GPU = 4 total
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    # Load ALL data without sample limits
    train_gen, val_gen, test_gen, steps = data_load_with_generators(
        path,
        users=users,
        frame_size=frame_size,
        batch_size=batch_size_per_replica,  # Per replica
        max_samples_per_session=None  # USE ALL DATA
    )

    print(f"Training steps per epoch: {steps['train']}")
    print(f"Validation steps per epoch: {steps['val']}")
    print(f"Test steps per epoch: {steps['test']}")
    print(f"Number of classes: {num_classes}")
    print(f"Batch size per replica: {batch_size_per_replica}")
    print(f"Effective batch size: {batch_size}")

    sample_batch = next(iter(train_gen))
    input_shape = sample_batch[0].shape[1:]
    print(f"Input shape: {input_shape}")

    def create_dataset(generator, steps_per_epoch):
        """Convert generator to tf.data.Dataset with prefetching"""
        dataset = tf.data.Dataset.from_generator(
            lambda: generator,
            output_signature=(
                tf.TensorSpec(shape=(None, frame_size, input_shape[-1]), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        # Critical optimizations for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_dataset(train_gen, steps['train'])
    val_dataset = create_dataset(val_gen, steps['val'])
    test_dataset = create_dataset(test_gen, steps['test'])

    # Build memory-efficient classifier within strategy scope
    with strategy.scope():
        inputs = Input(shape=input_shape)
        x = fet_extrct(inputs, training=False)
        
        # Memory-optimized classifier for 109 users
        # Reduced from original to avoid cuDNN memory issues
        x = Dense(384, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(192, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(96, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.3)(x)
        
        # Output layer - float32 for numerical stability
        outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
        resnettssd = Model(inputs, outputs)

        # Learning rate scaled for number of users and fine-tuning
        # More users = need more capacity to learn, so higher LR
        base_lr = 0.001 if num_users <= 50 else 0.0015
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=base_lr / max(1, ft_layers),
            decay_rate=0.96,
            decay_steps=steps['train']
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        resnettssd.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    resnettssd.summary()

    # Enhanced callbacks for 109-user training
    class MemoryCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 3 == 0:
                gc.collect()
            
            # Log progress for monitoring
            val_acc = logs.get('val_accuracy', 0)
            train_acc = logs.get('accuracy', 0)
            print(f"\nEpoch {epoch + 1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    # More patient early stopping for large-scale problem
    callback_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        restore_best_weights=True, 
        patience=15,  # More patience for 109 users
        mode='max',
        verbose=1
    )
    
    callback_memory = MemoryCallback()
    
    # Learning rate reduction
    callback_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        mode='max',
        verbose=1
    )

    print("\nStarting training with ALL data...")
    print(f"Target: 70%+ accuracy for {num_users} users")
    
    # More epochs for larger number of users
    max_epochs = 150 if num_users >= 80 else 100
    
    history = resnettssd.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=steps['train'],
        validation_steps=steps['val'],
        epochs=max_epochs,
        callbacks=[callback_early, callback_memory, callback_lr],
        verbose=1
    )

    # Evaluate
    print("\nEvaluating on test set...")
    results = resnettssd.evaluate(test_dataset, steps=steps['test'], verbose=1)
    test_acc = results[1]
    
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {test_acc:.4f}")
    status = "✓" if test_acc >= 0.70 else "⚠️" if test_acc >= 0.60 else "✗"
    print(f"{status} Target: 70%+")
    print(f"{'='*60}\n")

    # Calculate kappa score
    print("Computing Cohen's Kappa score...")
    y_true_all = []
    y_pred_all = []

    for step, (batch_x, batch_y) in enumerate(test_dataset):
        if step >= steps['test']:
            break
        y_pred_batch = resnettssd.predict(batch_x, verbose=0)
        y_true_all.extend(batch_y.numpy())
        y_pred_all.append(y_pred_batch)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.vstack(y_pred_all)

    kappa_score = compute_cohen_kappa(y_true_all, y_pred_all, num_classes)
    print(f'Kappa score: {kappa_score:.4f}')

    # Clean up
    del resnettssd, y_true_all, y_pred_all, train_dataset, val_dataset, test_dataset
    gc.collect()

    return test_acc, kappa_score
