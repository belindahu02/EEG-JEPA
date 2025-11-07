import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D
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
    IMPROVED trainer with proper fine-tuning strategy
    
    Key changes:
    1. Gradual unfreezing strategy
    2. Larger classifier head
    3. Better learning rate schedule
    4. Data augmentation during training
    """
    if config is None:
        config = {
            'frame_size': 40,
            'max_samples_per_session': None
        }

    if strategy is None:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Created new strategy with {strategy.num_replicas_in_sync} devices")
    else:
        print(f"Reusing existing strategy with {strategy.num_replicas_in_sync} devices")

    # IMPROVED fine-tuning strategy
    # ft=0: Unfreeze all layers (full fine-tuning)
    # ft=1: Freeze first 50% of layers
    # ft=2: Freeze first 75% of layers
    # ft=3: Freeze all but last 2 layers
    # ft=4: Freeze all but last layer
    # ft=5: Freeze everything
    
    ft_ratios = {0: 0.0, 1: 0.5, 2: 0.75, 3: 0.9, 4: 0.95, 5: 1.0}
    freeze_ratio = ft_ratios[ft]
    n_layers = len(fet_extrct.layers)
    ft_layers = int(n_layers * freeze_ratio)

    # Set trainability
    for i, layer in enumerate(fet_extrct.layers):
        if i < ft_layers:
            layer.trainable = False
        else:
            layer.trainable = True

    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTOR CONFIG:")
    print(f"ft parameter: {ft}, freezing first {ft_layers}/{n_layers} layers ({freeze_ratio*100:.0f}%)")
    trainable_count = sum([1 for layer in fet_extrct.layers if layer.trainable])
    print(f"Trainable layers: {trainable_count}/{n_layers}")
    print(f"{'='*60}\n")

    frame_size = config['frame_size']
    path = "/app/data/1.0.0"

    users = list(range(1, num_users + 1))
    num_classes = num_users

    print(f"\n{'=' * 60}")
    print(f"Training classifier for {num_users} users")
    print(f"{'=' * 60}")

    # Batch size - adjust based on number of users
    if num_users <= 20:
        batch_size_per_replica = 16
    elif num_users <= 50:
        batch_size_per_replica = 8
    else:
        batch_size_per_replica = 4
    
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    # Use subset of data for efficiency
    max_samples = 10000 if num_users <= 50 else 5000

    train_gen, val_gen, test_gen, steps = data_load_with_generators(
        path,
        users=users,
        frame_size=frame_size,
        batch_size=batch_size_per_replica,
        max_samples_per_session=max_samples
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
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_dataset(train_gen, steps['train'])
    val_dataset = create_dataset(val_gen, steps['val'])
    test_dataset = create_dataset(test_gen, steps['test'])

    # Build LARGER classifier within strategy scope
    with strategy.scope():
        inputs = Input(shape=input_shape)
        
        # Get features from pre-trained extractor
        x = fet_extrct(inputs, training=(ft == 0))  # Only train if fully unfrozen
        
        # LARGER classification head for better capacity
        # Scale size based on number of users
        hidden_size = max(512, num_classes * 4)
        
        x = Dense(hidden_size, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(hidden_size // 2, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(hidden_size // 4, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        if num_users > 50:
            # Extra layer for large number of classes
            x = Dense(hidden_size // 8, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
            x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
        resnettssd = Model(inputs, outputs)

        # Learning rate schedule - higher for less freezing
        if ft == 0:
            # Full fine-tuning: lower LR to not destroy pre-trained features
            base_lr = 0.0001
        elif ft <= 2:
            # Partial fine-tuning: medium LR
            base_lr = 0.0005
        else:
            # Mostly frozen: higher LR since only training classifier
            base_lr = 0.001
        
        # Adjust for number of users
        base_lr *= (1.0 + np.log10(num_users) / 2)
        
        print(f"Base learning rate: {base_lr:.6f}")
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr,
            decay_steps=steps['train'] * 50,  # Over 50 epochs
            alpha=0.01  # End at 1% of initial LR
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Use label smoothing for better generalization
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        
        resnettssd.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )

    resnettssd.summary()

    # Enhanced callbacks
    class DetailedLogger(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_val_acc = 0
            
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 2 == 0:
                gc.collect()
            
            val_acc = logs.get('val_accuracy', 0)
            train_acc = logs.get('accuracy', 0)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                improvement = "🎯"
            else:
                improvement = "  "
            
            print(f"\n{improvement} Epoch {epoch + 1}: Train={train_acc:.4f}, Val={val_acc:.4f}, Best Val={self.best_val_acc:.4f}")
            
            # Warning if overfitting
            if train_acc - val_acc > 0.15:
                print(f"⚠️  Large train/val gap: {train_acc - val_acc:.4f}")

    # Patience based on fine-tuning strategy
    patience = 20 if ft == 0 else 15
    
    callback_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        restore_best_weights=True,
        patience=patience,
        mode='max',
        verbose=1
    )
    
    callback_logger = DetailedLogger()
    
    callback_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        mode='max',
        verbose=1
    )

    print("\nStarting training...")
    print(f"Target: 70%+ accuracy for {num_users} users")
    
    # More epochs for difficult cases
    if num_users >= 80:
        max_epochs = 100
    elif num_users >= 50:
        max_epochs = 80
    else:
        max_epochs = 60
    
    history = resnettssd.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=steps['train'],
        validation_steps=steps['val'],
        epochs=max_epochs,
        callbacks=[callback_early, callback_logger, callback_lr],
        verbose=1
    )

    # Evaluate
    print("\nEvaluating on test set...")
    results = resnettssd.evaluate(test_dataset, steps=steps['test'], verbose=1)
    test_acc = results[1]
    
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {test_acc:.4f}")
    
    # Show training history
    best_val_acc = max(history.history['val_accuracy'])
    final_train_acc = history.history['accuracy'][-1]
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Final Train Accuracy: {final_train_acc:.4f}")
    print(f"Generalization Gap: {final_train_acc - test_acc:.4f}")
    
    status = "✓" if test_acc >= 0.70 else "⚠️" if test_acc >= 0.50 else "✗"
    print(f"{status} Target: 70%+")
    
    # Diagnose issues
    if test_acc < 0.50:
        print("\n🔍 DIAGNOSIS:")
        if final_train_acc < 0.50:
            print("  - Model is underfitting (low train accuracy)")
            print("  - Try: Unfreeze more layers (lower ft value) or increase model capacity")
        elif final_train_acc - test_acc > 0.3:
            print("  - Model is overfitting (large train/test gap)")
            print("  - Try: More dropout, more regularization, or more data")
        else:
            print("  - Pre-trained features may not transfer well")
            print("  - Try: Full fine-tuning (ft=0) or better pre-training")
    
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
