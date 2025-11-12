import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import gc

from backbones import *
from data_loader import *


def compute_cohen_kappa(y_true, y_pred, num_classes):
    """Compute Cohen's Kappa score manually"""
    if len(y_pred.shape) == 2:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true, pred in zip(y_true, y_pred_classes):
        confusion_matrix[int(true), int(pred)] += 1

    n = np.sum(confusion_matrix)
    if n == 0:
        return 0.0
    observed_agreement = np.trace(confusion_matrix) / n

    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    expected_agreement = np.sum(row_sums * col_sums) / (n * n)

    if expected_agreement == 1.0:
        return 1.0

    kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    return kappa


def trainer(num_users, fet_extrct, scen, ft, checkpoint_dir=None):
    """
    Train classifier following paper specifications:
    - Classifier: two hidden Dense layers (256, 64) with ReLU
    - Initial learning rate: 0.01 with exponential decay
    - Train up to 30 epochs with early stopping
    """
    strategy = tf.distribute.MirroredStrategy()
    print(f'\nNumber of GPUs: {strategy.num_replicas_in_sync}')
    
    ft_dict = {0: 17, 1: 12, 2: 11, 3: 8, 4: 5, 5: 0}
    ft = ft_dict[ft]

    frame_size = 40
    path = "/app/data/1.0.0"
    batch_size = 64 * strategy.num_replicas_in_sync
    users = list(range(1, num_users + 1))

    print(f"\n{'=' * 60}")
    print(f"Training with {num_users} users")
    print(f"Batch size: {batch_size}")
    print(f"{'=' * 60}")

    print("\nLoading all data into memory...")
    train_ds, val_ds, test_ds, steps = data_load_with_tf_datasets(
        path, users=users, 
        frame_size=frame_size,
        batch_size=batch_size
    )

    print(f"\nDataset steps:")
    print(f"  Train: {steps['train']} steps per epoch")
    print(f"  Val: {steps['val']} steps")
    print(f"  Test: {steps['test']} steps")

    for batch_x, batch_y in train_ds.take(1):
        n_channels = batch_x.shape[-1]
        num_classes = len(users)
        break

    print(f"\nData shape: ({frame_size}, {n_channels})")
    print(f"Number of classes: {num_classes}")

    with strategy.scope():
        fet_extrct_dist = tf.keras.models.clone_model(fet_extrct)
        fet_extrct_dist.set_weights(fet_extrct.get_weights())
        
        # Set trainability
        print(f"\nFine-tuning configuration (ft={ft}):")
        for i, layer in enumerate(fet_extrct_dist.layers):
            if i < ft:
                layer.trainable = False
            else:
                layer.trainable = True
        
        # Classifier with two hidden Dense layers (256, 64)
        inputs = Input(shape=(frame_size, n_channels))
        x = fet_extrct_dist(inputs, training=False)
        
        print(f"\nBackbone output shape: {x.shape}")
        
        # Dense(256, ReLU) → Dense(64, ReLU) → Dense(num_classes, softmax)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = Dense(64, activation='relu', name='fc2')(x)
        outputs = Dense(num_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs, outputs)

        # early stopping
        callback_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            restore_best_weights=True,
            patience=5,
            min_delta=0.0005,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )

        # Initial learning rate 0.01 for classifier with exponential decay
        total_steps = steps['train'] * 30
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=total_steps // 30,
            decay_rate=0.96
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    print("\nClassifier architecture (matching paper):")
    model.summary()
    
    print(f"\nInitial learning rate: 0.01")
    print(f"Training for up to 30 epochs with early stopping")
    
    # train up to 30 epochs
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps['train'],
        validation_steps=steps['val'],
        epochs=30,
        callbacks=[callback_early, reduce_lr],
        verbose=1
    )

    if len(history.history['loss']) > 0:
        print(f"\n{'=' * 60}")
        print("Training completed:")
        print(f"  Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"  Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        print(f"  Total epochs trained: {len(history.history['loss'])}")
        print(f"{'=' * 60}")

    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_ds, steps=steps['test'], verbose=1)
    test_acc = test_results[1]

    print(f"Test accuracy: {test_acc:.4f}")

    print("\nComputing Kappa score...")
    y_pred = model.predict(test_ds, steps=steps['test'], verbose=1)
    
    y_true = []
    for _, labels in test_ds:
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true, axis=0)
    
    kappa_score = compute_cohen_kappa(y_true, y_pred, num_classes)
    print(f'Kappa score: {kappa_score:.4f}')
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("\nPer-class statistics:")
    for class_id in range(min(10, num_classes)):
        mask = y_true == class_id
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == class_id)
            print(f"  Class {class_id}: {np.sum(mask)} samples, accuracy: {class_acc:.3f}")

    del y_pred, y_true, model, history, train_ds, val_ds, test_ds
    gc.collect()
    tf.keras.backend.clear_session()

    return test_acc, kappa_score
