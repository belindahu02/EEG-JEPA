import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
import gc
import os

from backbones import *
from data_loader import *
from transformations import *
from simsiam_paper import *


def pre_trainer(scen, fet, base_dir="/app/data/experiments/scaling/baselines"):
    frame_size = 40
    BATCH_SIZE = 40
    origin = False
    EPOCHS = 30
    path = "/app/data/1.0.0"

    checkpoint_dir = os.path.join(base_dir, "simsiam/checkpoints")
    graphs_dir = os.path.join(base_dir, "simsiam/graphs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    users = list(range(1, 110))
    folder_train = ["TrainingSet"]

    print("Loading pre-training data...")
    max_samples_per_user = 500

    x_train, y_train, sessions_train = data_load_origin(
        path, users=users, folders=folder_train,
        frame_size=frame_size, max_samples_per_user=max_samples_per_user
    )
    print("Training samples:", x_train.shape[0])

    if x_train.shape[0] == 0:
        raise ValueError("No training data loaded. Check path and user folders.")

    x_train = norma_pre(x_train)
    print("x_train normalized:", x_train.shape)

    # use Random Scaling, Magnitude Warping, Time Warping, Negation
    def aug1_numpy(x):
        x = DA_Scaling(x, sigma=1.0)  # Random Scaling
        x = DA_MagWarp(x, sigma=0.3)  # Magnitude Warping
        return x.astype(np.float32)

    @tf.function
    def tf_aug1(input):
        y = tf.numpy_function(aug1_numpy, [input], tf.float32)
        y.set_shape((BATCH_SIZE, input.shape[-1]))
        return y

    def aug2_numpy(x):
        x = DA_TimeWarp(x, sigma=0.2)  # Time Warping
        x = DA_Negation(x)  # Negation
        return x.astype(np.float32)

    @tf.function
    def tf_aug2(input):
        y = tf.numpy_function(aug2_numpy, [input], tf.float32)
        y.set_shape((BATCH_SIZE, input.shape[-1]))
        return y

    AUTO = tf.data.AUTOTUNE
    SEED = 34

    ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_one = (
        ssl_ds_one.shuffle(1024, seed=SEED)
            .map(tf_aug1, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
    )

    ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_two = (
        ssl_ds_two.shuffle(1024, seed=SEED)
            .map(tf_aug2, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
    )

    ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))

    mlp_s = 2048
    num_training_samples = len(x_train)
    steps = EPOCHS * (num_training_samples // BATCH_SIZE)
    
    # Initial learning rate 0.00003 for pre-training with exponential decay
    lr_decayed_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.00003,
        decay_steps=steps // EPOCHS,
        decay_rate=0.96
    )

    # Paper: early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True, min_delta=0.0001
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, f"simsiam_pretrain_scen{scen}_epoch{{epoch:02d}}.weights.h5"),
        save_weights_only=True,
        save_freq='epoch',
        verbose=0
    )

    # Get encoder and backbone separately
    encoder, backbone_model = get_encoder(frame_size, x_train.shape[-1], mlp_s, origin)
    
    print("\nEncoder (with projection head):")
    encoder.summary()
    print("\nBackbone (for downstream tasks):")
    backbone_model.summary()

    contrastive = Contrastive(
        encoder,
        get_predictor(mlp_s, origin),
        backbone=backbone_model
    )
    
    # Adam optimizer with L2 regularization λ=0.01 (already in layers)
    contrastive.compile(optimizer=tf.keras.optimizers.Adam(lr_decayed_fn))

    print(f"\nTraining SimSiam model for {EPOCHS} epochs...")
    print(f"Initial learning rate: 0.00003")
    history = contrastive.fit(ssl_ds, epochs=EPOCHS, callbacks=[early_stopping, checkpoint_callback])

    print(f"\nFinal loss: {history.history['loss'][-1]:.4f}")

    # Clean up
    del x_train, y_train, contrastive, encoder
    tf.keras.backend.clear_session()
    gc.collect()

    print("Pre-training completed! Returning backbone model.")
    return backbone_model
