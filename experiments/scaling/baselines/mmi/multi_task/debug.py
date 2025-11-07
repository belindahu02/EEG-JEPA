"""
Diagnostic script - SIMPLE & REGULARIZED configuration
"""
import numpy as np
import matplotlib.pyplot as plt
from data_loader import *
from transformations import *
from pre_trainers import AugmentedDataSequence
import tensorflow as tf
from backbones import resnetblock_final, resnetblock
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout, Dense
from tensorflow.keras import Input
import gc


def quick_pretraining_test():
    """Quick test with SIMPLE & REGULARIZED config"""
    print("=" * 60)
    print("QUICK TEST: SIMPLE & REGULARIZED CONFIGURATION")
    print("=" * 60)
    
    frame_size = 40
    path = "/app/data/1.0.0"
    batch_size = 8
    users = list(range(1, 11))
    train_sessions = [1, 2]
    
    print(f"\nUsing {len(users)} users, sessions {train_sessions}")
    
    mean, std = compute_normalization_stats(
        path, users=users, sessions=train_sessions,
        frame_size=frame_size, max_samples_per_session=5000
    )
    
    base_generator = EEGDataGenerator(
        path, users, train_sessions, frame_size,
        batch_size=batch_size,
        max_samples_per_session=5000,
        mean=mean, std=std, shuffle=True
    )
    
    first_batch_x, _ = next(iter(base_generator))
    n_channels = first_batch_x.shape[-1]
    print(f"Data shape: ({batch_size}, {frame_size}, {n_channels})")
    
    transformations = [
        DA_Jitter, DA_Scaling, DA_MagWarp,
        DA_RandSampling, DA_Flip, DA_Drop
    ]
    sigma_l = [0.25, 0.4, 0.35, None, None, 6]  # Slightly stronger
    
    aug_sequence = AugmentedDataSequence(
        base_generator, transformations, sigma_l,
        ext=False, batches_per_epoch=10
    )
    
    print(f"Cached batches: {len(aug_sequence.cached_batches)}")
    
    # SIMPLE architecture: con=3, 2 ResNet blocks
    con = 3
    ks = 3
    
    def trunk():
        """SIMPLE trunk with con=3"""
        input_ = Input(shape=(frame_size, n_channels), name='input_')
        
        x = Conv1D(filters=16 * con, kernel_size=ks, strides=1, padding='same')(input_)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling1D(pool_size=4, strides=4)(x)
        x = Dropout(rate=0.2)(x)
        
        x = resnetblock(x, CR=48 * con, KS=ks)
        x = Dropout(rate=0.2)(x)
        
        x = resnetblock_final(x, CR=96 * con, KS=ks)
        
        return tf.keras.models.Model(input_, x, name='trunk_')
    
    inputs = [Input(shape=(frame_size, n_channels), name=f'input_{i+1}') 
              for i in range(len(transformations))]
    
    trunk_model = trunk()
    fets = [trunk_model(inp) for inp in inputs]
    
    # SIMPLE HEADS: Just 128 → 1 with heavy dropout
    heads = []
    for i, fet in enumerate(fets):
        x = Dense(128, activation='relu')(fet)
        x = Dropout(0.5)(x)  # Very high dropout
        head = Dense(1, activation='sigmoid', name=f'head_{i+1}')(x)
        heads.append(head)
    
    model = tf.keras.models.Model(inputs, heads, name='test_model')
    
    # LOW learning rate
    model.compile(
        loss=['binary_crossentropy'] * len(transformations),
        loss_weights=[1/len(transformations)] * len(transformations),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        metrics=[[tf.keras.metrics.BinaryAccuracy()] for _ in transformations]
    )
    
    print("\nTraining for 10 epochs with SIMPLE & REGULARIZED config...")
    print("Goal: Steady improvement, reaching 65-75% by epoch 10")
    print("Key: Accuracies should INCREASE (not decrease like before)\n")
    
    history = model.fit(
        aug_sequence,
        epochs=10,
        verbose=1
    )
    
    # Analysis
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    print("\nLoss trajectory:")
    for ep in [0, 4, 9]:
        print(f"  Epoch {ep+1}: {history.history['loss'][ep]:.4f}")
    
    print("\nHead accuracy trajectories:")
    print("  Head | Epoch 1 | Epoch 5 | Epoch 10 | Change")
    print("  " + "-"*50)
    
    improvements = []
    for i in range(len(transformations)):
        acc_ep1 = history.history[f'head_{i+1}_binary_accuracy'][0]
        acc_ep5 = history.history[f'head_{i+1}_binary_accuracy'][4]
        acc_ep10 = history.history[f'head_{i+1}_binary_accuracy'][-1]
        change = acc_ep10 - acc_ep1
        improvements.append(change)
        arrow = "✓" if change > 0.05 else "→" if change > 0 else "✗"
        print(f"  {i+1}    | {acc_ep1:.4f}  | {acc_ep5:.4f}  | {acc_ep10:.4f}   | {arrow} {change:+.4f}")
    
    avg_improvement = np.mean(improvements)
    
    print("\nFinal accuracies (Epoch 10):")
    final_accs = []
    for i in range(len(transformations)):
        acc = history.history[f'head_{i+1}_binary_accuracy'][-1]
        final_accs.append(acc)
        status = "✓" if acc > 0.70 else "⚠️" if acc > 0.60 else "✗"
        print(f"  {status} head_{i+1}: {acc:.4f}")
    
    avg_acc = np.mean(final_accs)
    print(f"\nAverage: {avg_acc:.4f}")
    print(f"Average improvement: {avg_improvement:+.4f}")
    
    # Verdict
    print("\n" + "="*60)
    if avg_improvement > 0.08 and avg_acc > 0.65:
        print("✓✓ EXCELLENT! Steady improvement detected!")
        print("   Accuracies improving steadily, no overfitting")
        print("   → READY FOR FULL 50-EPOCH TRAINING!")
        success = True
    elif avg_improvement > 0.05:
        print("✓ GOOD! Slow but steady improvement")
        print("   Should reach 70-75% with 50 epochs")
        print("   → Worth trying full training")
        success = True
    elif avg_improvement > 0:
        print("⚠️  MARGINAL improvement")
        print("   May reach 60-70% with 50 epochs")
        print("   → Results may be moderate")
        success = False
    else:
        print("✗ STILL DECREASING - fundamental issue remains")
        print("   → Need different approach")
        success = False
    print("="*60)
    
    del model, aug_sequence, base_generator
    tf.keras.backend.clear_session()
    gc.collect()
    
    return success


if __name__ == "__main__":
    quick_pretraining_test()
