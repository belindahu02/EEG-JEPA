from plot_results import *
import tensorflow as tf
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"✓ Enabled memory growth for {len(gpus)} GPU(s)")
        
        # Enable mixed precision for memory efficiency
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"✓ Enabled mixed precision training (float16)")
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected, using CPU")

for layers in [0]:
    print(f"\n{'='*60}")
    print(f"Running experiments for ft={layers}")
    print(f"{'='*60}\n")
    
    try:
        plotspu(layers)
        print(f"\n✓ Successfully completed ft={layers}")
    except Exception as e:
        print(f"\n✗ Error during ft={layers}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tf.keras.backend.clear_session()
        gc.collect()

print(f"\n{'='*60}")
print("ALL EXPERIMENTS COMPLETE")
print(f"{'='*60}\n")
