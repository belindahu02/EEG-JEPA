from plot_results import *
import tensorflow as tf
import os

# Configure BOTH GPUs with memory growth and multi-GPU strategy
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth on ALL GPUs
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

# Run experiments for different fine-tuning configurations
# Start with ft=0 (full fine-tuning)
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
        # Aggressive cleanup
        tf.keras.backend.clear_session()
        gc.collect()

print(f"\n{'='*60}")
print("ALL EXPERIMENTS COMPLETE")
print(f"{'='*60}\n")
