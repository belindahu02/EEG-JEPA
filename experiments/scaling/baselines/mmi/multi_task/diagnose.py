#!/usr/bin/env python3
"""
Diagnose cuDNN installation and TensorFlow configuration
"""

import sys
import subprocess

print("="*70)
print("CUDNN AND TENSORFLOW DIAGNOSTIC")
print("="*70)

# Check TensorFlow installation
print("\n1. Checking TensorFlow installation...")
try:
    import tensorflow as tf
    print(f"   ✓ TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"   ✗ TensorFlow not installed: {e}")
    sys.exit(1)

# Check GPU availability
print("\n2. Checking GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   ✓ Found {len(gpus)} GPU(s)")
    for i, gpu in enumerate(gpus):
        print(f"     - GPU {i}: {gpu.name}")
else:
    print("   ✗ No GPUs detected")
    sys.exit(1)

# Check CUDA
print("\n3. Checking CUDA...")
try:
    cuda_version = tf.sysconfig.get_build_info()['cuda_version']
    print(f"   ✓ TensorFlow built with CUDA: {cuda_version}")
except:
    print("   ✗ CUDA information not available")

# Check cuDNN
print("\n4. Checking cuDNN...")
try:
    cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
    print(f"   ✓ TensorFlow built with cuDNN: {cudnn_version}")
except:
    print("   ✗ cuDNN information not available")

# Test cuDNN availability
print("\n5. Testing cuDNN operations...")
try:
    # Simple test of Conv1D which requires cuDNN
    from tensorflow.keras.layers import Conv1D
    import numpy as np
    
    # Create a simple Conv1D layer
    conv = Conv1D(filters=32, kernel_size=3, padding='same')
    test_input = tf.constant(np.random.randn(1, 10, 64), dtype=tf.float32)
    
    # Try to run it
    output = conv(test_input)
    print(f"   ✓ cuDNN operations work! Output shape: {output.shape}")
    
except Exception as e:
    print(f"   ✗ cuDNN operations failed: {e}")
    print("\n   This indicates cuDNN is not properly installed or configured.")

# Check library paths
print("\n6. Checking library paths...")
try:
    result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
    cudnn_libs = [line for line in result.stdout.split('\n') if 'cudnn' in line.lower()]
    
    if cudnn_libs:
        print("   ✓ cuDNN libraries found:")
        for lib in cudnn_libs[:5]:  # Show first 5
            print(f"     {lib.strip()}")
    else:
        print("   ✗ No cuDNN libraries found in ldconfig")
        
except Exception as e:
    print(f"   ⚠️  Could not check library paths: {e}")

# Check LD_LIBRARY_PATH
print("\n7. Checking LD_LIBRARY_PATH...")
import os
ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if ld_path:
    print(f"   LD_LIBRARY_PATH is set:")
    for path in ld_path.split(':'):
        if path:
            print(f"     - {path}")
else:
    print("   ⚠️  LD_LIBRARY_PATH is not set")

# Test with MirroredStrategy
print("\n8. Testing MirroredStrategy...")
try:
    strategy = tf.distribute.MirroredStrategy()
    print(f"   ✓ MirroredStrategy created with {strategy.num_replicas_in_sync} devices")
    
    # Try a simple operation within strategy
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(16, 3, input_shape=(10, 64))
        ])
        test_data = tf.random.normal((4, 10, 64))
        output = model(test_data)
        print(f"   ✓ Strategy operations work! Output shape: {output.shape}")
        
except Exception as e:
    print(f"   ✗ Strategy operations failed: {e}")

print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

# Provide recommendations
print("\nIf cuDNN operations failed:")
print("1. Check cuDNN installation:")
print("   - Ensure cuDNN version matches TensorFlow requirements")
print("   - TensorFlow 2.13+ requires cuDNN 8.6+")
print("")
print("2. Set library path (if needed):")
print("   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
print("")
print("3. Reinstall cuDNN:")
print("   conda install -c conda-forge cudnn")
print("   # or")
print("   pip install nvidia-cudnn-cu11")
print("")
print("4. Alternative: Use CPU-only (slower):")
print("   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'")
print("")
print("5. Alternative: Single GPU only:")
print("   os.environ['CUDA_VISIBLE_DEVICES'] = '0'")

print("\n" + "="*70)
