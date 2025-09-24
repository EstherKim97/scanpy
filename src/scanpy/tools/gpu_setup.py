"""

GPU Environment Setup and Library Loading
Handles CUDA library fixes and GPU initialization

"""

import os
import ctypes
import numpy as np
import scipy.sparse as sp

def fix_cuda_libraries():
    """Fix CUDA library loading issues - comprehensive version"""
    
    base_path = '/home/shadeform/.venv/lib/python3.12/site-packages/nvidia'
    
    # List of critical CUDA libraries to load
    cuda_libs = [
        ('cuda_runtime', 'libcudart.so.12'),
        ('cuda_nvrtc', 'libnvrtc.so.12'),
        ('cuda_nvrtc', 'libnvrtc-builtins.so.12'),
        ('nvjitlink', 'libnvjitlink.so.12'),
        ('cublas', 'libcublas.so.12'),
        ('cufft', 'libcufft.so.12'),
        ('cusolver', 'libcusolver.so.12'),
        ('cusparse', 'libcusparse.so.12'),
        ('curand', 'libcurand.so.12')
    ]
    
    loaded_libs = []
    failed_libs = []
    
    for lib_dir, lib_name in cuda_libs:
        lib_path = os.path.join(base_path, lib_dir, 'lib', lib_name)
        try:
            if os.path.exists(lib_path):
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                loaded_libs.append(lib_name)
            else:
                failed_libs.append(f"{lib_name} (not found at {lib_path})")
        except Exception as e:
            failed_libs.append(f"{lib_name} (error: {e})")
    
    # Set LD_LIBRARY_PATH as backup
    lib_paths = []
    for lib_dir, _ in cuda_libs:
        lib_path = os.path.join(base_path, lib_dir, 'lib')
        if os.path.exists(lib_path):
            lib_paths.append(lib_path)
    
    if lib_paths:
        current_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_path = ':'.join(lib_paths)
        os.environ['LD_LIBRARY_PATH'] = f"{new_path}:{current_path}" if current_path else new_path
    
    print(f"✅ Loaded {len(loaded_libs)} libraries: {', '.join(loaded_libs)}")
    if failed_libs:
        print(f"⚠️  Failed to load {len(failed_libs)} libraries: {', '.join(failed_libs)}")
    
    return len(loaded_libs) > 0

def init_gpu(seed=42, use_rmm=False):
    """Initialize GPU environment"""
    # Apply CUDA library fix first
    fix_cuda_libraries()
    
    # Import after library fix
    import cupy as cp
    from cuml.common import set_global_output_type
    
    # Test CuPy functionality
    try:
        test_array = cp.array([1, 2, 3])
        print(f"✅ CuPy working: {test_array}")
    except Exception as e:
        print(f"❌ CuPy failed: {e}")
        raise
    
    # Set random seed and output type
    cp.random.seed(seed)
    set_global_output_type("cupy")
    
    env_config = {"seed": seed, "rmm_enabled": use_rmm}
    print(f"[GPU READY] {env_config}")
    
    return env_config

# Constants
SEED = 42
EPS = 1e-12

if __name__ == "__main__":
    # Test the setup
    env = init_gpu(seed=SEED)
    print("GPU setup completed successfully!")