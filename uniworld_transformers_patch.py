"""
Compatibility patch for UniWorld with newer transformers versions.

The `restore_default_torch_dtype` function was removed in newer transformers.
This patch provides a compatible implementation.
"""

import torch
from contextlib import contextmanager

# Monkey-patch the missing function into transformers.modeling_utils
try:
    from transformers import modeling_utils
    
    if not hasattr(modeling_utils, 'restore_default_torch_dtype'):
        @contextmanager
        def restore_default_torch_dtype(*args, **kwargs):
            """
            Context manager to restore default torch dtype after temporarily changing it.
            This is a compatibility shim for older transformers code.
            Accepts any arguments for compatibility but ignores them.
            """
            original_dtype = torch.get_default_dtype()
            try:
                yield
            finally:
                torch.set_default_dtype(original_dtype)
        
        # Add the function to transformers.modeling_utils
        modeling_utils.restore_default_torch_dtype = restore_default_torch_dtype
        print("✅ Patched transformers.modeling_utils with restore_default_torch_dtype")
        
except ImportError:
    print("⚠️  transformers not installed, skipping patch")
