#!/usr/bin/env python3
"""
Test script to verify UniWorld import works with the transformers patch
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Testing UniWorld Import with Transformers Patch")
print("=" * 60)

try:
    print("\n1. Importing transformers...")
    from transformers import modeling_utils
    print(f"   ✅ transformers version: {__import__('transformers').__version__}")
    
    print("\n2. Checking for restore_default_torch_dtype...")
    has_function = hasattr(modeling_utils, 'restore_default_torch_dtype')
    print(f"   {'✅' if has_function else '⚠️ '} Has restore_default_torch_dtype: {has_function}")
    
    print("\n3. Importing UniWorld model...")
    from lmms_eval.models.simple.uniworld import UniWorld
    print("   ✅ UniWorld imported successfully!")
    
    print("\n4. Verifying patch was applied...")
    has_function_after = hasattr(modeling_utils, 'restore_default_torch_dtype')
    print(f"   {'✅' if has_function_after else '❌'} Has restore_default_torch_dtype after import: {has_function_after}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nPlease check:")
    print("  1. UniWorld repository is cloned to: ./UniWorld/UniWorld-V1/")
    print("  2. transformers is installed: pip install transformers")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
