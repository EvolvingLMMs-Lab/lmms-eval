"""
Shim: re-export the shared HD-EPIC utils so YAMLs in this subfolder can
reference !function utils.filter_* without a path prefix.

lmms-eval resolves !function module names relative to each YAML's own
directory, so each category subfolder needs `utils` reachable from here.
"""
import os
import sys

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from utils import *  # noqa: F401,F403,E402
