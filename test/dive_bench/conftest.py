"""Keep the opt-in GRT kernel tests out of incompatible stock environments.

Upstream's generic CPU environment intentionally uses newer Transformers. Native
task/metric/schema and source-AST contracts remain collectible there; execute the
kernel modules in the documented grt extra environment (Transformers 4.57.6).
"""

from importlib.metadata import PackageNotFoundError, version

PINNED_KERNEL_MODULES = {
    "test_llava_hf_grt_embeddings.py",
    "test_qwen25_grt_metrics.py",
    "test_qwen7_cap48_route_floor_plugin.py",
    "test_qwen_dual_route_plugin.py",
}


def pytest_ignore_collect(collection_path):
    if collection_path.name not in PINNED_KERNEL_MODULES:
        return None
    try:
        return version("transformers") != "4.57.6"
    except PackageNotFoundError:
        return True
