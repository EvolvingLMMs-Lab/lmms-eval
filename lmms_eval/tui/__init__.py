from lmms_eval.imports import make_lazy_getattr

# Lazy imports to avoid circular import issues and speed up startup
__getattr__ = make_lazy_getattr(
    {
        "run_tui": ("lmms_eval.tui.cli", "main"),
        "server_app": ("lmms_eval.tui.server", "app"),
    }
)

__all__ = ["run_tui", "server_app"]
