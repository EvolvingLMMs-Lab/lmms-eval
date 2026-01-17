def __getattr__(name: str):
    """Lazy imports to avoid circular import issues."""
    if name == "run_tui":
        from lmms_eval.tui.cli import main

        return main
    if name == "server_app":
        from lmms_eval.tui.server import app

        return app
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["run_tui", "server_app"]
