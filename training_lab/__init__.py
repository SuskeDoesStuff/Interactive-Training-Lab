"""Interactive Training Lab — a live, visual PyTorch training dashboard for CIFAR-10."""

__version__ = "1.0.0"
__all__ = ["main"]


def __getattr__(name):
    # Lazy so that importing a lightweight submodule (e.g. training_lab.config)
    # does not pull in gradio. `from training_lab import main` still works.
    if name == "main":
        from .app import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
