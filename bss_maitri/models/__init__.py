"""Models package for BSS Maitri"""

def __getattr__(name):
    if name == "GemmaModel":
        from .gemma_model import GemmaModel
        return GemmaModel
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["GemmaModel"]