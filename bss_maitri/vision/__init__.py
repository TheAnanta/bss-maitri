"""Vision processing package for BSS Maitri"""

def __getattr__(name):
    if name == "VisionEmotionDetector":
        from .emotion_detector import VisionEmotionDetector
        return VisionEmotionDetector
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["VisionEmotionDetector"]