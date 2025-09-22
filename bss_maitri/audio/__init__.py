"""Audio processing package for BSS Maitri"""

def __getattr__(name):
    if name == "AudioEmotionDetector":
        from .emotion_detector import AudioEmotionDetector
        return AudioEmotionDetector
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["AudioEmotionDetector"]