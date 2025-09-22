"""Core package for BSS Maitri"""

def __getattr__(name):
    if name == "MaitriAssistant":
        from .assistant import MaitriAssistant
        return MaitriAssistant
    elif name == "EmotionDetector":
        from .emotion_detector import EmotionDetector
        return EmotionDetector
    elif name == "EmotionResult":
        from .emotion_detector import EmotionResult
        return EmotionResult
    elif name == "ConversationEngine":
        from .conversation_engine import ConversationEngine
        return ConversationEngine
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "MaitriAssistant",
    "EmotionDetector", 
    "EmotionResult",
    "ConversationEngine"
]