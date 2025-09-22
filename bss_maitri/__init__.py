"""
BSS Maitri - Bharatiya Space Station Multimodal AI Assistant
A comprehensive AI system for crew emotional and physical well-being monitoring.
"""

__version__ = "0.1.0"
__author__ = "BSS Maitri Team"

# Import classes on demand to avoid heavy dependency loading
def __getattr__(name):
    if name == "MaitriAssistant":
        from .core.assistant import MaitriAssistant
        return MaitriAssistant
    elif name == "EmotionDetector":
        from .core.emotion_detector import EmotionDetector
        return EmotionDetector
    elif name == "ConversationEngine":
        from .core.conversation_engine import ConversationEngine
        return ConversationEngine
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "MaitriAssistant",
    "EmotionDetector", 
    "ConversationEngine",
]