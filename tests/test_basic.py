"""
Basic tests for BSS Maitri components
"""

import pytest
import numpy as np
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_import_main_modules():
    """Test that main modules can be imported"""
    try:
        from bss_maitri.audio.emotion_detector import AudioEmotionDetector
        from bss_maitri.video.facial_analyzer import FacialAnalyzer
        from bss_maitri.utils.multimodal_analyzer import MultimodalEmotionAnalyzer
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")

def test_audio_emotion_detector():
    """Test audio emotion detector with dummy data"""
    try:
        from bss_maitri.audio.emotion_detector import AudioEmotionDetector
        
        detector = AudioEmotionDetector()
        
        # Create dummy audio data (1 second of random noise)
        dummy_audio = np.random.randn(16000).astype(np.float32)
        
        result = detector.process_audio(dummy_audio)
        
        # Check that result has expected keys
        expected_keys = ['emotion', 'confidence', 'stress_level', 'features', 'concerns']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check that emotion is valid
        valid_emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'stressed']
        assert result['emotion'] in valid_emotions
        
        # Check confidence is in valid range
        assert 0.0 <= result['confidence'] <= 1.0
        
        # Check stress level is in valid range
        assert 0.0 <= result['stress_level'] <= 1.0
        
    except Exception as e:
        pytest.fail(f"Audio emotion detector test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])