"""Utility functions for BSS Maitri"""

import cv2
import numpy as np
import librosa
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_audio_file(filepath: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample if needed"""
    try:
        audio_data, sr = librosa.load(filepath, sr=target_sr)
        return audio_data, sr
    except Exception as e:
        logger.error(f"Error loading audio file {filepath}: {e}")
        raise

def load_video_frame(video_source: Any, frame_number: Optional[int] = None) -> Optional[np.ndarray]:
    """Load frame from video source"""
    try:
        if isinstance(video_source, str):
            # Load from file
            cap = cv2.VideoCapture(video_source)
            if frame_number is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        else:
            # Assume it's already a frame
            return video_source
    except Exception as e:
        logger.error(f"Error loading video frame: {e}")
        return None

def preprocess_audio(audio_data: np.ndarray, 
                    target_length: Optional[int] = None,
                    normalize: bool = True) -> np.ndarray:
    """Preprocess audio data"""
    try:
        # Normalize
        if normalize and len(audio_data) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Pad or truncate to target length
        if target_length is not None:
            if len(audio_data) < target_length:
                # Pad with zeros
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
            elif len(audio_data) > target_length:
                # Truncate
                audio_data = audio_data[:target_length]
        
        return audio_data
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        return audio_data

def preprocess_frame(frame: np.ndarray, 
                    target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Preprocess video frame"""
    try:
        # Resize if needed
        if target_size is not None:
            frame = cv2.resize(frame, target_size)
        
        # Ensure RGB format
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    except Exception as e:
        logger.error(f"Error preprocessing frame: {e}")
        return frame

def format_emotion_scores(emotions: Dict[str, float], top_k: int = 3) -> str:
    """Format emotion scores for display"""
    try:
        # Sort emotions by score
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        # Format top-k emotions
        formatted = []
        for emotion, score in sorted_emotions[:top_k]:
            formatted.append(f"{emotion}: {score:.2f}")
        
        return ", ".join(formatted)
    except Exception as e:
        logger.error(f"Error formatting emotion scores: {e}")
        return str(emotions)

def calculate_audio_metrics(audio_data: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """Calculate basic audio metrics"""
    try:
        metrics = {}
        
        # Duration
        metrics['duration'] = len(audio_data) / sr
        
        # Energy
        metrics['energy'] = np.sum(audio_data ** 2) / len(audio_data)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        metrics['zcr_mean'] = np.mean(zcr)
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        metrics['spectral_centroid_mean'] = np.mean(spectral_centroids)
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating audio metrics: {e}")
        return {}

def validate_audio_input(audio_data: np.ndarray) -> bool:
    """Validate audio input"""
    try:
        if audio_data is None or len(audio_data) == 0:
            return False
        
        if not isinstance(audio_data, np.ndarray):
            return False
        
        # Check for reasonable audio length (at least 0.1 seconds at 16kHz)
        if len(audio_data) < 1600:
            return False
        
        # Check for non-zero variance (not all silence)
        if np.var(audio_data) < 1e-10:
            return False
        
        return True
    except:
        return False

def validate_video_input(frame: np.ndarray) -> bool:
    """Validate video frame input"""
    try:
        if frame is None:
            return False
        
        if not isinstance(frame, np.ndarray):
            return False
        
        # Check dimensions
        if len(frame.shape) not in [2, 3]:
            return False
        
        # Check minimum size
        if frame.shape[0] < 50 or frame.shape[1] < 50:
            return False
        
        return True
    except:
        return False

class PerformanceMonitor:
    """Simple performance monitoring utility"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        import time
        self.metrics[f"{name}_start"] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration"""
        import time
        start_time = self.metrics.get(f"{name}_start")
        if start_time is None:
            return 0.0
        
        duration = time.time() - start_time
        self.metrics[f"{name}_duration"] = duration
        return duration
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics"""
        return {k: v for k, v in self.metrics.items() if not k.endswith('_start')}
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()