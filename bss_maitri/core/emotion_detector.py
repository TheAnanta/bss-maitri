"""
Core emotion detection module that combines audio and vision modalities
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

from ..audio.emotion_detector import AudioEmotionDetector
from ..vision.emotion_detector import VisionEmotionDetector

logger = logging.getLogger(__name__)

@dataclass
class EmotionResult:
    """Data class for emotion detection results"""
    emotions: Dict[str, float]
    confidence: float
    dominant_emotion: str
    timestamp: datetime
    modalities: List[str]
    features: Dict[str, Dict[str, float]]

class EmotionDetector:
    """Multimodal emotion detector combining audio and vision"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize modality-specific detectors
        self.audio_detector = AudioEmotionDetector(config.get('audio', {}))
        self.vision_detector = VisionEmotionDetector(config.get('vision', {}))
        
        # Fusion weights for combining modalities
        self.fusion_weights = {
            'audio': 0.6,
            'vision': 0.4
        }
        
        # Emotion mapping for consistency across modalities
        self.emotion_mapping = self._create_emotion_mapping()
        
        # History for temporal smoothing
        self.emotion_history = []
        self.history_size = 5
        
    def _create_emotion_mapping(self) -> Dict[str, str]:
        """Create mapping between different emotion labels"""
        return {
            'calm': 'calm',
            'stress': 'stress', 
            'anxiety': 'anxiety',
            'fatigue': 'fatigue',
            'sadness': 'sadness',
            'anger': 'anger',
            'fear': 'fear',
            'joy': 'joy',
            'surprise': 'surprise'
        }
    
    def _normalize_emotions(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """Normalize emotion scores to sum to 1"""
        total = sum(emotions.values())
        if total > 0:
            return {k: v/total for k, v in emotions.items()}
        else:
            return {'calm': 1.0}
    
    def _fuse_emotions(self, 
                      audio_emotions: Optional[Dict[str, float]], 
                      vision_emotions: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Fuse emotions from multiple modalities"""
        
        # Get all possible emotion labels
        all_emotions = set()
        if audio_emotions:
            all_emotions.update(audio_emotions.keys())
        if vision_emotions:
            all_emotions.update(vision_emotions.keys())
        
        # If no emotions detected, return calm
        if not all_emotions:
            return {'calm': 1.0}
        
        # Initialize fused emotions
        fused_emotions = {emotion: 0.0 for emotion in all_emotions}
        
        # Weighted fusion
        total_weight = 0.0
        
        if audio_emotions:
            weight = self.fusion_weights['audio']
            for emotion, score in audio_emotions.items():
                fused_emotions[emotion] += weight * score
            total_weight += weight
        
        if vision_emotions:
            weight = self.fusion_weights['vision']
            for emotion, score in vision_emotions.items():
                fused_emotions[emotion] += weight * score
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            fused_emotions = {k: v/total_weight for k, v in fused_emotions.items()}
        
        return self._normalize_emotions(fused_emotions)
    
    def _apply_temporal_smoothing(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """Apply temporal smoothing to emotion predictions"""
        
        # Add current emotions to history
        self.emotion_history.append(emotions)
        
        # Keep only recent history
        if len(self.emotion_history) > self.history_size:
            self.emotion_history = self.emotion_history[-self.history_size:]
        
        # If we don't have enough history, return current emotions
        if len(self.emotion_history) < 2:
            return emotions
        
        # Apply exponential smoothing
        alpha = 0.6  # Smoothing factor
        smoothed_emotions = {}
        
        # Get all emotion labels
        all_emotions = set()
        for hist_emotions in self.emotion_history:
            all_emotions.update(hist_emotions.keys())
        
        # Apply smoothing for each emotion
        for emotion in all_emotions:
            current_score = emotions.get(emotion, 0.0)
            
            # Get previous smoothed score
            if len(self.emotion_history) >= 2:
                prev_score = self.emotion_history[-2].get(emotion, 0.0)
            else:
                prev_score = 0.0
            
            # Apply exponential smoothing
            smoothed_score = alpha * current_score + (1 - alpha) * prev_score
            smoothed_emotions[emotion] = smoothed_score
        
        return self._normalize_emotions(smoothed_emotions)
    
    def detect_emotions(self, 
                       audio_data: Optional[np.ndarray] = None,
                       frame: Optional[np.ndarray] = None) -> EmotionResult:
        """Detect emotions from audio and/or visual input"""
        
        try:
            # Initialize results
            audio_emotions = None
            vision_emotions = None
            modalities = []
            features = {}
            
            # Process audio if provided
            if audio_data is not None:
                try:
                    audio_emotions = self.audio_detector.process_audio(audio_data)
                    modalities.append('audio')
                    features['audio'] = self.audio_detector.extract_audio_features(audio_data)
                    logger.debug(f"Audio emotions: {audio_emotions}")
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
            
            # Process vision if provided
            if frame is not None:
                try:
                    vision_emotions = self.vision_detector.process_frame(frame)
                    modalities.append('vision')
                    
                    # Extract landmarks for features
                    landmarks = self.vision_detector.extract_facial_landmarks(frame)
                    if landmarks:
                        features['vision'] = self.vision_detector.calculate_facial_features(landmarks)
                    
                    logger.debug(f"Vision emotions: {vision_emotions}")
                except Exception as e:
                    logger.error(f"Error processing vision: {e}")
            
            # Fuse emotions from available modalities
            fused_emotions = self._fuse_emotions(audio_emotions, vision_emotions)
            
            # Apply temporal smoothing
            smoothed_emotions = self._apply_temporal_smoothing(fused_emotions)
            
            # Calculate confidence based on consistency and strength
            confidence = self._calculate_confidence(
                smoothed_emotions, audio_emotions, vision_emotions
            )
            
            # Get dominant emotion
            dominant_emotion, dominant_score = max(
                smoothed_emotions.items(), key=lambda x: x[1]
            )
            
            # Create result
            result = EmotionResult(
                emotions=smoothed_emotions,
                confidence=confidence,
                dominant_emotion=dominant_emotion,
                timestamp=datetime.now(),
                modalities=modalities,
                features=features
            )
            
            logger.info(f"Emotion detection result: {dominant_emotion} ({dominant_score:.2f}) "
                       f"[confidence: {confidence:.2f}, modalities: {modalities}]")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            # Return default calm state
            return EmotionResult(
                emotions={'calm': 1.0},
                confidence=0.5,
                dominant_emotion='calm',
                timestamp=datetime.now(),
                modalities=[],
                features={}
            )
    
    def _calculate_confidence(self, 
                            fused_emotions: Dict[str, float],
                            audio_emotions: Optional[Dict[str, float]],
                            vision_emotions: Optional[Dict[str, float]]) -> float:
        """Calculate confidence in emotion prediction"""
        
        try:
            confidence_factors = []
            
            # Factor 1: Strength of dominant emotion
            if fused_emotions:
                max_score = max(fused_emotions.values())
                confidence_factors.append(max_score)
            
            # Factor 2: Consistency between modalities
            if audio_emotions and vision_emotions:
                # Calculate correlation between emotion vectors
                common_emotions = set(audio_emotions.keys()) & set(vision_emotions.keys())
                if common_emotions:
                    audio_vector = [audio_emotions.get(e, 0) for e in common_emotions]
                    vision_vector = [vision_emotions.get(e, 0) for e in common_emotions]
                    
                    # Simple correlation measure
                    if len(audio_vector) > 1:
                        correlation = np.corrcoef(audio_vector, vision_vector)[0, 1]
                        if not np.isnan(correlation):
                            confidence_factors.append(abs(correlation))
            
            # Factor 3: Temporal consistency
            if len(self.emotion_history) >= 2:
                current_dominant = max(fused_emotions.items(), key=lambda x: x[1])[0]
                prev_dominant = max(self.emotion_history[-2].items(), key=lambda x: x[1])[0]
                
                if current_dominant == prev_dominant:
                    confidence_factors.append(0.8)
                else:
                    confidence_factors.append(0.3)
            
            # Calculate overall confidence
            if confidence_factors:
                confidence = np.mean(confidence_factors)
                return min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def get_emotion_summary(self, time_window_minutes: int = 5) -> Dict[str, float]:
        """Get emotion summary over a time window"""
        
        if not self.emotion_history:
            return {'calm': 1.0}
        
        # For now, just return the average of recent emotions
        # In production, this would filter by actual time window
        recent_emotions = self.emotion_history[-min(len(self.emotion_history), 10):]
        
        # Average emotions over time window
        all_emotions = set()
        for emotions in recent_emotions:
            all_emotions.update(emotions.keys())
        
        summary = {}
        for emotion in all_emotions:
            scores = [emotions.get(emotion, 0.0) for emotions in recent_emotions]
            summary[emotion] = np.mean(scores)
        
        return self._normalize_emotions(summary)
    
    def is_critical_state(self, emotion_result: EmotionResult) -> bool:
        """Determine if the emotional state requires immediate attention"""
        
        critical_emotions = ['stress', 'anxiety', 'anger', 'fear', 'sadness']
        critical_threshold = self.config.get('critical_emotion_threshold', 0.8)
        
        for emotion in critical_emotions:
            if emotion_result.emotions.get(emotion, 0.0) > critical_threshold:
                return True
        
        # Check for fatigue which is critical in space operations
        if emotion_result.emotions.get('fatigue', 0.0) > 0.7:
            return True
        
        return False