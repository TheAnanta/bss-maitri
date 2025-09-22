"""
Multimodal emotion analysis system that combines audio and video analysis
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

from ..audio.emotion_detector import AudioEmotionDetector
from ..video.facial_analyzer import FacialAnalyzer

logger = logging.getLogger(__name__)


class MultimodalEmotionAnalyzer:
    """Combine audio and video analysis for comprehensive emotion detection"""
    
    def __init__(self, audio_weight: float = 0.5, video_weight: float = 0.5):
        """
        Initialize multimodal analyzer
        
        Args:
            audio_weight: Weight for audio analysis (0-1)
            video_weight: Weight for video analysis (0-1)
        """
        self.audio_detector = AudioEmotionDetector()
        self.facial_analyzer = FacialAnalyzer()
        
        # Ensure weights sum to 1
        total_weight = audio_weight + video_weight
        self.audio_weight = audio_weight / total_weight
        self.video_weight = video_weight / total_weight
        
        # Emotion mapping for consistency
        self.emotion_mapping = {
            'happy': 'happy',
            'sad': 'sad', 
            'angry': 'angry',
            'fear': 'fear',
            'stressed': 'fear',  # Map stressed to fear
            'surprised': 'fear',  # Map surprised to fear for simplicity
            'disgust': 'angry',   # Map disgust to angry for simplicity
            'neutral': 'neutral'
        }
        
        # History for temporal analysis
        self.analysis_history = []
        self.max_history = 10  # Keep last 10 analyses
    
    def normalize_emotions(self, audio_emotion: str, video_emotion: str) -> Tuple[str, str]:
        """
        Normalize emotion labels to common set
        
        Args:
            audio_emotion: Emotion from audio analysis
            video_emotion: Emotion from video analysis
            
        Returns:
            Tuple of normalized emotions
        """
        normalized_audio = self.emotion_mapping.get(audio_emotion, 'neutral')
        normalized_video = self.emotion_mapping.get(video_emotion, 'neutral')
        return normalized_audio, normalized_video
    
    def combine_emotions(self, audio_result: Dict[str, Any], 
                        video_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine audio and video emotion analysis results
        
        Args:
            audio_result: Results from audio emotion detection
            video_result: Results from video facial analysis
            
        Returns:
            Combined multimodal analysis results
        """
        try:
            # Extract individual results
            audio_emotion = audio_result.get('emotion', 'neutral')
            audio_confidence = audio_result.get('confidence', 0.0)
            audio_stress = audio_result.get('stress_level', 0.0)
            
            video_emotion = video_result.get('emotion', 'neutral')
            video_confidence = video_result.get('confidence', 0.0)
            video_stress = video_result.get('stress_level', 0.0)
            
            # Normalize emotions
            norm_audio_emotion, norm_video_emotion = self.normalize_emotions(
                audio_emotion, video_emotion
            )
            
            # Calculate weighted confidence scores
            audio_score = audio_confidence * self.audio_weight
            video_score = video_confidence * self.video_weight
            
            # Determine combined emotion
            if norm_audio_emotion == norm_video_emotion:
                # Both modalities agree
                combined_emotion = norm_audio_emotion
                combined_confidence = min(1.0, audio_score + video_score)
                agreement = 1.0
            else:
                # Modalities disagree - use the one with higher weighted confidence
                if audio_score > video_score:
                    combined_emotion = norm_audio_emotion
                    combined_confidence = audio_score
                else:
                    combined_emotion = norm_video_emotion
                    combined_confidence = video_score
                
                # Calculate agreement score
                agreement = 1.0 - abs(audio_score - video_score)
            
            # Combine stress levels
            combined_stress = (audio_stress * self.audio_weight + 
                             video_stress * self.video_weight)
            
            # Combine concerns
            combined_concerns = []
            combined_concerns.extend(audio_result.get('concerns', []))
            combined_concerns.extend(video_result.get('concerns', []))
            
            # Add multimodal concerns
            if agreement < 0.5:
                combined_concerns.append("Low agreement between audio and video analysis")
            
            if combined_stress > 0.8:
                combined_concerns.append("Very high stress level detected across modalities")
            
            # Check for critical combinations
            if (norm_audio_emotion in ['angry', 'fear'] and 
                norm_video_emotion in ['angry', 'fear']):
                combined_concerns.append("Consistent negative emotions across modalities")
            
            result = {
                'combined_emotion': combined_emotion,
                'combined_confidence': combined_confidence,
                'combined_stress': combined_stress,
                'agreement_score': agreement,
                'audio_emotion': audio_emotion,
                'audio_confidence': audio_confidence,
                'video_emotion': video_emotion,
                'video_confidence': video_confidence,
                'concerns': list(set(combined_concerns)),  # Remove duplicates
                'timestamp': datetime.now().isoformat(),
                'modality_weights': {
                    'audio': self.audio_weight,
                    'video': self.video_weight
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error combining emotions: {e}")
            return {
                'combined_emotion': 'neutral',
                'combined_confidence': 0.0,
                'combined_stress': 0.0,
                'agreement_score': 0.0,
                'audio_emotion': 'neutral',
                'audio_confidence': 0.0,
                'video_emotion': 'neutral', 
                'video_confidence': 0.0,
                'concerns': ['Error in multimodal analysis'],
                'timestamp': datetime.now().isoformat(),
                'modality_weights': {
                    'audio': self.audio_weight,
                    'video': self.video_weight
                }
            }
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """
        Analyze temporal patterns in emotion history
        
        Returns:
            Temporal analysis results
        """
        if len(self.analysis_history) < 3:
            return {
                'trend': 'stable',
                'avg_stress': 0.0,
                'dominant_emotion': 'neutral',
                'variability': 0.0,
                'declining_trend': False
            }
        
        try:
            # Extract recent data
            recent_stress = [item['combined_stress'] for item in self.analysis_history[-5:]]
            recent_emotions = [item['combined_emotion'] for item in self.analysis_history[-5:]]
            
            # Calculate average stress
            avg_stress = np.mean(recent_stress)
            
            # Detect stress trend
            if len(recent_stress) >= 3:
                trend_slope = np.polyfit(range(len(recent_stress)), recent_stress, 1)[0]
                if trend_slope > 0.1:
                    trend = 'increasing'
                elif trend_slope < -0.1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            # Find dominant emotion
            emotion_counts = {}
            for emotion in recent_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
            
            # Calculate emotional variability
            unique_emotions = len(set(recent_emotions))
            variability = unique_emotions / len(recent_emotions) if recent_emotions else 0.0
            
            # Check for declining mental state
            declining_trend = (
                avg_stress > 0.7 or
                trend == 'increasing' or
                dominant_emotion in ['angry', 'fear', 'sad']
            )
            
            return {
                'trend': trend,
                'avg_stress': avg_stress,
                'dominant_emotion': dominant_emotion,
                'variability': variability,
                'declining_trend': declining_trend,
                'history_length': len(self.analysis_history)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            return {
                'trend': 'stable',
                'avg_stress': 0.0,
                'dominant_emotion': 'neutral',
                'variability': 0.0,
                'declining_trend': False
            }
    
    def process_multimodal(self, audio_data: Optional[np.ndarray] = None,
                          video_frame: Optional[np.ndarray] = None,
                          audio_sample_rate: Optional[int] = None) -> Dict[str, Any]:
        """
        Process multimodal input and return comprehensive analysis
        
        Args:
            audio_data: Audio signal as numpy array
            video_frame: Video frame as numpy array
            audio_sample_rate: Sample rate for audio data
            
        Returns:
            Comprehensive multimodal analysis results
        """
        try:
            results = {
                'multimodal_available': False,
                'audio_available': audio_data is not None,
                'video_available': video_frame is not None,
                'timestamp': datetime.now().isoformat()
            }
            
            # Process audio if available
            if audio_data is not None:
                audio_result = self.audio_detector.process_audio(audio_data, audio_sample_rate)
                results['audio_analysis'] = audio_result
            else:
                results['audio_analysis'] = {
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'stress_level': 0.0,
                    'concerns': ['No audio input']
                }
            
            # Process video if available
            if video_frame is not None:
                video_result = self.facial_analyzer.process_image(video_frame)
                results['video_analysis'] = video_result
            else:
                results['video_analysis'] = {
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'stress_level': 0.0,
                    'concerns': ['No video input'],
                    'face_detected': False
                }
            
            # Combine results if both modalities are available
            if audio_data is not None and video_frame is not None:
                combined_result = self.combine_emotions(
                    results['audio_analysis'], 
                    results['video_analysis']
                )
                results['combined_analysis'] = combined_result
                results['multimodal_available'] = True
                
                # Add to history
                self.analysis_history.append(combined_result)
                if len(self.analysis_history) > self.max_history:
                    self.analysis_history.pop(0)
                
            else:
                # Use single modality result
                if audio_data is not None:
                    primary_result = results['audio_analysis']
                else:
                    primary_result = results['video_analysis']
                
                results['combined_analysis'] = {
                    'combined_emotion': primary_result['emotion'],
                    'combined_confidence': primary_result['confidence'],
                    'combined_stress': primary_result['stress_level'],
                    'agreement_score': 1.0,  # Single modality, so perfect agreement
                    'concerns': primary_result['concerns'],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Add temporal analysis
            results['temporal_analysis'] = self.analyze_temporal_patterns()
            
            # Overall assessment
            combined = results['combined_analysis']
            temporal = results['temporal_analysis']
            
            results['overall_assessment'] = {
                'current_state': combined['combined_emotion'],
                'stress_level': combined['combined_stress'],
                'confidence': combined['combined_confidence'],
                'trend': temporal['trend'],
                'requires_attention': (
                    combined['combined_stress'] > 0.7 or
                    temporal['declining_trend'] or
                    len(combined['concerns']) > 2
                ),
                'critical_level': self._assess_criticality(combined, temporal)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multimodal processing: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'overall_assessment': {
                    'current_state': 'neutral',
                    'stress_level': 0.0,
                    'confidence': 0.0,
                    'requires_attention': False,
                    'critical_level': 'low'
                }
            }
    
    def _assess_criticality(self, combined: Dict[str, Any], 
                           temporal: Dict[str, Any]) -> str:
        """
        Assess the criticality level of the current emotional state
        
        Args:
            combined: Combined analysis results
            temporal: Temporal analysis results
            
        Returns:
            Criticality level: 'low', 'medium', 'high', 'critical'
        """
        stress = combined.get('combined_stress', 0.0)
        emotion = combined.get('combined_emotion', 'neutral')
        concerns = len(combined.get('concerns', []))
        declining = temporal.get('declining_trend', False)
        avg_stress = temporal.get('avg_stress', 0.0)
        
        # Critical level assessment
        if (stress > 0.9 or 
            (emotion in ['angry', 'fear'] and stress > 0.7) or
            (declining and avg_stress > 0.8)):
            return 'critical'
        
        elif (stress > 0.7 or 
              (emotion in ['angry', 'fear', 'sad'] and stress > 0.5) or
              concerns > 3 or
              declining):
            return 'high'
        
        elif (stress > 0.5 or 
              emotion in ['angry', 'fear', 'sad'] or
              concerns > 1):
            return 'medium'
        
        else:
            return 'low'