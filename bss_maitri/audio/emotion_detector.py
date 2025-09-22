"""
Audio emotion detection module for analyzing emotional state from voice
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class AudioEmotionDetector:
    """Detect emotions from audio input using voice characteristics"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio emotion detector
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'stressed']
        
    def extract_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Extract audio features for emotion analysis
        
        Args:
            audio_data: Audio signal as numpy array
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}
            
            # Basic audio properties
            features['duration'] = len(audio_data) / self.sample_rate
            features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # Pitch and fundamental frequency
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
            pitches = pitches[magnitudes > np.percentile(magnitudes, 85)]
            if len(pitches) > 0:
                features['pitch_mean'] = np.mean(pitches[pitches > 0])
                features['pitch_std'] = np.std(pitches[pitches > 0])
                features['pitch_range'] = np.max(pitches) - np.min(pitches[pitches > 0])
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            features['tempo'] = tempo
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {}
    
    def analyze_stress_indicators(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze stress indicators from audio features
        
        Args:
            features: Extracted audio features
            
        Returns:
            Stress analysis results
        """
        stress_indicators = {}
        
        # High fundamental frequency can indicate stress
        if features.get('pitch_mean', 0) > 180:  # Higher than normal speaking
            stress_indicators['high_pitch'] = min(1.0, (features['pitch_mean'] - 180) / 100)
        else:
            stress_indicators['high_pitch'] = 0.0
        
        # High pitch variability can indicate emotional instability
        if features.get('pitch_std', 0) > 50:
            stress_indicators['pitch_variability'] = min(1.0, (features['pitch_std'] - 50) / 100)
        else:
            stress_indicators['pitch_variability'] = 0.0
        
        # Fast speaking rate can indicate anxiety
        if features.get('tempo', 0) > 140:
            stress_indicators['fast_tempo'] = min(1.0, (features['tempo'] - 140) / 60)
        else:
            stress_indicators['fast_tempo'] = 0.0
        
        # High energy can indicate agitation
        if features.get('rms_energy', 0) > 0.1:
            stress_indicators['high_energy'] = min(1.0, (features['rms_energy'] - 0.1) / 0.1)
        else:
            stress_indicators['high_energy'] = 0.0
        
        # Calculate overall stress level
        stress_level = np.mean(list(stress_indicators.values()))
        stress_indicators['overall_stress'] = stress_level
        
        return stress_indicators
    
    def classify_emotion(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify emotion based on audio features using rule-based approach
        
        Args:
            features: Extracted audio features
            
        Returns:
            Tuple of (emotion, confidence)
        """
        try:
            emotion_scores = {}
            
            # Rule-based emotion classification
            pitch_mean = features.get('pitch_mean', 150)
            pitch_std = features.get('pitch_std', 30)
            energy = features.get('rms_energy', 0.05)
            tempo = features.get('tempo', 120)
            spectral_centroid = features.get('spectral_centroid_mean', 2000)
            
            # Happy: Higher pitch, higher energy, faster tempo
            emotion_scores['happy'] = (
                (pitch_mean - 150) / 100 * 0.3 +
                (energy - 0.05) / 0.1 * 0.3 +
                (tempo - 120) / 60 * 0.2 +
                (spectral_centroid - 2000) / 1000 * 0.2
            )
            
            # Sad: Lower pitch, lower energy, slower tempo
            emotion_scores['sad'] = (
                (150 - pitch_mean) / 50 * 0.4 +
                (0.05 - energy) / 0.03 * 0.3 +
                (120 - tempo) / 40 * 0.3
            )
            
            # Angry: Higher pitch variation, higher energy, faster tempo
            emotion_scores['angry'] = (
                pitch_std / 100 * 0.3 +
                (energy - 0.05) / 0.1 * 0.4 +
                (tempo - 120) / 60 * 0.3
            )
            
            # Fear/Stressed: Very high pitch, high variation, high energy
            emotion_scores['fear'] = (
                (pitch_mean - 180) / 100 * 0.3 +
                pitch_std / 100 * 0.4 +
                (energy - 0.08) / 0.1 * 0.3
            )
            
            emotion_scores['stressed'] = emotion_scores['fear'] * 0.8 + (tempo - 140) / 60 * 0.2
            
            # Neutral: Average values
            emotion_scores['neutral'] = 1.0 - max(emotion_scores.values())
            
            # Normalize scores to [0, 1]
            for emotion in emotion_scores:
                emotion_scores[emotion] = max(0.0, min(1.0, emotion_scores[emotion]))
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion]
            
            # Ensure minimum confidence for neutral
            if confidence < 0.3:
                dominant_emotion = 'neutral'
                confidence = 0.5
            
            return dominant_emotion, confidence
            
        except Exception as e:
            logger.error(f"Error classifying emotion: {e}")
            return 'neutral', 0.5
    
    def process_audio(self, audio_data: np.ndarray, 
                     sample_rate: Optional[int] = None) -> Dict[str, any]:
        """
        Process audio and return emotion analysis
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of the audio (if different from default)
            
        Returns:
            Dictionary containing emotion analysis results
        """
        try:
            # Resample if necessary
            if sample_rate and sample_rate != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Extract features
            features = self.extract_features(audio_data)
            
            if not features:
                return {
                    'emotion': 'neutral',
                    'confidence': 0.5,
                    'stress_level': 0.0,
                    'features': {},
                    'stress_indicators': {},
                    'concerns': []
                }
            
            # Classify emotion
            emotion, confidence = self.classify_emotion(features)
            
            # Analyze stress indicators
            stress_indicators = self.analyze_stress_indicators(features)
            stress_level = stress_indicators.get('overall_stress', 0.0)
            
            # Identify concerns
            concerns = []
            if stress_level > 0.7:
                concerns.append("High stress level detected")
            if emotion in ['angry', 'fear', 'stressed']:
                concerns.append(f"Negative emotion detected: {emotion}")
            if features.get('pitch_mean', 150) > 250:
                concerns.append("Unusually high voice pitch")
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'stress_level': stress_level,
                'features': features,
                'stress_indicators': stress_indicators,
                'concerns': concerns
            }
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.5,
                'stress_level': 0.0,
                'features': {},
                'stress_indicators': {},
                'concerns': ['Error processing audio']
            }
    
    def load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data and sample rate
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=self.sample_rate)
            return audio_data, sample_rate
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            return np.array([]), 0