"""
Audio-based emotion detection for BSS Maitri
"""

import librosa
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioEmotionDetector:
    """Audio emotion detection using voice tone and acoustic features"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        self.frame_length = config.get('frame_length', 2048)
        self.hop_length = config.get('hop_length', 512)
        self.n_mels = config.get('n_mels', 128)
        
        # Emotion categories relevant for space crew monitoring
        self.emotion_labels = [
            'calm', 'stress', 'anxiety', 'fatigue', 
            'sadness', 'anger', 'fear', 'joy'
        ]
        
        self.scaler = StandardScaler()
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained emotion detection models"""
        # For now, we'll use a rule-based approach
        # In production, this would load trained ML models
        logger.info("Initializing audio emotion detection models")
        
    def extract_audio_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive audio features for emotion detection"""
        try:
            # Ensure audio is 1D
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data.T)
            
            features = {}
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
            )[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
            )[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                audio_data, frame_length=self.frame_length, hop_length=self.hop_length
            )[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data, sr=self.sample_rate, n_mfcc=13,
                hop_length=self.hop_length
            )
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
            )
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data, sr=self.sample_rate, n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            features['mel_mean'] = np.mean(mel_spec)
            features['mel_std'] = np.std(mel_spec)
            
            # Energy and loudness
            features['energy'] = np.sum(audio_data ** 2)
            features['loudness'] = np.sqrt(features['energy'])
            
            # Pitch features
            try:
                pitches, magnitudes = librosa.core.piptrack(
                    y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
                )
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    features['pitch_mean'] = np.mean(pitch_values)
                    features['pitch_std'] = np.std(pitch_values)
                else:
                    features['pitch_mean'] = 0
                    features['pitch_std'] = 0
            except:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {}
    
    def detect_emotion_from_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Detect emotions using extracted audio features"""
        # Rule-based emotion detection (simplified for demo)
        # In production, this would use trained ML models
        
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_labels}
        
        try:
            # Stress indicators
            if features.get('zcr_mean', 0) > 0.1:
                emotion_scores['stress'] += 0.3
            if features.get('spectral_centroid_mean', 0) > 2000:
                emotion_scores['stress'] += 0.2
                emotion_scores['anxiety'] += 0.2
            
            # Fatigue indicators
            if features.get('energy', 0) < 0.1:
                emotion_scores['fatigue'] += 0.4
            if features.get('pitch_std', 0) < 10:
                emotion_scores['fatigue'] += 0.2
            
            # Sadness indicators
            if features.get('pitch_mean', 0) < 150 and features.get('energy', 0) < 0.2:
                emotion_scores['sadness'] += 0.3
            
            # Anger indicators
            if features.get('loudness', 0) > 0.5 and features.get('zcr_mean', 0) > 0.15:
                emotion_scores['anger'] += 0.4
            
            # Joy indicators
            if (features.get('pitch_mean', 0) > 200 and 
                features.get('energy', 0) > 0.3 and 
                features.get('spectral_centroid_mean', 0) > 1500):
                emotion_scores['joy'] += 0.4
            
            # Calm baseline
            total_negative = sum(emotion_scores[e] for e in ['stress', 'anxiety', 'fatigue', 'sadness', 'anger', 'fear'])
            if total_negative < 0.3:
                emotion_scores['calm'] = max(0.5, 1.0 - total_negative)
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            else:
                emotion_scores['calm'] = 1.0
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            return {emotion: 1.0/len(self.emotion_labels) for emotion in self.emotion_labels}
    
    def process_audio(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Process audio and return emotion predictions"""
        try:
            # Extract features
            features = self.extract_audio_features(audio_data)
            
            if not features:
                logger.warning("No features extracted from audio")
                return {'calm': 1.0}
            
            # Detect emotions
            emotions = self.detect_emotion_from_features(features)
            
            logger.debug(f"Audio emotion detection results: {emotions}")
            return emotions
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {'calm': 1.0}
    
    def process_audio_file(self, audio_path: str) -> Dict[str, float]:
        """Process audio file and return emotion predictions"""
        try:
            # Load audio file
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Process audio
            return self.process_audio(audio_data)
            
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            return {'calm': 1.0}
    
    def get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """Get the dominant emotion from scores"""
        if not emotion_scores:
            return 'calm', 1.0
        
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return dominant_emotion