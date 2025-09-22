"""
Facial expression analysis module for detecting emotions from facial expressions
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FacialAnalyzer:
    """Analyze facial expressions to detect emotions and stress indicators"""
    
    def __init__(self):
        """Initialize facial analyzer with MediaPipe"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face mesh and detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        # Emotion categories
        self.emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fear', 'disgust']
        
        # Key landmark indices for facial features
        self.landmark_indices = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'mouth': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 78],
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 285, 336, 296, 334, 293, 300, 276, 283, 282, 295],
            'forehead': [10, 151, 9, 10, 151, 337, 299, 333, 298, 301],
            'nose': [1, 2, 5, 4, 6, 168, 8, 9, 10, 151]
        }
    
    def extract_facial_landmarks(self, image: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """
        Extract facial landmarks from image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of (x, y) landmark coordinates or None if no face detected
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image with face mesh
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                landmarks = []
                face_landmarks = results.multi_face_landmarks[0]
                
                height, width = image.shape[:2]
                
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    landmarks.append((x, y))
                
                return landmarks
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting facial landmarks: {e}")
            return None
    
    def calculate_facial_features(self, landmarks: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Calculate facial features from landmarks
        
        Args:
            landmarks: List of facial landmark coordinates
            
        Returns:
            Dictionary of calculated facial features
        """
        try:
            features = {}
            
            if not landmarks or len(landmarks) < 468:  # MediaPipe face mesh has 468 landmarks
                return features
            
            # Calculate eye aspect ratios (EAR) for blink detection and alertness
            def eye_aspect_ratio(eye_landmarks):
                # Calculate distances between eye landmarks
                if len(eye_landmarks) < 6:
                    return 0.0
                
                # Vertical distances
                A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
                B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
                
                # Horizontal distance
                C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
                
                # Eye aspect ratio
                ear = (A + B) / (2.0 * C)
                return ear
            
            # Get eye landmarks
            left_eye_points = [landmarks[i] for i in self.landmark_indices['left_eye'][:6]]
            right_eye_points = [landmarks[i] for i in self.landmark_indices['right_eye'][:6]]
            
            if len(left_eye_points) >= 6 and len(right_eye_points) >= 6:
                features['left_eye_ar'] = eye_aspect_ratio(left_eye_points)
                features['right_eye_ar'] = eye_aspect_ratio(right_eye_points)
                features['avg_eye_ar'] = (features['left_eye_ar'] + features['right_eye_ar']) / 2
            
            # Calculate mouth aspect ratio for smile/frown detection
            mouth_points = [landmarks[i] for i in self.landmark_indices['mouth'][:8]]
            if len(mouth_points) >= 8:
                # Vertical mouth opening
                mouth_height = np.linalg.norm(np.array(mouth_points[2]) - np.array(mouth_points[6]))
                # Horizontal mouth width
                mouth_width = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[4]))
                
                if mouth_width > 0:
                    features['mouth_ar'] = mouth_height / mouth_width
                else:
                    features['mouth_ar'] = 0.0
                
                # Mouth corners - smile detection
                left_mouth_corner = mouth_points[0]
                right_mouth_corner = mouth_points[4]
                mouth_center = mouth_points[2]
                
                # Calculate if corners are raised (smile) or lowered (frown)
                left_corner_height = left_mouth_corner[1] - mouth_center[1]
                right_corner_height = right_mouth_corner[1] - mouth_center[1]
                features['mouth_corner_avg'] = (left_corner_height + right_corner_height) / 2
            
            # Calculate eyebrow position for surprise/anger detection
            eyebrow_points = [landmarks[i] for i in self.landmark_indices['eyebrows'][:10]]
            if len(eyebrow_points) >= 10:
                # Average eyebrow height relative to eye
                eyebrow_y = np.mean([point[1] for point in eyebrow_points])
                eye_y = np.mean([point[1] for point in left_eye_points + right_eye_points])
                features['eyebrow_height'] = eye_y - eyebrow_y  # Positive means eyebrows are raised
            
            # Calculate face width/height ratio
            if len(landmarks) >= 468:
                face_top = min([point[1] for point in landmarks])
                face_bottom = max([point[1] for point in landmarks])
                face_left = min([point[0] for point in landmarks])
                face_right = max([point[0] for point in landmarks])
                
                face_height = face_bottom - face_top
                face_width = face_right - face_left
                
                if face_height > 0:
                    features['face_ratio'] = face_width / face_height
                else:
                    features['face_ratio'] = 1.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating facial features: {e}")
            return {}
    
    def analyze_stress_indicators(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze facial stress indicators
        
        Args:
            features: Calculated facial features
            
        Returns:
            Stress analysis results
        """
        stress_indicators = {}
        
        # Low eye aspect ratio can indicate fatigue or stress
        avg_eye_ar = features.get('avg_eye_ar', 0.3)
        if avg_eye_ar < 0.25:  # Normal range is around 0.3
            stress_indicators['eye_fatigue'] = (0.25 - avg_eye_ar) / 0.1
        else:
            stress_indicators['eye_fatigue'] = 0.0
        
        # Tense mouth (small mouth opening)
        mouth_ar = features.get('mouth_ar', 0.5)
        if mouth_ar < 0.3:
            stress_indicators['mouth_tension'] = (0.3 - mouth_ar) / 0.2
        else:
            stress_indicators['mouth_tension'] = 0.0
        
        # Frowning (negative mouth corner average)
        mouth_corner = features.get('mouth_corner_avg', 0)
        if mouth_corner < -2:  # Negative indicates downward corners
            stress_indicators['frowning'] = min(1.0, abs(mouth_corner) / 10)
        else:
            stress_indicators['frowning'] = 0.0
        
        # Raised eyebrows can indicate worry or surprise
        eyebrow_height = features.get('eyebrow_height', 0)
        if eyebrow_height > 5:  # Raised eyebrows
            stress_indicators['eyebrow_tension'] = min(1.0, eyebrow_height / 20)
        else:
            stress_indicators['eyebrow_tension'] = 0.0
        
        # Calculate overall stress level
        stress_level = np.mean(list(stress_indicators.values()))
        stress_indicators['overall_stress'] = stress_level
        
        return stress_indicators
    
    def classify_emotion(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify emotion based on facial features
        
        Args:
            features: Calculated facial features
            
        Returns:
            Tuple of (emotion, confidence)
        """
        try:
            emotion_scores = {}
            
            mouth_corner = features.get('mouth_corner_avg', 0)
            mouth_ar = features.get('mouth_ar', 0.5)
            eyebrow_height = features.get('eyebrow_height', 0)
            avg_eye_ar = features.get('avg_eye_ar', 0.3)
            
            # Happy: Raised mouth corners, normal/wide mouth opening
            emotion_scores['happy'] = max(0, mouth_corner / 10) * 0.6 + max(0, (mouth_ar - 0.4) / 0.3) * 0.4
            
            # Sad: Lowered mouth corners, potentially lowered eyebrows
            emotion_scores['sad'] = max(0, -mouth_corner / 10) * 0.7 + max(0, -eyebrow_height / 10) * 0.3
            
            # Angry: Lowered eyebrows, tense mouth
            emotion_scores['angry'] = max(0, -eyebrow_height / 15) * 0.5 + max(0, (0.3 - mouth_ar) / 0.2) * 0.5
            
            # Surprised: Raised eyebrows, wide eyes, open mouth
            emotion_scores['surprised'] = (
                max(0, eyebrow_height / 20) * 0.4 +
                max(0, (avg_eye_ar - 0.3) / 0.2) * 0.3 +
                max(0, (mouth_ar - 0.6) / 0.4) * 0.3
            )
            
            # Fear: Similar to surprised but with more tension
            emotion_scores['fear'] = (
                emotion_scores['surprised'] * 0.6 +
                max(0, (0.25 - avg_eye_ar) / 0.1) * 0.4
            )
            
            # Disgust: Lowered mouth corners, raised upper lip
            emotion_scores['disgust'] = max(0, -mouth_corner / 15) * 0.8 + max(0, (0.2 - mouth_ar) / 0.2) * 0.2
            
            # Neutral: Default when other scores are low
            emotion_scores['neutral'] = 1.0 - max(emotion_scores.values()) if emotion_scores else 1.0
            
            # Normalize scores
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
            logger.error(f"Error classifying facial emotion: {e}")
            return 'neutral', 0.5
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process image and return facial emotion analysis
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing facial analysis results
        """
        try:
            # Extract landmarks
            landmarks = self.extract_facial_landmarks(image)
            
            if not landmarks:
                return {
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'stress_level': 0.0,
                    'features': {},
                    'stress_indicators': {},
                    'concerns': ['No face detected'],
                    'face_detected': False
                }
            
            # Calculate features
            features = self.calculate_facial_features(landmarks)
            
            if not features:
                return {
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'stress_level': 0.0,
                    'features': {},
                    'stress_indicators': {},
                    'concerns': ['Could not extract facial features'],
                    'face_detected': True
                }
            
            # Classify emotion
            emotion, confidence = self.classify_emotion(features)
            
            # Analyze stress
            stress_indicators = self.analyze_stress_indicators(features)
            stress_level = stress_indicators.get('overall_stress', 0.0)
            
            # Identify concerns
            concerns = []
            if stress_level > 0.7:
                concerns.append("High facial stress indicators")
            if emotion in ['angry', 'fear', 'sad']:
                concerns.append(f"Negative facial emotion: {emotion}")
            if features.get('avg_eye_ar', 0.3) < 0.2:
                concerns.append("Signs of fatigue detected")
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'stress_level': stress_level,
                'features': features,
                'stress_indicators': stress_indicators,
                'concerns': concerns,
                'face_detected': True
            }
            
        except Exception as e:
            logger.error(f"Error processing facial image: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'stress_level': 0.0,
                'features': {},
                'stress_indicators': {},
                'concerns': ['Error processing image'],
                'face_detected': False
            }
    
    def process_video_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process single video frame for real-time analysis
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            Facial analysis results for the frame
        """
        return self.process_image(frame)