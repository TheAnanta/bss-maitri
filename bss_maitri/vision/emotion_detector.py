"""
Vision-based emotion detection using facial expressions for BSS Maitri
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VisionEmotionDetector:
    """Vision-based emotion detection using facial landmarks and expressions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.face_detection_confidence = config.get('face_detection_confidence', 0.5)
        self.emotion_threshold = config.get('emotion_threshold', 0.6)
        self.max_faces = config.get('max_faces', 1)
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=self.face_detection_confidence
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_faces,
            refine_landmarks=True,
            min_detection_confidence=self.face_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Define emotion categories
        self.emotion_labels = [
            'calm', 'stress', 'anxiety', 'fatigue',
            'sadness', 'anger', 'fear', 'joy', 'surprise'
        ]
        
        # Define key facial landmarks for emotion detection
        self._define_landmark_indices()
        
    def _define_landmark_indices(self):
        """Define facial landmark indices for different facial features"""
        # MediaPipe face mesh landmark indices
        self.landmark_indices = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [296, 334, 293, 300, 276, 283, 282, 295, 285, 336],
            'mouth': [0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 176, 148, 149, 150, 151, 152, 148],
            'mouth_outer': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 291, 328, 462, 457, 438, 424, 394, 418, 421, 351, 6],
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
    
    def extract_facial_landmarks(self, image: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """Extract facial landmarks from image"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                # Get first face landmarks
                face_landmarks = results.multi_face_landmarks[0]
                
                # Convert to list of (x, y) coordinates
                landmarks = []
                h, w = image.shape[:2]
                
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
                
                return landmarks
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting facial landmarks: {e}")
            return None
    
    def calculate_facial_features(self, landmarks: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate facial features for emotion detection"""
        try:
            features = {}
            
            if len(landmarks) < 468:  # MediaPipe face mesh has 468 landmarks
                return features
            
            # Eye aspect ratio (EAR) - fatigue/drowsiness indicator
            left_ear = self._calculate_eye_aspect_ratio(landmarks, 'left')
            right_ear = self._calculate_eye_aspect_ratio(landmarks, 'right')
            features['eye_aspect_ratio'] = (left_ear + right_ear) / 2.0
            
            # Mouth aspect ratio (MAR) - surprise/joy indicator
            features['mouth_aspect_ratio'] = self._calculate_mouth_aspect_ratio(landmarks)
            
            # Eyebrow distance - stress/anger indicator
            features['eyebrow_distance'] = self._calculate_eyebrow_distance(landmarks)
            
            # Mouth curvature - sadness/joy indicator
            features['mouth_curvature'] = self._calculate_mouth_curvature(landmarks)
            
            # Eye distance - focus/attention indicator
            features['eye_distance'] = self._calculate_eye_distance(landmarks)
            
            # Facial symmetry - overall emotional state
            features['facial_symmetry'] = self._calculate_facial_symmetry(landmarks)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating facial features: {e}")
            return {}
    
    def _calculate_eye_aspect_ratio(self, landmarks: List[Tuple[float, float]], eye_side: str) -> float:
        """Calculate Eye Aspect Ratio (EAR)"""
        try:
            if eye_side == 'left':
                eye_points = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
            else:  # right eye
                eye_points = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
            
            # Calculate vertical distances
            vertical_1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
            vertical_2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
            
            # Calculate horizontal distance
            horizontal = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
            
            # Calculate EAR
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
            
        except:
            return 0.2  # Default normal EAR value
    
    def _calculate_mouth_aspect_ratio(self, landmarks: List[Tuple[float, float]]) -> float:
        """Calculate Mouth Aspect Ratio (MAR)"""
        try:
            # Mouth landmarks
            mouth_points = [landmarks[i] for i in [61, 291, 39, 181, 0, 17]]
            
            # Calculate vertical distances
            vertical_1 = np.linalg.norm(np.array(mouth_points[2]) - np.array(mouth_points[4]))
            vertical_2 = np.linalg.norm(np.array(mouth_points[3]) - np.array(mouth_points[5]))
            
            # Calculate horizontal distance
            horizontal = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[1]))
            
            # Calculate MAR
            mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return mar
            
        except:
            return 0.1  # Default closed mouth value
    
    def _calculate_eyebrow_distance(self, landmarks: List[Tuple[float, float]]) -> float:
        """Calculate eyebrow distance from eyes"""
        try:
            # Left eyebrow and eye
            left_eyebrow = landmarks[70]  # Left eyebrow center
            left_eye = landmarks[159]     # Left eye center
            left_distance = np.linalg.norm(np.array(left_eyebrow) - np.array(left_eye))
            
            # Right eyebrow and eye
            right_eyebrow = landmarks[300]  # Right eyebrow center
            right_eye = landmarks[386]      # Right eye center
            right_distance = np.linalg.norm(np.array(right_eyebrow) - np.array(right_eye))
            
            return (left_distance + right_distance) / 2.0
            
        except:
            return 20.0  # Default distance
    
    def _calculate_mouth_curvature(self, landmarks: List[Tuple[float, float]]) -> float:
        """Calculate mouth curvature (smile/frown)"""
        try:
            # Mouth corners and center
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            mouth_center = landmarks[13]
            
            # Calculate if corners are above or below center
            left_curve = left_corner[1] - mouth_center[1]  # Negative if above (smile)
            right_curve = right_corner[1] - mouth_center[1]
            
            # Average curvature
            curvature = -(left_curve + right_curve) / 2.0  # Negative for smile, positive for frown
            return curvature
            
        except:
            return 0.0  # Neutral
    
    def _calculate_eye_distance(self, landmarks: List[Tuple[float, float]]) -> float:
        """Calculate distance between eyes"""
        try:
            left_eye_center = landmarks[159]
            right_eye_center = landmarks[386]
            
            distance = np.linalg.norm(np.array(left_eye_center) - np.array(right_eye_center))
            return distance
            
        except:
            return 60.0  # Default distance
    
    def _calculate_facial_symmetry(self, landmarks: List[Tuple[float, float]]) -> float:
        """Calculate facial symmetry"""
        try:
            # Compare left and right side landmarks
            symmetry_pairs = [
                (33, 362),    # Eye corners
                (61, 291),    # Mouth corners
                (126, 355),   # Face sides
            ]
            
            symmetry_scores = []
            face_center_x = landmarks[1][0]  # Nose tip x-coordinate
            
            for left_idx, right_idx in symmetry_pairs:
                left_point = landmarks[left_idx]
                right_point = landmarks[right_idx]
                
                # Calculate distance from center for both sides
                left_dist = abs(left_point[0] - face_center_x)
                right_dist = abs(right_point[0] - face_center_x)
                
                # Calculate symmetry ratio
                if max(left_dist, right_dist) > 0:
                    symmetry = min(left_dist, right_dist) / max(left_dist, right_dist)
                    symmetry_scores.append(symmetry)
            
            return np.mean(symmetry_scores) if symmetry_scores else 1.0
            
        except:
            return 1.0  # Perfect symmetry default
    
    def detect_emotion_from_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Detect emotions from facial features"""
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_labels}
        
        try:
            ear = features.get('eye_aspect_ratio', 0.2)
            mar = features.get('mouth_aspect_ratio', 0.1)
            eyebrow_dist = features.get('eyebrow_distance', 20.0)
            mouth_curve = features.get('mouth_curvature', 0.0)
            symmetry = features.get('facial_symmetry', 1.0)
            
            # Fatigue detection (low EAR)
            if ear < 0.15:
                emotion_scores['fatigue'] += 0.5
            
            # Surprise detection (high MAR)
            if mar > 0.3:
                emotion_scores['surprise'] += 0.4
            
            # Joy detection (positive mouth curvature + normal EAR)
            if mouth_curve > 5 and ear > 0.18:
                emotion_scores['joy'] += 0.6
            
            # Sadness detection (negative mouth curvature + low EAR)
            if mouth_curve < -3 and ear < 0.18:
                emotion_scores['sadness'] += 0.4
            
            # Stress/Anger detection (low eyebrow distance + asymmetry)
            if eyebrow_dist < 15 and symmetry < 0.9:
                emotion_scores['stress'] += 0.3
                emotion_scores['anger'] += 0.2
            
            # Anxiety detection (asymmetry + normal other features)
            if symmetry < 0.85:
                emotion_scores['anxiety'] += 0.3
            
            # Fear detection (high eyebrow + wide eyes)
            if eyebrow_dist > 25 and ear > 0.25:
                emotion_scores['fear'] += 0.4
            
            # Calm baseline
            total_negative = sum(emotion_scores[e] for e in ['stress', 'anxiety', 'fatigue', 'sadness', 'anger', 'fear'])
            if total_negative < 0.3:
                emotion_scores['calm'] = max(0.4, 1.0 - total_negative)
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            else:
                emotion_scores['calm'] = 1.0
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            return {'calm': 1.0}
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Process a single frame and return emotion predictions"""
        try:
            # Extract facial landmarks
            landmarks = self.extract_facial_landmarks(frame)
            
            if landmarks is None:
                logger.debug("No face detected in frame")
                return {'calm': 1.0}
            
            # Calculate facial features
            features = self.calculate_facial_features(landmarks)
            
            if not features:
                logger.warning("No features calculated from landmarks")
                return {'calm': 1.0}
            
            # Detect emotions
            emotions = self.detect_emotion_from_features(features)
            
            logger.debug(f"Vision emotion detection results: {emotions}")
            return emotions
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {'calm': 1.0}
    
    def get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """Get the dominant emotion from scores"""
        if not emotion_scores:
            return 'calm', 1.0
        
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return dominant_emotion
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()