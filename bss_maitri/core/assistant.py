"""
Main BSS Maitri AI Assistant that orchestrates all components
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

from ..config import Config
from ..models.gemma_model import GemmaModel
from .emotion_detector import EmotionDetector, EmotionResult
from .conversation_engine import ConversationEngine

logger = logging.getLogger(__name__)

class MaitriAssistant:
    """
    Main Maitri AI Assistant for Bharatiya Space Station
    
    Provides multimodal emotion detection and psychological support
    for crew members through audio-visual analysis and AI conversation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Maitri Assistant"""
        
        # Load configuration
        self.config = Config(config_path)
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Initializing BSS Maitri AI Assistant...")
        
        # Initialize components
        self._initialize_components()
        
        # State tracking
        self.session_start_time = datetime.now()
        self.total_interactions = 0
        self.critical_alerts_sent = 0
        self.last_health_check = datetime.now()
        
        # Data storage
        self.emotion_log = []
        self.conversation_log = []
        self.alert_log = []
        
        logger.info("Maitri Assistant initialized successfully!")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.system.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_components(self):
        """Initialize all AI components"""
        
        try:
            # Initialize Gemma model
            logger.info("Loading Gemma model...")
            self.gemma_model = GemmaModel(self.config.model)
            
            # Initialize emotion detector
            logger.info("Initializing emotion detection...")
            self.emotion_detector = EmotionDetector(self.config.emotion_detection)
            
            # Initialize conversation engine
            logger.info("Initializing conversation engine...")
            self.conversation_engine = ConversationEngine(
                self.config.conversation, 
                self.gemma_model
            )
            
            logger.info("All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def process_interaction(self, 
                          user_input: Optional[str] = None,
                          audio_data: Optional[np.ndarray] = None,
                          frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process a complete interaction with multimodal input
        
        Args:
            user_input: Text input from crew member
            audio_data: Audio data for emotion detection  
            frame: Video frame for facial emotion detection
            
        Returns:
            Dictionary containing response and analysis results
        """
        
        try:
            logger.debug("Processing new interaction...")
            
            # Detect emotions from multimodal input
            emotion_result = None
            if audio_data is not None or frame is not None:
                emotion_result = self.emotion_detector.detect_emotions(
                    audio_data=audio_data,
                    frame=frame
                )
                
                # Log emotion detection
                self._log_emotion_result(emotion_result)
            
            # Generate conversational response
            response = None
            if user_input:
                response = self.conversation_engine.process_user_input(
                    user_input, 
                    emotion_result
                )
                
                # Log conversation
                self._log_conversation(user_input, response, emotion_result)
            
            # Check for critical states and alerts
            alerts = self._check_critical_states(emotion_result)
            
            # Update interaction count
            self.total_interactions += 1
            
            # Prepare response
            result = {
                'response': response,
                'emotion_analysis': {
                    'emotions': emotion_result.emotions if emotion_result else {},
                    'dominant_emotion': emotion_result.dominant_emotion if emotion_result else 'unknown',
                    'confidence': emotion_result.confidence if emotion_result else 0.0,
                    'modalities_used': emotion_result.modalities if emotion_result else []
                },
                'alerts': alerts,
                'timestamp': datetime.now().isoformat(),
                'interaction_id': self.total_interactions
            }
            
            logger.info(f"Interaction {self.total_interactions} processed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing interaction: {e}")
            return {
                'response': "I'm having technical difficulties. Please contact ground control if you need immediate assistance.",
                'emotion_analysis': {'emotions': {}, 'dominant_emotion': 'unknown', 'confidence': 0.0},
                'alerts': [],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def process_audio_input(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Process audio-only input for emotion detection"""
        return self.process_interaction(audio_data=audio_data)
    
    def process_video_input(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process video-only input for emotion detection"""
        return self.process_interaction(frame=frame)
    
    def process_text_input(self, user_input: str) -> Dict[str, Any]:
        """Process text-only input for conversation"""
        return self.process_interaction(user_input=user_input)
    
    def process_multimodal_input(self, 
                                user_input: str,
                                audio_data: Optional[np.ndarray] = None,
                                frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process complete multimodal input"""
        return self.process_interaction(
            user_input=user_input,
            audio_data=audio_data, 
            frame=frame
        )
    
    def _log_emotion_result(self, emotion_result: EmotionResult):
        """Log emotion detection result"""
        
        log_entry = {
            'timestamp': emotion_result.timestamp.isoformat(),
            'emotions': emotion_result.emotions,
            'dominant_emotion': emotion_result.dominant_emotion,
            'confidence': emotion_result.confidence,
            'modalities': emotion_result.modalities
        }
        
        self.emotion_log.append(log_entry)
        
        # Keep only recent entries (last 1000)
        if len(self.emotion_log) > 1000:
            self.emotion_log = self.emotion_log[-1000:]
    
    def _log_conversation(self, user_input: str, response: str, emotion_result: Optional[EmotionResult]):
        """Log conversation interaction"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'assistant_response': response,
            'emotion_context': emotion_result.emotions if emotion_result else {}
        }
        
        self.conversation_log.append(log_entry)
        
        # Keep only recent entries (last 500)
        if len(self.conversation_log) > 500:
            self.conversation_log = self.conversation_log[-500:]
    
    def _check_critical_states(self, emotion_result: Optional[EmotionResult]) -> List[Dict[str, Any]]:
        """Check for critical emotional states that require alerts"""
        
        alerts = []
        
        if not emotion_result:
            return alerts
        
        try:
            # Check for critical emotional states
            if self.emotion_detector.is_critical_state(emotion_result):
                
                alert = {
                    'type': 'critical_emotional_state',
                    'severity': 'high',
                    'dominant_emotion': emotion_result.dominant_emotion,
                    'confidence': emotion_result.confidence,
                    'timestamp': datetime.now().isoformat(),
                    'recommendation': 'Immediate psychological support recommended'
                }
                
                alerts.append(alert)
                self.critical_alerts_sent += 1
                
                # Log alert
                self.alert_log.append(alert)
                
                logger.warning(f"Critical emotional state detected: {emotion_result.dominant_emotion} "
                             f"(confidence: {emotion_result.confidence:.2f})")
            
            # Check for severe fatigue
            fatigue_level = emotion_result.emotions.get('fatigue', 0.0)
            if fatigue_level > 0.8:
                alert = {
                    'type': 'severe_fatigue',
                    'severity': 'high',
                    'fatigue_level': fatigue_level,
                    'timestamp': datetime.now().isoformat(),
                    'recommendation': 'Rest period recommended - consult with mission control'
                }
                
                alerts.append(alert)
                self.alert_log.append(alert)
                
                logger.warning(f"Severe fatigue detected: {fatigue_level:.2f}")
            
            # Check for sustained stress
            stress_level = emotion_result.emotions.get('stress', 0.0)
            if stress_level > 0.7:
                # Check if stress has been sustained
                recent_emotions = self.emotion_log[-5:] if len(self.emotion_log) >= 5 else self.emotion_log
                sustained_stress = all(
                    entry.get('emotions', {}).get('stress', 0.0) > 0.6 
                    for entry in recent_emotions
                )
                
                if sustained_stress:
                    alert = {
                        'type': 'sustained_stress',
                        'severity': 'medium',
                        'stress_level': stress_level,
                        'timestamp': datetime.now().isoformat(),
                        'recommendation': 'Stress management intervention recommended'
                    }
                    
                    alerts.append(alert)
                    self.alert_log.append(alert)
                    
                    logger.warning(f"Sustained stress detected: {stress_level:.2f}")
            
        except Exception as e:
            logger.error(f"Error checking critical states: {e}")
        
        return alerts
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health and status report"""
        
        try:
            # Update health check time
            self.last_health_check = datetime.now()
            
            # Get emotion summary
            emotion_summary = self.emotion_detector.get_emotion_summary()
            
            # Get conversation summary  
            conversation_summary = self.conversation_engine.get_conversation_summary()
            
            # Calculate session duration
            session_duration = datetime.now() - self.session_start_time
            
            # Recent alerts (last 24 hours)
            recent_alerts = [
                alert for alert in self.alert_log
                if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(hours=24)
            ]
            
            status = {
                'session_info': {
                    'start_time': self.session_start_time.isoformat(),
                    'duration_hours': session_duration.total_seconds() / 3600,
                    'total_interactions': self.total_interactions,
                    'critical_alerts_sent': self.critical_alerts_sent
                },
                'current_emotional_state': {
                    'summary': emotion_summary,
                    'dominant_emotion': max(emotion_summary.items(), key=lambda x: x[1])[0] if emotion_summary else 'unknown'
                },
                'conversation_status': conversation_summary,
                'recent_alerts_24h': len(recent_alerts),
                'alert_details': recent_alerts[-5:],  # Last 5 alerts
                'system_status': {
                    'gemma_model_loaded': hasattr(self, 'gemma_model') and self.gemma_model is not None,
                    'emotion_detector_active': hasattr(self, 'emotion_detector') and self.emotion_detector is not None,
                    'conversation_engine_active': hasattr(self, 'conversation_engine') and self.conversation_engine is not None
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error generating health status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def check_proactive_outreach(self) -> Optional[str]:
        """Check if proactive outreach is needed"""
        return self.conversation_engine.check_for_proactive_outreach()
    
    def save_session_data(self, filepath: Optional[str] = None) -> str:
        """Save session data to file"""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"maitri_session_{timestamp}.json"
        
        try:
            session_data = {
                'session_info': {
                    'start_time': self.session_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_interactions': self.total_interactions,
                    'critical_alerts_sent': self.critical_alerts_sent
                },
                'emotion_log': self.emotion_log,
                'conversation_log': self.conversation_log,
                'alert_log': self.alert_log,
                'config': {
                    'model_name': self.config.model.get('name'),
                    'emotion_detection_config': self.config.emotion_detection,
                    'conversation_config': self.config.conversation
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"Session data saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
            raise
    
    def load_session_data(self, filepath: str):
        """Load previous session data"""
        
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)
            
            # Restore logs
            self.emotion_log = session_data.get('emotion_log', [])
            self.conversation_log = session_data.get('conversation_log', [])
            self.alert_log = session_data.get('alert_log', [])
            
            # Restore counters
            session_info = session_data.get('session_info', {})
            self.total_interactions = session_info.get('total_interactions', 0)
            self.critical_alerts_sent = session_info.get('critical_alerts_sent', 0)
            
            logger.info(f"Session data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
            raise
    
    def shutdown(self):
        """Gracefully shutdown the assistant"""
        
        try:
            logger.info("Shutting down Maitri Assistant...")
            
            # Save final session data
            final_save_path = self.save_session_data()
            
            # Cleanup resources
            if hasattr(self, 'gemma_model'):
                del self.gemma_model
            
            logger.info(f"Maitri Assistant shutdown complete. Final data saved to {final_save_path}")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()