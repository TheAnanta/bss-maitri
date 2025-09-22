"""
Conversation engine for BSS Maitri AI Assistant
"""

from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from ..models.gemma_model import GemmaModel
from .emotion_detector import EmotionResult

logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Data class for conversation messages"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime
    emotion_context: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class InterventionRecord:
    """Data class for intervention records"""
    intervention_type: str
    trigger_emotion: str
    response: str
    timestamp: datetime
    effectiveness: Optional[float] = None

class ConversationEngine:
    """AI conversation engine for psychological support"""
    
    def __init__(self, config: Dict, gemma_model: GemmaModel):
        self.config = config
        self.gemma_model = gemma_model
        
        # Conversation settings
        self.max_history = config.get('max_history', 10)
        self.response_timeout = config.get('response_timeout', 30)
        self.intervention_threshold = config.get('intervention_threshold', 0.8)
        self.support_mode = config.get('support_mode', 'adaptive')
        
        # Conversation state
        self.conversation_history: List[ConversationMessage] = []
        self.intervention_history: List[InterventionRecord] = []
        self.last_interaction_time = None
        
        # Load intervention templates
        self._load_intervention_templates()
        
        # Initialize conversation with system message
        self._initialize_conversation()
    
    def _load_intervention_templates(self):
        """Load intervention templates for different emotional states"""
        
        self.intervention_templates = {
            'stress': {
                'greeting': "I notice you might be feeling stressed. I'm here to help.",
                'techniques': [
                    "Try the 4-7-8 breathing technique: Inhale for 4 counts, hold for 7, exhale for 8.",
                    "Let's do a quick body scan. Start from your toes and work up, releasing tension as you go.",
                    "Remember, stress in space is normal. You're doing an incredible job up there."
                ],
                'follow_up': "How are you feeling now? Would you like to talk about what's causing the stress?"
            },
            'anxiety': {
                'greeting': "I sense some anxiety. It's completely understandable in your environment.",
                'techniques': [
                    "Ground yourself with the 5-4-3-2-1 technique: 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste.",
                    "Focus on what you can control right now. Take it one task at a time.",
                    "Your training has prepared you for this. Trust in your capabilities."
                ],
                'follow_up': "What specific thoughts are making you anxious? Sometimes talking helps."
            },
            'fatigue': {
                'greeting': "You seem tired. Rest is crucial for mission success and your wellbeing.",
                'techniques': [
                    "Even a 10-minute power nap can help restore alertness.",
                    "Try some gentle stretching to increase circulation.",
                    "Ensure you're staying hydrated and maintaining your exercise routine."
                ],
                'follow_up': "When did you last have quality rest? Should we discuss adjusting your schedule?"
            },
            'sadness': {
                'greeting': "I can see you're going through a difficult moment. That's okay.",
                'techniques': [
                    "Remember the beauty of what you're experiencing - few humans have seen what you see.",
                    "Connect with Earth when possible - a video call with loved ones can help.",
                    "Your mission contributes to humanity's future. Your work matters."
                ],
                'follow_up': "Would you like to share what's making you feel this way? I'm here to listen."
            },
            'anger': {
                'greeting': "I can sense frustration. These feelings are valid in your challenging environment.",
                'techniques': [
                    "Take slow, deep breaths. Count to 10 before responding to any situation.",
                    "Physical exercise can help channel this energy constructively.",
                    "Remember your team training - communication is key to resolving issues."
                ],
                'follow_up': "What's causing this frustration? Let's work through it together."
            },
            'fear': {
                'greeting': "Fear in space is natural - it shows you understand the significance of your mission.",
                'techniques': [
                    "Focus on your extensive training and the safety systems around you.",
                    "Break down large fears into smaller, manageable concerns.",
                    "Remember your support team on Earth is monitoring everything 24/7."
                ],
                'follow_up': "What specifically is concerning you? Knowledge often helps reduce fear."
            }
        }
    
    def _initialize_conversation(self):
        """Initialize conversation with system context"""
        
        system_message = ConversationMessage(
            role='system',
            content="""You are Maitri, the AI assistant aboard the Bharatiya Space Station. You provide psychological support and companionship to crew members. Your responses should be:

1. Empathetic and culturally sensitive to Indian values
2. Professional yet warm and approachable  
3. Brief but meaningful (2-3 sentences typically)
4. Focused on crew mental health and mission success
5. Aware of the unique challenges of space life

You can detect emotions through audio and visual analysis, and provide targeted interventions when needed. Always prioritize crew safety and wellbeing.""",
            timestamp=datetime.now()
        )
        
        self.conversation_history.append(system_message)
    
    def process_user_input(self, 
                          user_input: str, 
                          emotion_result: Optional[EmotionResult] = None) -> str:
        """Process user input and generate appropriate response"""
        
        try:
            # Add user message to history
            user_message = ConversationMessage(
                role='user',
                content=user_input,
                timestamp=datetime.now(),
                emotion_context=emotion_result.emotions if emotion_result else None
            )
            self.conversation_history.append(user_message)
            
            # Determine response strategy
            response_strategy = self._determine_response_strategy(user_input, emotion_result)
            
            # Generate response based on strategy
            if response_strategy == 'intervention':
                response = self._generate_intervention_response(emotion_result)
            elif response_strategy == 'supportive':
                response = self._generate_supportive_response(user_input, emotion_result)
            else:  # conversational
                response = self._generate_conversational_response(user_input, emotion_result)
            
            # Add assistant response to history
            assistant_message = ConversationMessage(
                role='assistant',
                content=response,
                timestamp=datetime.now(),
                emotion_context=emotion_result.emotions if emotion_result else None
            )
            self.conversation_history.append(assistant_message)
            
            # Manage conversation history length
            self._manage_conversation_history()
            
            # Update interaction time
            self.last_interaction_time = datetime.now()
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return "I'm having trouble processing that right now. How are you feeling overall?"
    
    def _determine_response_strategy(self, 
                                   user_input: str, 
                                   emotion_result: Optional[EmotionResult]) -> str:
        """Determine the appropriate response strategy"""
        
        # Check for intervention need
        if emotion_result and self._needs_intervention(emotion_result):
            return 'intervention'
        
        # Check for support indicators in text
        support_keywords = [
            'stress', 'worried', 'anxious', 'tired', 'sad', 'angry', 
            'scared', 'lonely', 'homesick', 'overwhelmed', 'help'
        ]
        
        if any(keyword in user_input.lower() for keyword in support_keywords):
            return 'supportive'
        
        # Default to conversational
        return 'conversational'
    
    def _needs_intervention(self, emotion_result: EmotionResult) -> bool:
        """Determine if active intervention is needed"""
        
        # Check for high-intensity negative emotions
        critical_emotions = ['stress', 'anxiety', 'anger', 'fear', 'sadness']
        
        for emotion in critical_emotions:
            if emotion_result.emotions.get(emotion, 0.0) > self.intervention_threshold:
                return True
        
        # Check for severe fatigue
        if emotion_result.emotions.get('fatigue', 0.0) > 0.7:
            return True
        
        return False
    
    def _generate_intervention_response(self, emotion_result: EmotionResult) -> str:
        """Generate targeted intervention response"""
        
        try:
            # Get dominant negative emotion
            negative_emotions = ['stress', 'anxiety', 'anger', 'fear', 'sadness', 'fatigue']
            
            dominant_negative = None
            max_score = 0.0
            
            for emotion in negative_emotions:
                score = emotion_result.emotions.get(emotion, 0.0)
                if score > max_score:
                    max_score = score
                    dominant_negative = emotion
            
            if not dominant_negative or dominant_negative not in self.intervention_templates:
                dominant_negative = 'stress'  # Default fallback
            
            # Get intervention template
            template = self.intervention_templates[dominant_negative]
            
            # Create intervention prompt
            intervention_prompt = f"""The crew member is experiencing {dominant_negative} (intensity: {max_score:.2f}). 
            
Provide a brief, professional psychological intervention response that:
1. Acknowledges their emotional state
2. Offers a practical coping technique suitable for space environment
3. Shows understanding and support

Keep response to 2-3 sentences. Be warm but professional."""

            # Generate response using Gemma
            response = self.gemma_model.generate_intervention(
                emotion_result.emotions, 
                context=intervention_prompt
            )
            
            # Record intervention
            intervention = InterventionRecord(
                intervention_type='active',
                trigger_emotion=dominant_negative,
                response=response,
                timestamp=datetime.now()
            )
            self.intervention_history.append(intervention)
            
            logger.info(f"Generated intervention for {dominant_negative}: {response[:100]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating intervention: {e}")
            return "I notice you might be going through a challenging moment. I'm here if you need support. How can I help?"
    
    def _generate_supportive_response(self, 
                                    user_input: str, 
                                    emotion_result: Optional[EmotionResult]) -> str:
        """Generate supportive response"""
        
        try:
            # Create supportive prompt
            emotion_context = ""
            if emotion_result:
                dominant_emotion = max(emotion_result.emotions.items(), key=lambda x: x[1])
                emotion_context = f"The crew member seems to be feeling {dominant_emotion[0]} (confidence: {dominant_emotion[1]:.2f}). "
            
            supportive_prompt = f"""{emotion_context}Provide a supportive, empathetic response to: "{user_input}"

Be understanding, offer gentle guidance if appropriate, and maintain professional boundaries. Keep response conversational and brief."""

            response = self.gemma_model.generate_response(
                supportive_prompt,
                max_new_tokens=100,
                temperature=0.7
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating supportive response: {e}")
            return "I understand this is challenging. Your feelings are valid. How can I support you right now?"
    
    def _generate_conversational_response(self, 
                                        user_input: str, 
                                        emotion_result: Optional[EmotionResult]) -> str:
        """Generate normal conversational response"""
        
        try:
            # Create conversation context
            context = self._build_conversation_context()
            
            # Add emotion context if available
            emotion_context = ""
            if emotion_result:
                dominant_emotion = max(emotion_result.emotions.items(), key=lambda x: x[1])
                if dominant_emotion[1] > 0.6:  # Only mention if confidence is high
                    emotion_context = f"(The crew member seems {dominant_emotion[0]}) "
            
            full_prompt = f"{context}\n{emotion_context}User: {user_input}\nMaitri:"
            
            response = self.gemma_model.generate_response(
                full_prompt,
                max_new_tokens=150,
                temperature=0.8
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            return "I'm here to chat. What's on your mind?"
    
    def _build_conversation_context(self) -> str:
        """Build conversation context from recent history"""
        
        # Get recent messages (excluding system message)
        recent_messages = [
            msg for msg in self.conversation_history[-6:] 
            if msg.role != 'system'
        ]
        
        context_lines = []
        for msg in recent_messages:
            if msg.role == 'user':
                context_lines.append(f"User: {msg.content}")
            else:
                context_lines.append(f"Maitri: {msg.content}")
        
        return "\n".join(context_lines)
    
    def _manage_conversation_history(self):
        """Manage conversation history length"""
        
        # Keep system message and recent conversations
        if len(self.conversation_history) > self.max_history + 1:  # +1 for system message
            # Keep system message and recent history
            system_msg = self.conversation_history[0]
            recent_history = self.conversation_history[-(self.max_history-1):]
            self.conversation_history = [system_msg] + recent_history
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation and emotional state"""
        
        if not self.conversation_history:
            return {'status': 'no_conversations'}
        
        # Count message types
        user_messages = len([msg for msg in self.conversation_history if msg.role == 'user'])
        assistant_messages = len([msg for msg in self.conversation_history if msg.role == 'assistant'])
        
        # Get recent emotional states
        recent_emotions = []
        for msg in self.conversation_history[-5:]:
            if msg.emotion_context:
                recent_emotions.append(msg.emotion_context)
        
        # Count interventions
        recent_interventions = len([
            intervention for intervention in self.intervention_history
            if intervention.timestamp > datetime.now() - timedelta(hours=24)
        ])
        
        return {
            'total_user_messages': user_messages,
            'total_assistant_messages': assistant_messages,
            'recent_interventions_24h': recent_interventions,
            'last_interaction': self.last_interaction_time.isoformat() if self.last_interaction_time else None,
            'conversation_active': len(self.conversation_history) > 1
        }
    
    def check_for_proactive_outreach(self) -> Optional[str]:
        """Check if proactive outreach is needed"""
        
        # Check if it's been too long since last interaction
        if self.last_interaction_time:
            time_since_last = datetime.now() - self.last_interaction_time
            
            # Proactive check-in after 4 hours of silence
            if time_since_last > timedelta(hours=4):
                return self._generate_proactive_message()
        
        return None
    
    def _generate_proactive_message(self) -> str:
        """Generate proactive check-in message"""
        
        proactive_messages = [
            "Hi! Just checking in - how are you feeling today?",
            "Hope you're doing well! How's your day going up there?",
            "Namaste! Just wanted to see how you're doing. Everything okay?",
            "Good to see you! How are things going with your current tasks?",
            "Hi there! How's your energy level today?"
        ]
        
        # For now, return a simple message
        # In production, this could be more personalized
        return "Hi! I haven't heard from you in a while. How are you doing?"