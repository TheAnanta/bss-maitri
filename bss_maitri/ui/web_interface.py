"""
Gradio-based web interface for BSS Maitri AI Assistant
"""

import gradio as gr
import numpy as np
import cv2
import logging
from typing import Optional, Tuple, Dict, Any
import time
from datetime import datetime

from ..models.ollama_client import OllamaClient
from ..utils.multimodal_analyzer import MultimodalEmotionAnalyzer

logger = logging.getLogger(__name__)


class MaitriWebInterface:
    """Web interface for the Maitri AI Assistant"""
    
    def __init__(self):
        """Initialize the web interface"""
        self.ollama_client = OllamaClient()
        self.analyzer = MultimodalEmotionAnalyzer()
        self.conversation_history = []
        
        # Check if Ollama model is available
        self.model_ready = False
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check if Ollama model is available and ready"""
        try:
            if self.ollama_client.ensure_model():
                self.model_ready = True
                logger.info("Ollama model is ready")
            else:
                logger.warning("Ollama model not available")
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
    
    def process_audio_video(self, audio_data: Optional[Tuple], 
                           video_frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Process audio and video inputs
        
        Args:
            audio_data: Tuple of (sample_rate, audio_array) or None
            video_frame: Video frame as numpy array or None
            
        Returns:
            Analysis results
        """
        try:
            # Process audio
            audio_array = None
            audio_sr = None
            if audio_data is not None and len(audio_data) == 2:
                audio_sr, audio_array = audio_data
                if isinstance(audio_array, np.ndarray) and len(audio_array) > 0:
                    # Convert to mono if stereo
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)
                else:
                    audio_array = None
            
            # Process video frame
            if video_frame is not None and not isinstance(video_frame, np.ndarray):
                video_frame = None
            
            # Run multimodal analysis
            results = self.analyzer.process_multimodal(
                audio_data=audio_array,
                video_frame=video_frame,
                audio_sample_rate=audio_sr
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing audio/video: {e}")
            return {
                'error': str(e),
                'overall_assessment': {
                    'current_state': 'neutral',
                    'stress_level': 0.0,
                    'requires_attention': False
                }
            }
    
    def get_ai_response(self, analysis_results: Dict[str, Any], 
                       user_message: str = "") -> str:
        """
        Get AI response based on analysis results
        
        Args:
            analysis_results: Results from multimodal analysis
            user_message: Optional user message
            
        Returns:
            AI response string
        """
        try:
            if not self.model_ready:
                return "I'm currently initializing. Please wait a moment while I get ready to assist you."
            
            # Prepare emotion data for the AI
            overall = analysis_results.get('overall_assessment', {})
            combined = analysis_results.get('combined_analysis', {})
            
            emotion_data = {
                'audio_emotion': combined.get('audio_emotion', 'unknown'),
                'audio_confidence': combined.get('audio_confidence', 0),
                'facial_emotion': combined.get('video_emotion', 'unknown'), 
                'facial_confidence': combined.get('video_confidence', 0),
                'stress_level': overall.get('stress_level', 0),
                'concerns': combined.get('concerns', [])
            }
            
            if user_message:
                # User provided message - respond conversationally
                self.conversation_history.append(f"User: {user_message}")
                
                response = self.ollama_client.provide_companionship(
                    self.conversation_history,
                    overall.get('current_state', 'neutral')
                )
                
                self.conversation_history.append(f"Maitri: {response}")
                
                # Keep conversation history manageable
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-10:]
                
            else:
                # No user message - provide emotion-based response
                response = self.ollama_client.analyze_emotion_context(emotion_data)
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "I'm having some technical difficulties right now, but I'm here for you. How are you feeling?"
    
    def create_interface(self) -> gr.Blocks:
        """Create and return the Gradio interface"""
        
        with gr.Blocks(
            title="BSS Maitri - AI Assistant for Crew Well-being",
            theme=gr.themes.Soft()
        ) as interface:
            
            gr.Markdown("""
            # 🚀 BSS Maitri - AI Assistant for Crew Well-being
            
            Welcome to Maitri, your AI companion designed to support emotional and physical well-being during space missions.
            I use on-device AI to analyze your voice and facial expressions, providing personalized support and companionship.
            
            **Features:**
            - 🎤 Voice emotion analysis
            - 📹 Facial expression recognition  
            - 🤝 Psychological companionship
            - 🔒 Complete privacy (runs on-device)
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Audio Input")
                    audio_input = gr.Audio(
                        source="microphone",
                        type="numpy",
                        label="Speak to Maitri"
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Video Input")
                    video_input = gr.Image(
                        source="webcam",
                        type="numpy",
                        label="Video feed for facial analysis"
                    )
            
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze My State", variant="primary")
                clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Conversation with Maitri")
                    user_message = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Your message (optional)",
                        lines=2
                    )
                    ai_response = gr.Textbox(
                        label="Maitri's Response",
                        lines=4,
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Analysis Results")
                    emotion_output = gr.Textbox(
                        label="Detected Emotion",
                        interactive=False
                    )
                    stress_output = gr.Slider(
                        label="Stress Level",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        interactive=False
                    )
                    concerns_output = gr.Textbox(
                        label="Concerns Detected",
                        lines=3,
                        interactive=False
                    )
            
            # Status indicators
            with gr.Row():
                model_status = gr.Textbox(
                    value="🟢 Ollama Model Ready" if self.model_ready else "🔴 Ollama Model Not Ready",
                    label="System Status",
                    interactive=False
                )
            
            def analyze_and_respond(audio, video, user_msg):
                """Process inputs and generate response"""
                try:
                    # Process multimodal input
                    results = self.process_audio_video(audio, video)
                    
                    # Get AI response
                    response = self.get_ai_response(results, user_msg)
                    
                    # Extract display information
                    overall = results.get('overall_assessment', {})
                    combined = results.get('combined_analysis', {})
                    
                    emotion = overall.get('current_state', 'neutral')
                    stress = overall.get('stress_level', 0.0)
                    concerns = combined.get('concerns', [])
                    concerns_text = '\n'.join(concerns) if concerns else "No concerns detected"
                    
                    return response, emotion, stress, concerns_text
                    
                except Exception as e:
                    logger.error(f"Error in analysis: {e}")
                    return (
                        "I encountered an error while analyzing. Please try again.",
                        "error",
                        0.0,
                        f"Error: {str(e)}"
                    )
            
            def clear_conversation():
                """Clear conversation history"""
                self.conversation_history.clear()
                return "", "", 0.0, ""
            
            # Event handlers
            analyze_btn.click(
                fn=analyze_and_respond,
                inputs=[audio_input, video_input, user_message],
                outputs=[ai_response, emotion_output, stress_output, concerns_output]
            )
            
            clear_btn.click(
                fn=clear_conversation,
                outputs=[ai_response, emotion_output, stress_output, concerns_output]
            )
            
            # Auto-analyze when audio/video is provided
            audio_input.change(
                fn=analyze_and_respond,
                inputs=[audio_input, video_input, gr.Textbox(value="", visible=False)],
                outputs=[ai_response, emotion_output, stress_output, concerns_output]
            )
            
            gr.Markdown("""
            ---
            **Privacy Notice:** All processing happens on your device. No data is sent to external servers.
            
            **Usage Tips:**
            - Speak naturally for a few seconds for better audio analysis
            - Ensure good lighting for facial analysis
            - You can chat with Maitri by typing messages
            - The system analyzes your emotional state and provides supportive responses
            """)
        
        return interface
    
    def launch(self, share: bool = False, port: int = 7860):
        """Launch the web interface"""
        interface = self.create_interface()
        
        logger.info(f"Launching BSS Maitri web interface on port {port}")
        
        interface.launch(
            share=share,
            server_port=port,
            server_name="0.0.0.0" if share else "127.0.0.1",
            show_error=True
        )