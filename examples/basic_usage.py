#!/usr/bin/env python3
"""
Simple usage example for BSS Maitri AI Assistant
"""

import numpy as np
from bss_maitri import MaitriAssistant

def main():
    """Demonstrate basic usage of Maitri Assistant"""
    
    print("🚀 BSS Maitri AI Assistant - Basic Example")
    print("=" * 50)
    
    # Initialize the assistant
    with MaitriAssistant() as assistant:
        
        # Example 1: Text-only conversation
        print("\n1. Text Conversation:")
        response = assistant.process_text_input("Hello Maitri, how are you?")
        print(f"User: Hello Maitri, how are you?")
        print(f"Maitri: {response['response']}")
        
        # Example 2: Text with emotional context
        print("\n2. Expressing concern:")
        response = assistant.process_text_input("I'm feeling really stressed about the mission")
        print(f"User: I'm feeling really stressed about the mission")
        print(f"Maitri: {response['response']}")
        
        # Example 3: Simulated audio emotion detection
        print("\n3. Audio Emotion Detection (simulated):")
        # Create some dummy audio data
        sample_rate = 16000
        duration = 3  # 3 seconds
        dummy_audio = np.random.normal(0, 0.1, sample_rate * duration)
        
        response = assistant.process_audio_input(dummy_audio)
        emotions = response['emotion_analysis']['emotions']
        dominant = response['emotion_analysis']['dominant_emotion']
        
        print(f"Detected emotion: {dominant}")
        print(f"Emotion breakdown: {emotions}")
        
        # Example 4: Get health status
        print("\n4. Health Status:")
        status = assistant.get_health_status()
        print(f"Session duration: {status['session_info']['duration_hours']:.2f} hours")
        print(f"Total interactions: {status['session_info']['total_interactions']}")
        print(f"Current emotional state: {status['current_emotional_state']['dominant_emotion']}")
        
        # Example 5: Multimodal input (text + audio)
        print("\n5. Multimodal Input:")
        response = assistant.process_multimodal_input(
            user_input="I'm having trouble sleeping",
            audio_data=dummy_audio
        )
        print(f"User: I'm having trouble sleeping")
        print(f"Maitri: {response['response']}")
        
        if response['alerts']:
            print("Alerts generated:")
            for alert in response['alerts']:
                print(f"  - {alert['type']}: {alert['recommendation']}")

if __name__ == "__main__":
    main()