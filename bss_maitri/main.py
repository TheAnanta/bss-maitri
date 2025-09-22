"""
Main entry point for BSS Maitri AI Assistant
"""

import argparse
import logging
import sys
from pathlib import Path
import cv2
import numpy as np

from bss_maitri import MaitriAssistant
from bss_maitri.utils import load_audio_file, validate_audio_input, validate_video_input

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('maitri.log')
        ]
    )

def run_interactive_mode(assistant: MaitriAssistant):
    """Run interactive text-based mode"""
    print("\n🚀 BSS Maitri AI Assistant - Interactive Mode")
    print("Type 'quit' to exit, 'status' for health status")
    print("=" * 50)
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Maitri: Take care! Stay safe up there! 🛰️")
                break
            
            if user_input.lower() == 'status':
                status = assistant.get_health_status()
                print(f"\nSystem Status:")
                print(f"- Session Duration: {status['session_info']['duration_hours']:.1f} hours")
                print(f"- Total Interactions: {status['session_info']['total_interactions']}")
                print(f"- Current Emotion: {status['current_emotional_state']['dominant_emotion']}")
                print(f"- Recent Alerts: {status['recent_alerts_24h']}")
                continue
            
            if not user_input:
                continue
            
            # Process the input
            result = assistant.process_text_input(user_input)
            
            # Display response
            print(f"Maitri: {result['response']}")
            
            # Show alerts if any
            if result['alerts']:
                print("\n⚠️ Alerts:")
                for alert in result['alerts']:
                    print(f"  - {alert['type']}: {alert['recommendation']}")
    
    except KeyboardInterrupt:
        print("\n\nMaitri: Goodbye! Stay safe! 🛰️")
    except Exception as e:
        logger.error(f"Error in interactive mode: {e}")
        print(f"Error: {e}")

def run_audio_demo(assistant: MaitriAssistant, audio_file: str):
    """Run audio emotion detection demo"""
    print(f"\n🎵 Processing audio file: {audio_file}")
    
    try:
        # Load audio
        audio_data, sr = load_audio_file(audio_file)
        
        if not validate_audio_input(audio_data):
            print("❌ Invalid audio input")
            return
        
        # Process audio
        result = assistant.process_audio_input(audio_data)
        
        # Display results
        emotions = result['emotion_analysis']['emotions']
        dominant = result['emotion_analysis']['dominant_emotion']
        confidence = result['emotion_analysis']['confidence']
        
        print(f"✅ Audio Analysis Complete!")
        print(f"Dominant Emotion: {dominant} (confidence: {confidence:.2f})")
        print("All Emotions:")
        for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {score:.3f}")
        
        # Show alerts
        if result['alerts']:
            print("\n⚠️ Alerts:")
            for alert in result['alerts']:
                print(f"  - {alert['type']}: {alert['recommendation']}")
    
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        print(f"❌ Error: {e}")

def run_video_demo(assistant: MaitriAssistant, video_source):
    """Run video emotion detection demo"""
    print(f"\n📹 Processing video from: {video_source}")
    
    try:
        # Open video source
        if video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
        else:
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ Could not open video source")
            return
        
        print("📹 Video capture started. Press 'q' to quit, 's' for status")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 30th frame to reduce computation
            if frame_count % 30 == 0:
                if validate_video_input(frame):
                    result = assistant.process_video_input(frame)
                    
                    # Overlay emotion info on frame
                    emotions = result['emotion_analysis']['emotions']
                    dominant = result['emotion_analysis']['dominant_emotion']
                    confidence = result['emotion_analysis']['confidence']
                    
                    # Display emotion info
                    cv2.putText(frame, f"Emotion: {dominant} ({confidence:.2f})", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show top 3 emotions
                    y_offset = 70
                    for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]:
                        cv2.putText(frame, f"{emotion}: {score:.2f}", 
                                  (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        y_offset += 25
                    
                    # Show alerts
                    if result['alerts']:
                        cv2.putText(frame, "⚠️ ALERT", (10, frame.shape[0] - 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('BSS Maitri - Emotion Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                status = assistant.get_health_status()
                print(f"\nStatus: {status['current_emotional_state']['dominant_emotion']} | "
                      f"Interactions: {status['session_info']['total_interactions']}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("📹 Video demo completed")
    
    except Exception as e:
        logger.error(f"Error in video demo: {e}")
        print(f"❌ Error: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="BSS Maitri AI Assistant")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', choices=['interactive', 'audio', 'video'], 
                       default='interactive', help='Operating mode')
    parser.add_argument('--audio-file', type=str, help='Audio file for audio mode')
    parser.add_argument('--video-source', type=str, default='0', 
                       help='Video source (file path or camera index)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("🚀 Initializing BSS Maitri AI Assistant...")
    
    try:
        # Initialize assistant
        with MaitriAssistant(args.config) as assistant:
            print("✅ Maitri Assistant ready!")
            
            if args.mode == 'interactive':
                run_interactive_mode(assistant)
            elif args.mode == 'audio':
                if not args.audio_file:
                    print("❌ Audio file required for audio mode (--audio-file)")
                    return
                run_audio_demo(assistant, args.audio_file)
            elif args.mode == 'video':
                run_video_demo(assistant, args.video_source)
            
            print("\n📊 Final Status:")
            status = assistant.get_health_status()
            print(f"- Session Duration: {status['session_info']['duration_hours']:.1f} hours")
            print(f"- Total Interactions: {status['session_info']['total_interactions']}")
            print(f"- Critical Alerts: {status['session_info']['critical_alerts_sent']}")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()