# BSS Maitri - Bharatiya Space Station AI Assistant

## Overview

BSS Maitri is a multimodal AI assistant designed for the Bharatiya Space Station to monitor crew emotional and physical well-being through audio-video inputs. The system uses the Gemma 3 language model as its foundation for psychological support and conversation.

## Features

- **Multimodal Emotion Detection**: Combines audio and visual cues to detect emotional states
- **AI-Powered Conversations**: Uses Gemma 3 model for empathetic and culturally-sensitive responses
- **Real-time Monitoring**: Continuous assessment of crew psychological state
- **Intervention System**: Automatic detection of critical emotional states with appropriate interventions
- **Offline Operation**: Designed to run standalone without internet connectivity
- **Cultural Sensitivity**: Tailored for Indian cultural values and space crew needs

## Emotion Detection Capabilities

### Audio Analysis
- Voice tone analysis
- Speech pattern recognition
- Stress indicators in speech
- Fatigue detection through vocal characteristics

### Visual Analysis
- Facial expression recognition
- Eye movement and blink patterns
- Facial symmetry analysis
- Fatigue detection through facial cues

### Supported Emotions
- Calm/Neutral
- Stress
- Anxiety
- Fatigue
- Sadness
- Anger
- Fear
- Joy
- Surprise

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- At least 8GB RAM (16GB recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/TheAnanta/bss-maitri.git
cd bss-maitri
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from bss_maitri import MaitriAssistant

# Initialize the assistant
with MaitriAssistant() as assistant:
    # Text conversation
    response = assistant.process_text_input("Hello Maitri, how are you?")
    print(response['response'])
    
    # Audio emotion detection
    audio_result = assistant.process_audio_input(audio_data)
    print(f"Detected emotion: {audio_result['emotion_analysis']['dominant_emotion']}")
    
    # Video emotion detection
    video_result = assistant.process_video_input(frame)
    print(f"Facial emotion: {video_result['emotion_analysis']['dominant_emotion']}")
```

### Command Line Interface

```bash
# Interactive text mode
python -m bss_maitri.main --mode interactive

# Audio analysis mode
python -m bss_maitri.main --mode audio --audio-file path/to/audio.wav

# Video analysis mode (webcam)
python -m bss_maitri.main --mode video --video-source 0

# Video analysis mode (file)
python -m bss_maitri.main --mode video --video-source path/to/video.mp4
```

## Configuration

The system can be configured through `bss_maitri/config/config.yaml`:

```yaml
model:
  name: "google/gemma-2-2b-it"
  device: "cuda"  # or "cpu"
  precision: "fp16"

emotion_detection:
  audio:
    sample_rate: 16000
    frame_length: 2048
  vision:
    face_detection_confidence: 0.5
    emotion_threshold: 0.6

conversation:
  max_history: 10
  intervention_threshold: 0.8
  support_mode: "adaptive"
```

## API Reference

### MaitriAssistant

Main class for the AI assistant.

#### Methods

- `process_text_input(user_input: str)` - Process text-only input
- `process_audio_input(audio_data: np.ndarray)` - Process audio for emotion detection
- `process_video_input(frame: np.ndarray)` - Process video frame for emotion detection
- `process_multimodal_input(user_input, audio_data, frame)` - Process combined inputs
- `get_health_status()` - Get comprehensive status report
- `save_session_data(filepath)` - Save session data to file

### EmotionDetector

Multimodal emotion detection system.

#### Methods

- `detect_emotions(audio_data, frame)` - Detect emotions from multimodal input
- `get_emotion_summary(time_window_minutes)` - Get emotion summary over time
- `is_critical_state(emotion_result)` - Check if intervention is needed

### ConversationEngine

AI conversation and intervention system.

#### Methods

- `process_user_input(user_input, emotion_result)` - Generate conversational response
- `generate_intervention(emotion_state, context)` - Generate targeted intervention
- `check_for_proactive_outreach()` - Check if proactive contact is needed

## Monitoring and Alerts

The system automatically monitors for:

- **Critical Emotional States**: High stress, severe anxiety, extreme sadness
- **Fatigue Levels**: Dangerous fatigue that could affect mission safety
- **Sustained Negative Emotions**: Prolonged periods of stress or anxiety
- **Communication Gaps**: Extended periods without crew interaction

Alerts are generated with recommendations for:
- Immediate psychological support
- Rest periods
- Communication with ground control
- Stress management techniques

## Data Privacy and Security

- All processing is done locally/offline
- No data transmitted to external servers
- Session data can be optionally saved for medical review
- Configurable data retention policies

## Examples

See the `examples/` directory for:
- `basic_usage.py` - Simple API usage examples
- `audio_demo.py` - Audio emotion detection demo
- `video_demo.py` - Video emotion detection demo
- `conversation_demo.py` - AI conversation examples

## Model Information

### Gemma 3 Integration

The system uses Google's Gemma 3 language model (specifically the 2B parameter instruction-tuned variant) as the foundation for:

- Natural language understanding
- Contextual response generation
- Psychological intervention strategies
- Cultural sensitivity in communication

### Performance Optimization

- Model quantization for reduced memory usage
- Optimized inference pipeline
- Caching strategies for repeated operations
- GPU acceleration where available

## Space Environment Considerations

The system is specifically designed for space station environments:

- **Low-latency Processing**: Real-time emotion detection and response
- **Resource Constraints**: Optimized for limited computational resources
- **Isolation Factors**: Addresses psychological challenges of space isolation
- **Mission Criticality**: Balances psychological support with mission requirements
- **Cultural Adaptation**: Incorporates Indian cultural values and communication styles

## Contributing

Please see CONTRIBUTING.md for guidelines on contributing to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google for the Gemma language model
- ISRO for space mission insights
- MediaPipe for computer vision capabilities
- The open-source AI community for tools and frameworks

## Support

For technical support or questions:
- Create an issue on GitHub
- Contact the development team
- Refer to the documentation in `docs/`

---

**Disclaimer**: This system is designed for psychological support and monitoring. It does not replace professional medical care and should be used as part of a comprehensive crew health monitoring system.