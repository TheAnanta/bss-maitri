# BSS Maitri - Bharatiya Space Station AI Assistant 🚀

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

## 🌟 Overview

BSS Maitri is a comprehensive multimodal AI assistant designed for the Bharatiya Space Station (BAS) to monitor and support crew emotional and physical well-being through advanced audio-video analysis. Built on Google's Gemma 3 model architecture, it provides real-time psychological support and intervention capabilities.

## 🎯 Problem Statement

**Background**: Crew members on-board space stations face isolation, sleep disruption, tight schedules and physical discomforts which can trigger psychological & physical issues. Early intervention can prevent errors and serious health issues.

**Objective**: Develop a multimodal AI assistant for detecting emotional and physical well-being of crew using audio-video inputs.

## ✨ Key Features

- **🧠 Multimodal Emotion Detection**: Real-time analysis of audio (voice tone) and visual (facial expressions) cues
- **🤖 AI-Powered Conversations**: Gemma 3-based empathetic responses with cultural sensitivity
- **⚡ Real-time Monitoring**: Continuous assessment of crew psychological state
- **🚨 Intelligent Alerts**: Automatic detection of critical emotional states with intervention recommendations
- **📱 Offline Operation**: Standalone system requiring no internet connectivity
- **🇮🇳 Cultural Adaptation**: Tailored for Indian cultural values and communication styles

## 🎨 Emotion Detection Capabilities

### Audio Analysis
- Voice tone and pitch analysis
- Speech pattern recognition  
- Stress indicators in vocal characteristics
- Fatigue detection through speech patterns

### Visual Analysis
- Facial expression recognition using MediaPipe
- Eye movement and blink pattern analysis
- Facial landmark detection for emotion mapping
- Real-time facial symmetry analysis

### Supported Emotional States
- 😌 Calm/Neutral
- 😰 Stress  
- 😟 Anxiety
- 😴 Fatigue
- 😢 Sadness
- 😠 Anger
- 😨 Fear
- 😊 Joy
- 😲 Surprise

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/TheAnanta/bss-maitri.git
cd bss-maitri

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

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
    print(f"Detected: {audio_result['emotion_analysis']['dominant_emotion']}")
```

### Command Line Interface

```bash
# Interactive text mode
python -m bss_maitri.main --mode interactive

# Audio analysis mode
python -m bss_maitri.main --mode audio --audio-file audio.wav

# Video analysis mode (webcam)
python -m bss_maitri.main --mode video --video-source 0
```

## 🏗️ Architecture

```
BSS Maitri AI Assistant
├── 🧠 Core Components
│   ├── MaitriAssistant (Main orchestrator)
│   ├── EmotionDetector (Multimodal fusion)
│   └── ConversationEngine (AI responses)
├── 🎵 Audio Processing
│   └── AudioEmotionDetector (Voice analysis)
├── 👁️ Vision Processing  
│   └── VisionEmotionDetector (Facial analysis)
├── 🤖 AI Models
│   └── GemmaModel (Language model integration)
└── ⚙️ Configuration & Utils
    ├── Config management
    └── Utility functions
```

## 🔧 Configuration

Customize the system through `bss_maitri/config/config.yaml`:

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

## 📊 Monitoring & Alerts

The system automatically monitors for:
- **🚨 Critical Emotional States**: High stress, severe anxiety
- **😴 Fatigue Levels**: Dangerous fatigue affecting mission safety  
- **⏱️ Sustained Negative Emotions**: Prolonged stress/anxiety periods
- **📞 Communication Gaps**: Extended periods without interaction

## 🛡️ Space Environment Considerations

- **Low-latency Processing**: Real-time emotion detection and response
- **Resource Optimization**: Efficient operation on limited computational resources
- **Isolation Support**: Addresses psychological challenges of space isolation
- **Mission Integration**: Balances support with mission requirements
- **Cultural Sensitivity**: Incorporates Indian values and communication styles

## 📁 Project Structure

```
bss-maitri/
├── bss_maitri/           # Main package
│   ├── audio/            # Audio emotion detection
│   ├── vision/           # Visual emotion detection  
│   ├── models/           # AI model integration
│   ├── core/             # Core components
│   ├── config/           # Configuration management
│   ├── utils/            # Utility functions
│   └── main.py           # Entry point
├── examples/             # Usage examples
├── tests/                # Test suite
├── docs/                 # Documentation
└── requirements.txt      # Dependencies
```

## 🧪 Testing

```bash
# Run basic tests
python -m pytest tests/

# Run specific test
python tests/test_basic.py

# Example usage
python examples/basic_usage.py
```

## 📚 Documentation

- **[Complete Documentation](docs/README.md)** - Detailed API reference and guides
- **[Configuration Guide](bss_maitri/config/config.yaml)** - System configuration options
- **[Examples](examples/)** - Code examples and demos

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and submit pull requests for any improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google** for the Gemma language model
- **ISRO** for space mission insights and requirements
- **MediaPipe** for computer vision capabilities
- **Open Source Community** for tools and frameworks

## 📞 Support

- 🐛 **Issues**: Report bugs via GitHub Issues
- 💬 **Discussions**: Join our community discussions
- 📧 **Contact**: Reach out to the development team

---

**⚠️ Disclaimer**: This system provides psychological support and monitoring capabilities. It should be used as part of a comprehensive crew health monitoring system and does not replace professional medical care.

**🚀 Made with ❤️ for the Bharatiya Space Station Mission**
