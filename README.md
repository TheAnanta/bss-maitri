# 🚀 BSS Maitri - AI Assistant for Crew Well-being

## Background

Crew members on-board space stations face isolation, sleep disruption, tight schedules and physical discomforts which can trigger psychological & physical issues. Early intervention can prevent errors and serious health issues.

**BSS Maitri** is a multimodal AI assistant that detects emotional and physical well-being of crew using audio-video inputs, providing psychological companionship and support during space missions.

## Key Features

🎤 **Voice Emotion Analysis** - Analyzes speech patterns, tone, and vocal characteristics to detect emotional states

📹 **Facial Expression Recognition** - Uses computer vision to analyze facial expressions and micro-expressions

🤖 **On-Device AI** - Runs completely offline using Ollama with Gemma2:4b model (no cloud dependencies)

💬 **Psychological Companionship** - Provides supportive conversations and evidence-based interventions

📊 **Real-time Analysis** - Continuous monitoring of emotional and stress indicators

🔒 **Privacy-First** - All data processing happens locally on the device

## Installation

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed on your system

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TheAnanta/bss-maitri.git
   cd bss-maitri
   ```

2. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

### Manual Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install the package:**
   ```bash
   pip install -e .
   ```

3. **Setup Ollama model:**
   ```bash
   bss-maitri --mode setup
   ```

## Usage

### Web Interface (Recommended)

Start the web interface:
```bash
bss-maitri --mode web
```

Open your browser and navigate to `http://localhost:7860`

Features:
- Real-time audio and video analysis
- Interactive conversation with Maitri
- Visual stress and emotion indicators
- Privacy-preserving processing

### Command Line Interface

For text-based interaction:
```bash
bss-maitri --mode cli
```

### Advanced Options

```bash
# Custom port for web interface
bss-maitri --mode web --port 8080

# Create shareable public link
bss-maitri --mode web --share

# Enable debug logging
bss-maitri --mode web --debug

# Show all options
bss-maitri --help
```

## How It Works

1. **Audio Analysis**: Extracts features like pitch, tempo, spectral characteristics, and MFCC coefficients to detect emotional states and stress indicators

2. **Facial Analysis**: Uses MediaPipe to detect facial landmarks and analyze expressions, eye movements, and micro-expressions

3. **Multimodal Fusion**: Combines audio and visual cues for more accurate emotion detection

4. **AI Response**: Uses Ollama with Gemma2:4b model to generate contextual, supportive responses based on detected emotional state

5. **Temporal Analysis**: Tracks emotional patterns over time to identify trends and potential concerns

## Architecture

```
BSS Maitri
├── Audio Processing (librosa, scipy)
│   ├── Feature extraction (MFCC, pitch, tempo)
│   ├── Emotion classification
│   └── Stress indicator analysis
│
├── Video Processing (MediaPipe, OpenCV)
│   ├── Facial landmark detection
│   ├── Expression analysis
│   └── Micro-expression recognition
│
├── Multimodal Analysis
│   ├── Cross-modal fusion
│   ├── Temporal pattern analysis
│   └── Confidence scoring
│
├── AI Response (Ollama + Gemma2:4b)
│   ├── Emotion-aware responses
│   ├── Psychological support
│   └── Conversation management
│
└── User Interface (Gradio)
    ├── Web interface
    ├── Real-time processing
    └── Privacy controls
```

## Technical Details

### Supported Emotions
- Neutral, Happy, Sad, Angry, Fear, Surprised, Stressed

### Audio Features
- Mel-frequency Cepstral Coefficients (MFCC)
- Spectral centroid, rolloff, and bandwidth
- Pitch and fundamental frequency
- Zero-crossing rate and RMS energy
- Tempo and rhythm analysis

### Facial Features
- Eye aspect ratio (fatigue detection)
- Mouth curvature (smile/frown detection)
- Eyebrow position (surprise/concern)
- Facial geometry and proportions

### Privacy & Security
- **No data transmission**: All processing happens locally
- **No data storage**: No personal data is permanently stored
- **Open source**: Full transparency of algorithms
- **Offline operation**: Works without internet connection

## Use Cases

### 🚀 Space Missions
- Continuous crew emotional monitoring
- Early intervention for psychological issues
- Companionship during long-duration missions
- Stress management and support

### 🏥 Healthcare
- Patient emotional state monitoring
- Mental health assessment
- Therapy support and companionship

### 🏢 Workplace Wellness
- Employee stress monitoring
- Mental health support
- Work-life balance assessment

## Configuration

Edit `config.env` to customize:

```env
# Model Configuration
OLLAMA_MODEL=gemma2:4b

# Analysis Weights
MULTIMODAL_AUDIO_WEIGHT=0.5
MULTIMODAL_VIDEO_WEIGHT=0.5

# Interface Settings
WEB_PORT=7860
LOG_LEVEL=INFO
```

## Development

### Running Tests
```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Formatting
```bash
black bss_maitri/
flake8 bss_maitri/
```

### Type Checking
```bash
mypy bss_maitri/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Ollama Team** for the local LLM infrastructure
- **MediaPipe** for facial analysis capabilities
- **Librosa** for audio processing tools
- **Gradio** for the web interface framework

## Citation

```bibtex
@software{bss_maitri_2024,
  title={BSS Maitri: Multimodal AI Assistant for Crew Well-being},
  author={BSS Maitri Team},
  year={2024},
  url={https://github.com/TheAnanta/bss-maitri}
}
```

---

**Note**: This system is designed to support and complement human judgment, not replace professional psychological care. For serious mental health concerns, please consult qualified healthcare professionals.
