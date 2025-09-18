
<img src="public/logo.png" alt="drawing" width="200"/>

<br/>

# Kani TTS

A modular Human-Like TTS Model that generates high-quality speech from text input.

## Features
- **450M Parameters**: optimized for edge devices and affordable servers.
- **High-Quality Speech**: 22kHz, 0.6kbps compression.

## Installation

### Prerequisites

```bash
# Core dependencies
pip install torch librosa soundfile numpy huggingface_hub
pip install "nemo_toolkit[tts]"

# CRITICAL: Custom transformers build required for "lfm2" model type
pip install -U "git+https://github.com/huggingface/transformers.git"

# Optional: For web interface
pip install fastapi uvicorn
```

## Quick Start

```bash
# Generate audio with default sample text
python basic/main.py

# Generate audio with custom text
python basic/main.py --prompt "Hello world! My name is Kani, I'm a speech generation model!"
```

This will:
1. Load the TTS model
2. Generate speech from the provided text (or built-in sample text if no prompt given)
3. Save audio as `generated_audio_YYYYMMDD_HHMMSS.wav`

## Web Interface

For a browser-based interface with real-time audio playback:

```bash
# Start the FastAPI server
python fastapi_example/server.py

# Open fastapi_example/client.html in your web browser
# Server runs on http://localhost:8000
```

The web interface provides:
- Interactive text input with example prompts
- Parameter adjustment (temperature, max tokens)
- Real-time audio generation and playback
- Download functionality for generated audio
- Server health monitoring

## Architecture

The system uses a layered architecture with clear separation of concerns:

- **Configuration Layer**: Centralized settings for models and audio processing
- **Token Management**: Handles special tokens for speech/text boundaries  
- **Audio Processing**: Strategy pattern for different codec implementations
- **Model Inference**: Text-to-token generation with the LLM
- **Audio Extraction**: Validates and processes audio codes from token sequences

## Configuration

Default configuration uses:
- **Text Model**: `https://huggingface.co/nineninesix/kani-tts-450m-0.1-pt`
- **Audio Codec**: `nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps`
- **Sample Rate**: 22,050 Hz
- **Generation**: 1200 max tokens, temperature 0.6

Modify `config.py` to customize these settings.

## Tested on

- NVIDIA GeForce RTX 5080
- Driver Version: 570.169
- CUDA Version: 12.8
- 16GB GPU memory
- Python: 3.12
- Transformers: 4.57.0.dev0

## Inference speed
In order to generate 15sec audio it takes ~1sec and ~2Gb GPU VRAM





