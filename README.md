
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

## Configuration

Default configuration uses:
- **Model**: `https://huggingface.co/nineninesix/kani-tts-450m-0.1-pt`
- **Sample Rate**: 22,050 Hz
- **Generation**: 1200 max tokens, temperature 1.4

### Model Variants

Choose different models for specific voice characteristics:

- **Base model (default)**: `nineninesix/kani-tts-450m-0.1-pt`
- **Female voice**: `nineninesix/kani-tts-450m-0.2-ft`
- **Male voice**: `nineninesix/kani-tts-450m-0.1-ft`

> Base model generates random voices

To use a different model, modify the class `ModelConfig` in `config.py`.

## Examples

| Text | Audio |
|---|---|
| I do believe Marsellus Wallace, MY husband, YOUR boss, told you to take me out and do WHATEVER I WANTED. | [Play](https://github.com/nineninesix-ai/kani-tts/raw/82dbdaa2f76a6ad135f7b08f974e72046c51d3ba/public/mia.wav) |
| What do we say the the god of death? Not today! | [Play](https://github.com/nineninesix-ai/kani-tts/raw/82dbdaa2f76a6ad135f7b08f974e72046c51d3ba/public/arya.wav) |
| What do you call a lawyer with an IQ of 60? Your honor | [Play](https://github.com/nineninesix-ai/kani-tts/raw/82dbdaa2f76a6ad135f7b08f974e72046c51d3ba/public/saul.wav) |
| You mean, let me understand this cause, you know maybe it's me, it's a little fucked up maybe, but I'm funny how, I mean funny like I'm a clown, I amuse you? I make you laugh, I'm here to fucking amuse you? | [Play](https://github.com/nineninesix-ai/kani-tts/raw/82dbdaa2f76a6ad135f7b08f974e72046c51d3ba/public/tommy.wav) |


## Architecture

The system uses a layered architecture with clear separation of concerns:

- **Configuration Layer**: Centralized settings for models and audio processing
- **Token Management**: Handles special tokens for speech/text boundaries  
- **Audio Processing**: Strategy pattern for different codec implementations
- **Model Inference**: Text-to-token generation with the LLM
- **Audio Extraction**: Validates and processes audio codes from token sequences

## Tested on

- NVIDIA GeForce RTX 5080
- Driver Version: 570.169
- CUDA Version: 12.8
- 16GB GPU memory
- Python: 3.12
- Transformers: 4.57.0.dev0

## Inference speed
In order to generate 15sec audio it takes ~1sec and ~2Gb GPU VRAM





