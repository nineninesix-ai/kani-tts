# NanoCodec TTS System

A modular Text-to-Speech system that combines NVIDIA NeMo audio codecs with Hugging Face transformers to generate high-quality speech from text input.

## Features

- **High-Quality Audio**: Uses NVIDIA NeMo nano codec (22kHz, 0.6kbps compression)
- **Modular Architecture**: Clean separation of concerns with pluggable components
- **Automatic File Saving**: Generates timestamped WAV files
- **Configurable Parameters**: Easy configuration for models, audio settings, and generation parameters

## Installation

### Prerequisites

This project requires a custom build of transformers due to the specialized "lfm2" model type.

```bash
# Core dependencies
pip install torch librosa soundfile numpy huggingface_hub
pip install "nemo_toolkit[tts]"

# CRITICAL: Custom transformers build required for "lfm2" model type
pip install -U "git+https://github.com/huggingface/transformers.git"

# Authentication for model access
hf auth login
```

## Quick Start

```bash
# Generate audio with default sample text
python basic/main.py

# Generate audio with custom text
python basic/main.py --prompt "Hello world! My name is Kani, I'm a speech generation model!"
```

This will:
1. Load the TTS model and audio codec
2. Generate speech from the provided text (or built-in sample text if no prompt given)
3. Save audio as `generated_audio_YYYYMMDD_HHMMSS.wav`

## Architecture

The system uses a layered architecture with clear separation of concerns:

- **Configuration Layer**: Centralized settings for models and audio processing
- **Token Management**: Handles special tokens for speech/text boundaries  
- **Audio Processing**: Strategy pattern for different codec implementations
- **Model Inference**: Text-to-token generation with the LLM
- **Audio Extraction**: Validates and processes audio codes from token sequences

## Configuration

Default configuration uses:
- **Text Model**: `url of your hf model`
- **Audio Codec**: `nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps`
- **Sample Rate**: 22,050 Hz
- **Generation**: 1200 max tokens, temperature 0.6

Modify `config.py` to customize these settings.

## Project Structure

```
├── main.py          # Entry point and execution
├── config.py        # Configuration classes
├── tokens.py        # Token registry and management
├── audio.py         # Audio processing and NeMo integration
├── models.py        # Model inference and text processing
├── extractors.py    # Audio/text extraction from token sequences
├── factory.py       # Factory pattern for component creation
└── __init__.py      # Package initialization
```

## Tested on

- NVIDIA GeForce RTX 5080
- NVIDIA-SMI 570.169
- Driver Version: 570.169
- CUDA Version: 12.8
- 16GB GPU memory
- Python: 3.12
- Transformers: 4.57.0.dev0

## Inference speed
In order to generate 10sec audio it takes 4sec and ~2Gb GPU VRAM





