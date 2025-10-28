
# Basic example for KaniTTS

## Installation
### Prerequisites

- Python 3.10
- Git

### Setup

```bash
chmod +x setup.sh
./setup.sh
```

## Usage

### Option 1: Standalone Generation (Local WAV File)

Generate audio and save it to a WAV file:

```bash
python main.py
```

This will:
- Generate speech from the prompt in [main.py:28](main.py#L28)
- Save output to `output.wav`
- Display timing metrics for performance analysis

### Option 2: FastAPI Server + Web Interface

1. Start the server:
```bash
python server.py
```

The server will start on `http://localhost:8000`

2. Open the web interface:
```bash
open client.html
```

Or navigate to `http://localhost:8000` in your browser

## API Endpoints

### `POST /tts`
Generate complete audio file (non-streaming)

**Request:**
```json
{
  "text": "Hello world!",
  "temperature": 0.6,
  "max_tokens": 1200,
  "top_p": 0.95,
  "chunk_size": 25,
  "lookback_frames": 15
}
```

**Response:** WAV audio file

### `POST /stream-tts`
Stream audio chunks for immediate playback

**Request:** Same as `/tts`

**Response:** Streaming PCM audio chunks with metadata headers

## Configuration

Edit [config.py](config.py) to customize:

```python
# Audio settings
CHUNK_SIZE = 25 # Frames per streaming iteration (2.0s)
LOOKBACK_FRAMES = 15 # Context frames for continuity (1.2s)

# Generation parameters
TEMPERATURE = 0.6
TOP_P = 0.95
REPETITION_PENALTY = 1.1
MAX_TOKENS = 1200

# Model configuration
MODEL_NAME = "nineninesix/kani-tts-370m"
CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
```

## Technical Details

### Streaming Architecture

The system uses a **sliding window decoder** for smooth audio generation:

1. **Chunk Size (25 frames)** - Outputs ~2.0 seconds of new audio per iteration
2. **Lookback Frames (15 frames)** - Includes ~1.2 seconds of context from previous output

### Tested on

- NVIDIA GeForce RTX 5080
- Driver Version: 570.169
- CUDA Version: 12.8
- 16GB GPU memory
- Python: 3.12
- Transformers: 4.57.1

In order to generate 15sec audio it takes ~1sec and ~2Gb GPU VRAM

> **Note:** If you experience audio breaks during streaming, try increasing `CHUNK_SIZE` in [config.py](config.py) to buffer more frames per chunk.

## Models

- **TTS Model:** [nineninesix/kani-tts-370m](https://huggingface.co/nineninesix/kani-tts-370m)
- **Codec Model:** [nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps)

Models are automatically downloaded from Hugging Face on first run.

## Browser Compatibility

The web interface requires a modern browser with support for:
- Web Audio API
- Fetch API with streaming

## License
Apache 2

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.






