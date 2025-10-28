"""Configuration and constants for Kani TTS"""

# Tokenizer configuration
TOKENIZER_LENGTH = 64400

# Special tokens
START_OF_TEXT = 1
END_OF_TEXT = 2
START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2
START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4
START_OF_AI = TOKENIZER_LENGTH + 5
END_OF_AI = TOKENIZER_LENGTH + 6
PAD_TOKEN = TOKENIZER_LENGTH + 7
AUDIO_TOKENS_START = TOKENIZER_LENGTH + 10

# Audio configuration
CODEBOOK_SIZE = 4032
SAMPLE_RATE = 22050

# Streaming configuration
CHUNK_SIZE = 25  # Number of new frames to output per iteration
LOOKBACK_FRAMES = 15  # Number of frames to include from previous context

# Generation configuration
TEMPERATURE = 0.6
TOP_P = 0.95
REPETITION_PENALTY = 1.1
REPETITION_CONTEXT_SIZE = 20
MAX_TOKENS = 1200

# Model paths
MODEL_NAME = "nineninesix/kani-tts-370m"
CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
