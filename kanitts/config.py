"""Configuration classes for the TTS system."""

from dataclasses import dataclass
from typing import Optional
from .tokens import TokenRegistry


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    nemo_model_name: str = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
    sample_rate: int = 22050
    device: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for language model."""
    model_name: str = 'kyrgyz-ai-research/lfm-nano-codec-tts-exp-4-large-61468-st'
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 1200
    temperature: float = 0.6
    top_p: float = 0.95
    repetition_penalty: float = 1.1


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig
    audio: AudioConfig
    tokens: TokenRegistry
    
    @classmethod
    def default(cls) -> 'Config':
        return cls(
            model=ModelConfig(),
            audio=AudioConfig(),
            tokens=TokenRegistry()
        )