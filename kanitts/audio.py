"""Audio processing components for the TTS system."""

import torch
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from nemo.collections.tts.models import AudioCodecModel
from transformers import AutoTokenizer
from .config import Config, AudioConfig
from .extractors import AudioCodeExtractor, TextExtractor

from nemo.utils.nemo_logging import Logger

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()

logger = logging.getLogger(__name__)


class AudioProcessor(ABC):
    """Abstract base class for audio processing strategies."""
    
    @abstractmethod
    def decode_audio(self, audio_codes: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        pass


class NemoAudioProcessor(AudioProcessor):
    """NeMo-based audio processing implementation."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading NeMo codec model: {self.config.nemo_model_name}")
            self._model = AudioCodecModel.from_pretrained(self.config.nemo_model_name).eval()
            self._model.to(self.device)
        return self._model
    
    def decode_audio(self, audio_codes: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        audio_codes, length = audio_codes.to(self.device), length.to(self.device)
        with torch.inference_mode():
            reconstructed_audio, _ = self.model.decode(tokens=audio_codes, tokens_len=length)
            return reconstructed_audio.cpu().detach().numpy().squeeze()


class NemoAudioPlayer:
    """Orchestrates audio generation from token sequences."""
    
    def __init__(self, config: Config, text_tokenizer_name: Optional[str] = None):
        self.config = config
        self.tokens = config.tokens
        self.audio_processor = NemoAudioProcessor(config.audio)
        self.code_extractor = AudioCodeExtractor(config.tokens)
        
        self.text_extractor = None
        if text_tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
            self.text_extractor = TextExtractor(config.tokens, tokenizer)
    
    def get_waveform(self, out_ids: torch.Tensor) -> Tuple[torch.Tensor, Optional[str]]:
        """Generate waveform from model output tokens."""
        try:
            out_ids = out_ids.flatten()
            self.code_extractor.validate_output(out_ids)
            audio_codes, length = self.code_extractor.extract_audio_codes(out_ids)
            
            output_audio = self.audio_processor.decode_audio(audio_codes, length)
            
            text = None
            if self.text_extractor:
                text = self.text_extractor.extract_text(out_ids)
            
            return output_audio, text
            
        except Exception as e:
            logger.error(f"Error generating waveform: {e}")
            raise