"""Extractors for processing audio and text from token sequences."""

import torch
from typing import Tuple, Optional
from .tokens import TokenRegistry


class AudioCodeExtractor:
    """Handles extraction and validation of audio codes from token sequences."""
    
    def __init__(self, token_registry: TokenRegistry):
        self.tokens = token_registry
    
    def validate_output(self, out_ids: torch.Tensor) -> None:
        """Validate that required speech tokens are present."""
        start_present = self.tokens.start_of_speech in out_ids
        end_present = self.tokens.end_of_speech in out_ids
        
        if not (start_present and end_present):
            raise ValueError('Special speech tokens not found in output')
    
    def extract_audio_codes(self, out_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and process audio codes from token sequence."""
        try:
            start_idx = (out_ids == self.tokens.start_of_speech).nonzero(as_tuple=True)[0].item()
            end_idx = (out_ids == self.tokens.end_of_speech).nonzero(as_tuple=True)[0].item()
        except IndexError:
            raise ValueError('Speech tokens not found in sequence')
        
        if start_idx >= end_idx:
            raise ValueError('Invalid audio codes sequence - start token after end token')
        
        audio_codes = out_ids[start_idx + 1:end_idx]
        
        if len(audio_codes) % 4 != 0:
            raise ValueError('Audio codes length must be multiple of 4')
        
        audio_codes = audio_codes.reshape(-1, 4)
        audio_codes = audio_codes - torch.tensor([self.tokens.codebook_size * i for i in range(4)])
        audio_codes = audio_codes - self.tokens.audio_tokens_start
        
        if (audio_codes < 0).sum().item() > 0:
            raise ValueError('Invalid audio tokens detected')
        
        audio_codes = audio_codes.T.unsqueeze(0)
        length = torch.tensor([audio_codes.shape[-1]])
        
        return audio_codes, length


class TextExtractor:
    """Handles text extraction from token sequences."""
    
    def __init__(self, token_registry: TokenRegistry, tokenizer):
        self.tokens = token_registry
        self.tokenizer = tokenizer
    
    def extract_text(self, out_ids: torch.Tensor) -> Optional[str]:
        """Extract text from token sequence."""
        try:
            start_idx = (out_ids == self.tokens.start_of_text).nonzero(as_tuple=True)[0].item()
            end_idx = (out_ids == self.tokens.end_of_text).nonzero(as_tuple=True)[0].item()
            text_tokens = out_ids[start_idx:end_idx + 1]
            return self.tokenizer.decode(text_tokens, skip_special_tokens=True)
        except (IndexError, AttributeError):
            return None