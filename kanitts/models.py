"""Model inference components for the TTS system."""

import torch
import logging
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import Config, ModelConfig
from .tokens import TokenRegistry
from .audio import NemoAudioPlayer

logger = logging.getLogger(__name__)


class InputProcessor:
    """Handles input text processing and tokenization."""
    
    def __init__(self, tokenizer, token_registry: TokenRegistry):
        self.tokenizer = tokenizer
        self.tokens = token_registry
    
    def prepare_input(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input text for model inference."""
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        
        start_token = torch.tensor([[self.tokens.start_of_human]], dtype=torch.int64)
        end_tokens = torch.tensor([[self.tokens.end_of_text, self.tokens.end_of_human]], dtype=torch.int64)
        
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
        attention_mask = torch.ones(1, modified_input_ids.shape[1], dtype=torch.int64)
        
        return modified_input_ids, attention_mask


class ModelInference:
    """Handles model inference operations."""
    
    def __init__(self, model, config: ModelConfig, token_registry: TokenRegistry, device: Optional[str] = None):
        self.model = model
        self.config = config
        self.tokens = token_registry
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Generate tokens from input."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                num_return_sequences=1,
                eos_token_id=self.tokens.end_of_speech,
            )
        return generated_ids.to('cpu')


class KaniModel:
    """Main text-to-speech model orchestrator."""
    
    def __init__(self, config: Config, player: NemoAudioPlayer):
        self.config = config
        self.player = player
        
        logger.info(f"Loading model: {config.model.model_name}")
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        torch_dtype_name = config.model.torch_dtype
        torch_dtype = getattr(torch, torch_dtype_name)
        device_map = config.model.device_map

        if device == 'mps' and torch_dtype_name not in {"float16", "float32"}:
            logger.warning(
                "MPS backend does not support %s precision; falling back to float32.",
                torch_dtype_name,
            )
            torch_dtype = torch.float32
            device_map = None

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        if device != 'cpu' and device_map is None:
            self.model.to(device)
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        self.input_processor = InputProcessor(self.tokenizer, config.tokens)
        self.inference = ModelInference(self.model, config.model, config.tokens, device=self.device)
    
    def run_model(self, text: str) -> Tuple[torch.Tensor, str]:
        """Generate audio from input text."""
        try:
            logger.info(f"Processing text: {text[:50]}...")
            input_ids, attention_mask = self.input_processor.prepare_input(text)
            model_output = self.inference.generate(input_ids, attention_mask)
            audio, _ = self.player.get_waveform(model_output)
            return audio, text
        except Exception as e:
            logger.error(f"Error in model execution: {e}")
            raise
