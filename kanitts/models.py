"""Model inference components for the TTS system."""

import torch
import logging
from typing import Tuple
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
    
    def __init__(self, model, config: ModelConfig, token_registry: TokenRegistry):
        self.model = model
        self.config = config
        self.tokens = token_registry
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
        torch_dtype = getattr(torch, config.model.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            torch_dtype=torch_dtype,
            device_map=config.model.device_map,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        self.input_processor = InputProcessor(self.tokenizer, config.tokens)
        self.inference = ModelInference(self.model, config.model, config.tokens)
    
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