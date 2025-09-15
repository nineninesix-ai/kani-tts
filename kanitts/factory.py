"""Factory for creating TTS system components."""

from typing import Optional, Tuple
from .config import Config
from .audio import NemoAudioPlayer
from .models import KaniModel


class TTSFactory:
    """Factory for creating TTS system components."""
    
    @staticmethod
    def create_system(config: Optional[Config] = None) -> Tuple[KaniModel, NemoAudioPlayer]:
        """Create a complete TTS system."""
        if config is None:
            config = Config.default()
        
        player = NemoAudioPlayer(config)
        model = KaniModel(config, player)
        
        return model, player