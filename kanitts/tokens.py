"""Token registry for managing special tokens in the TTS system."""


class TokenRegistry:
    """Centralized token management for audio codec operations."""
    
    def __init__(self, tokenizer_length: int = 64400):
        self.tokenizer_length = tokenizer_length
        self.start_of_text = 1
        self.end_of_text = 2
        self.start_of_speech = tokenizer_length + 1
        self.end_of_speech = tokenizer_length + 2
        self.start_of_human = tokenizer_length + 3
        self.end_of_human = tokenizer_length + 4
        self.start_of_ai = tokenizer_length + 5
        self.end_of_ai = tokenizer_length + 6
        self.pad_token = tokenizer_length + 7
        self.audio_tokens_start = tokenizer_length + 10
        self.codebook_size = 4032