"""Audio player for LLM-generated speech tokens"""

import torch
import numpy as np
from nemo.collections.tts.models import AudioCodecModel

from config import (
    TOKENIZER_LENGTH, START_OF_TEXT, END_OF_TEXT,
    START_OF_SPEECH, END_OF_SPEECH, START_OF_HUMAN, END_OF_HUMAN,
    START_OF_AI, END_OF_AI, PAD_TOKEN, AUDIO_TOKENS_START, CODEBOOK_SIZE
)


class LLMAudioPlayer:
    def __init__(self, tokenizer) -> None:
        self.nemo_codec_model = AudioCodecModel\
                .from_pretrained("nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps").eval()

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.nemo_codec_model.to(self.device)
        self.tokenizer = tokenizer

        self.tokeniser_length = TOKENIZER_LENGTH
        self.start_of_text = START_OF_TEXT
        self.end_of_text = END_OF_TEXT
        self.start_of_speech = START_OF_SPEECH
        self.end_of_speech = END_OF_SPEECH
        self.start_of_human = START_OF_HUMAN
        self.end_of_human = END_OF_HUMAN
        self.start_of_ai = START_OF_AI
        self.end_of_ai = END_OF_AI
        self.pad_token = PAD_TOKEN
        self.audio_tokens_start = AUDIO_TOKENS_START
        self.codebook_size = CODEBOOK_SIZE

    def output_validation(self, out_ids):
        start_of_speech_flag = self.start_of_speech in out_ids
        end_of_speech_flag = self.end_of_speech in out_ids
        if not (start_of_speech_flag and end_of_speech_flag):
            raise ValueError('Special speech tokens not exist!')

    def get_nano_codes(self, out_ids):
        start_a_idx = (out_ids == self.start_of_speech).nonzero(as_tuple=True)[0].item()
        end_a_idx   = (out_ids == self.end_of_speech).nonzero(as_tuple=True)[0].item()
        if start_a_idx >= end_a_idx:
            raise ValueError('Invalid audio codes sequence!')

        audio_codes = out_ids[start_a_idx+1 : end_a_idx]
        if len(audio_codes) % 4:
            raise ValueError('The length of the sequence must be a multiple of 4!')
        audio_codes = audio_codes.reshape(-1, 4)
        audio_codes = audio_codes - torch.tensor([self.codebook_size * i for i in range(4)])
        audio_codes = audio_codes - self.audio_tokens_start
        if (audio_codes < 0).sum().item() > 0:
            raise ValueError('Invalid audio tokens!')

        audio_codes = audio_codes.T.unsqueeze(0)
        len_ = torch.tensor([audio_codes.shape[-1]])
        return audio_codes, len_

    def get_text(self, out_ids):
        try:
            start_t_idx = (out_ids == self.start_of_text).tolist().index(True)
            end_t_idx   = (out_ids == self.end_of_text).tolist().index(True)
            txt_tokens = out_ids[start_t_idx : end_t_idx+1]
            text = self.tokenizer.decode(txt_tokens, skip_special_tokens=True)
            return text
        except ValueError:
            return None

    def get_waveform(self, out_ids):
        out_ids = out_ids.flatten()
        self.output_validation(out_ids)
        audio_codes, len_ = self.get_nano_codes(out_ids)
        audio_codes, len_ = audio_codes.to(self.device), len_.to(self.device)
        with torch.inference_mode():
            reconstructed_audio, _ = self.nemo_codec_model.decode(tokens=audio_codes, tokens_len=len_)
            output_audio = reconstructed_audio.cpu().detach().numpy().squeeze()

        text = self.get_text(out_ids)
        return output_audio, text

    def decode_audio_chunk(self, audio_codes):
        """Decode a chunk of audio codes (shape: [num_frames, 4])"""
        if len(audio_codes) == 0:
            return None

        # Process audio codes: subtract offsets for each codebook
        audio_codes = torch.tensor(audio_codes, device=self.device)
        audio_codes = audio_codes - torch.tensor([self.codebook_size * i for i in range(4)], device=self.device)
        audio_codes = audio_codes - self.audio_tokens_start

        if (audio_codes < 0).sum().item() > 0:
            return None  # Invalid tokens, skip

        # Shape: (1, 4, num_frames) - batch_size=1, num_codebooks=4, num_frames
        audio_codes = audio_codes.T.unsqueeze(0)
        len_ = torch.tensor([audio_codes.shape[-1]], device=self.device)

        with torch.inference_mode():
            reconstructed_audio, _ = self.nemo_codec_model.decode(tokens=audio_codes, tokens_len=len_)
            output_audio = reconstructed_audio.cpu().detach().numpy().squeeze()

        return output_audio
