from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
import librosa
import torch
import soundfile as sf
from nemo.collections.tts.models import AudioCodecModel
import locale
import os
import numpy as np
from threading import Thread

from scipy.io.wavfile import write
import threading
import queue

import time

def time_report(point_1, point_2, point_3):
    model_request = point_2 - point_1
    player_time = point_3 - point_2
    total_time = point_3 - point_1
    report = f"SPEECH TOKENS: {model_request:.2f}\nCODEC: {player_time:.2f}\nTOTAL: {total_time:.2f}"
    return report


model_name = 'nineninesix/kani-tts-370m'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)


class LLMAudioPlayer:
    def __init__(self, tokenizer) -> None:
        self.nemo_codec_model = AudioCodecModel\
                .from_pretrained("nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps").eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nemo_codec_model.to(self.device)
        self.tokenizer = tokenizer


        self.tokeniser_length = 64400
        self.start_of_text = 1
        self.end_of_text = 2
        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2
        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4
        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7
        self.audio_tokens_start = self.tokeniser_length + 10
        self.codebook_size = 4032

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
    

    def get_waveform(self, out_ids):
        out_ids = out_ids.flatten()
        self.output_validation(out_ids)
        audio_codes, len_ = self.get_nano_codes(out_ids)
        audio_codes, len_ = audio_codes.to(self.device), len_.to(self.device)
        with torch.inference_mode():
            reconstructed_audio, _ = self.nemo_codec_model.decode(tokens=audio_codes, tokens_len=len_)
            output_audio = reconstructed_audio.cpu().detach().numpy().squeeze()

        return output_audio, None
        
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

        
player = LLMAudioPlayer(tokenizer)

TOKENIZER_LENGTH = 64400
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

prompt = "andrew: Holy fu- Oh my God! Don't you understand how dangerous it is, huh?"

if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
modified_input_ids = modified_input_ids.to(device)

attention_mask = torch.ones(1, modified_input_ids.shape[1], dtype=torch.int64)
attention_mask = attention_mask.to(device)

class TokenIDStreamer(BaseStreamer):
    """Custom streamer that yields token IDs instead of decoded text"""
    def __init__(self, callback):
        self.callback = callback

    def put(self, value):
        """Called by model.generate() with token IDs"""
        if len(value.shape) > 1:
            # Batch dimension, take first element
            token_ids = value[0].tolist()
        else:
            token_ids = value.tolist()

        for token_id in token_ids:
            self.callback(token_id)

    def end(self):
        """Called when generation is complete"""
        pass

class StreamingAudioWriter:
    def __init__(self, player, output_file, sample_rate=22050, chunk_size=25, lookback_frames=15):
        """
        Sliding window decoder with lookback context.

        Args:
            player: LLMAudioPlayer instance
            output_file: Output WAV file path
            sample_rate: Audio sample rate (22050 Hz for nanocodec)
            chunk_size: Number of NEW frames to output per iteration
            lookback_frames: Number of frames to include from previous context for continuity
        """
        self.player = player
        self.output_file = output_file
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.lookback_frames = lookback_frames
        self.token_queue = queue.Queue()
        self.audio_chunks = []
        self.inside_speech = False
        self.speech_ended = False
        self.audio_token_buffer = []
        self.all_tokens = []  # Store all audio tokens for sliding window decoding
        self.frames_decoded = 0  # Track how many frames we've already output

    def decoder_worker(self):
        """Background thread that decodes audio chunks as they arrive"""
        while True:
            try:
                token_id = self.token_queue.get(timeout=0.1)

                # Check for sentinel value (end of stream)
                if token_id is None:
                    print(f"[DECODER] Received sentinel, shutting down")
                    break

                # Check for start/end of speech markers
                if token_id == self.player.start_of_speech:
                    print(f"[DECODER] START_OF_SPEECH detected")
                    self.inside_speech = True
                    self.speech_ended = False
                    self.audio_token_buffer = []
                    self.all_tokens = []
                    self.frames_decoded = 0
                    continue

                if token_id == self.player.end_of_speech:
                    if self.speech_ended:
                        print(f"[DECODER] Warning: Duplicate END_OF_SPEECH detected, ignoring")
                        continue

                    print(f"[DECODER] END_OF_SPEECH detected")

                    # Decode any remaining frames with sliding window
                    total_frames = len(self.all_tokens) // 4
                    remaining_frames = total_frames - self.frames_decoded

                    if remaining_frames >= 1:
                        # Decode from lookback point to end
                        start_frame = max(0, self.frames_decoded - self.lookback_frames)
                        start_token = start_frame * 4

                        tokens_to_decode = self.all_tokens[start_token:]
                        num_frames = len(tokens_to_decode) // 4

                        if num_frames > 0:
                            codes = np.array(tokens_to_decode[:num_frames * 4]).reshape(-1, 4)
                            audio_chunk = self.player.decode_audio_chunk(codes)

                            if audio_chunk is not None:
                                samples_per_frame = len(audio_chunk) // num_frames

                                # Skip lookback portion, only save new frames
                                lookback_skip = min(self.frames_decoded, self.lookback_frames)
                                skip_samples = lookback_skip * samples_per_frame
                                new_audio = audio_chunk[skip_samples:]

                                self.audio_chunks.append(new_audio)
                                print(f"[DECODER] Final chunk: {remaining_frames} frames ({remaining_frames/12.5:.2f}s audio)")

                    self.inside_speech = False
                    self.speech_ended = True
                    self.audio_token_buffer = []
                    continue

                # Accumulate audio tokens (only if speech hasn't ended)
                if self.inside_speech and not self.speech_ended:
                    self.audio_token_buffer.append(token_id)
                    self.all_tokens.append(token_id)  # Keep all tokens for sliding window

                    # Decode when we have enough NEW frames to process
                    total_frames = len(self.all_tokens) // 4
                    new_frames = total_frames - self.frames_decoded

                    if new_frames >= self.chunk_size:
                        # Calculate sliding window: include lookback_frames from previous context
                        start_frame = max(0, self.frames_decoded - self.lookback_frames)
                        start_token = start_frame * 4

                        # Decode from start_frame to current end
                        tokens_to_decode = self.all_tokens[start_token:]
                        num_frames = len(tokens_to_decode) // 4

                        codes = np.array(tokens_to_decode[:num_frames * 4]).reshape(-1, 4)
                        audio_chunk = self.player.decode_audio_chunk(codes)

                        if audio_chunk is not None:
                            samples_per_frame = len(audio_chunk) // num_frames

                            # Skip the lookback portion - only save the NEW frames
                            lookback_skip = min(self.frames_decoded, self.lookback_frames)
                            skip_samples = lookback_skip * samples_per_frame

                            # Extract only the new chunk_size frames worth of audio
                            new_samples = self.chunk_size * samples_per_frame
                            new_audio = audio_chunk[skip_samples:skip_samples + new_samples]

                            self.audio_chunks.append(new_audio)
                            self.frames_decoded += self.chunk_size

                            print(f"[DECODER] Decoded {self.chunk_size} frames ({self.chunk_size/12.5:.2f}s audio) with {self.lookback_frames}-frame lookback context")

                        # Clear buffer (we've stored everything in all_tokens)
                        self.audio_token_buffer = []

            except queue.Empty:
                continue

    def add_token(self, token_id):
        """Add a token to the processing queue"""
        self.token_queue.put(token_id)

    def finalize(self):
        """Stop the decoder thread and write final audio file"""
        # Send sentinel to signal end of stream
        self.token_queue.put(None)

        # Wait for decoder thread with timeout
        self.decoder_thread.join(timeout=10.0)
        if self.decoder_thread.is_alive():
            print("[WRITER] Warning: Decoder thread did not finish in time!")
            return None

        if self.audio_chunks:
            # Concatenate all audio chunks
            full_audio = np.concatenate(self.audio_chunks)
            write(self.output_file, self.sample_rate, full_audio)
            print(f"[WRITER] Wrote {len(full_audio)/self.sample_rate:.2f}s of audio to {self.output_file}")
            return full_audio
        return None

    def start(self):
        """Start the decoder thread"""
        self.decoder_thread = threading.Thread(target=self.decoder_worker)
        self.decoder_thread.start()


point_1 = time.time()


audio_writer = StreamingAudioWriter(
    player,
    'output.wav',
    chunk_size=25,        # Output 25 new frames (2.0s) per iteration
    lookback_frames=15    # Include 15 previous frames (1.2s) for context
)
audio_writer.start()

all_token_ids = []

def on_token_generated(token_id):
    """Callback for each generated token"""
    all_token_ids.append(token_id)
    print(f"[LLM] Token {len(all_token_ids)}: {token_id}")
    audio_writer.add_token(token_id)

streamer = TokenIDStreamer(callback=on_token_generated)

generation_kwargs = dict(
    input_ids=modified_input_ids,
    attention_mask=attention_mask,
    max_new_tokens=1200,
    do_sample=True,
    temperature=.6,
    top_p=.95,
    repetition_penalty=1.1,
    num_return_sequences=1,
    eos_token_id=END_OF_SPEECH,
    streamer=streamer,
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()
thread.join()


point_2 = time.time()


print(f"\n[MAIN] Generation complete. Total tokens: {len(all_token_ids)}")

audio = audio_writer.finalize()


point_3 = time.time()

print(time_report(point_1, point_2, point_3))



