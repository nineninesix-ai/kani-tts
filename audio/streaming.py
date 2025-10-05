"""Streaming audio writer with sliding window decoder"""

import threading
import queue
import numpy as np
from scipy.io.wavfile import write

from config import SAMPLE_RATE, CHUNK_SIZE, LOOKBACK_FRAMES


class StreamingAudioWriter:
    def __init__(self, player, output_file, sample_rate=SAMPLE_RATE,
                 chunk_size=CHUNK_SIZE, lookback_frames=LOOKBACK_FRAMES):
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
        self.running = True
        self.inside_speech = False
        self.audio_token_buffer = []
        self.all_tokens = []  # Store all audio tokens for sliding window decoding
        self.frames_decoded = 0  # Track how many frames we've already output

    def decoder_worker(self):
        """Background thread that decodes audio chunks as they arrive"""
        speech_ended = False

        while self.running or not self.token_queue.empty():
            try:
                token_id = self.token_queue.get(timeout=0.1)

                # Check for start/end of speech markers
                if token_id == self.player.start_of_speech:
                    print(f"[DECODER] START_OF_SPEECH detected")
                    self.inside_speech = True
                    speech_ended = False
                    self.audio_token_buffer = []
                    continue

                if token_id == self.player.end_of_speech:
                    if speech_ended:
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
                    speech_ended = True
                    self.audio_token_buffer = []
                    continue

                # Accumulate audio tokens (only if speech hasn't ended)
                if self.inside_speech and not speech_ended:
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
        self.running = False
        self.decoder_thread.join()

        if self.audio_chunks:
            # Concatenate all audio chunks
            full_audio = np.concatenate(self.audio_chunks)

            # Only write to file if output_file is specified
            if self.output_file:
                write(self.output_file, self.sample_rate, full_audio)
                print(f"[WRITER] Wrote {len(full_audio)/self.sample_rate:.2f}s of audio to {self.output_file}")

            return full_audio
        return None

    def start(self):
        """Start the decoder thread"""
        self.decoder_thread = threading.Thread(target=self.decoder_worker)
        self.decoder_thread.start()
