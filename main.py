"""Kani TTS - Text to Speech Generation"""

import time
from audio import LLMAudioPlayer, StreamingAudioWriter
from generation import TTSGenerator
from config import CHUNK_SIZE, LOOKBACK_FRAMES

from nemo.utils.nemo_logging import Logger

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()


def time_report(point_1, point_2, point_3):
    model_request = point_2 - point_1
    player_time = point_3 - point_2
    total_time = point_3 - point_1
    report = f"SPEECH TOKENS: {model_request:.2f}\nCODEC: {player_time:.2f}\nTOTAL: {total_time:.2f}"
    return report


def main():
    # Initialize generator and audio player
    generator = TTSGenerator()
    player = LLMAudioPlayer(generator.tokenizer)

    # Set prompt
    prompt = "katie: Oh, yeah. I mean did you want to get a quick snack together or maybe something before you go?"

    # Create streaming audio writer with sliding window decoder
    # Uses lookback context from previous frames to maintain codec continuity
    audio_writer = StreamingAudioWriter(
        player,
        'output.wav',
        chunk_size=CHUNK_SIZE,        # Output 25 new frames (2.0s) per iteration
        lookback_frames=LOOKBACK_FRAMES    # Include 15 previous frames (1.2s) for context
    )
    audio_writer.start()

    # Generate speech
    result = generator.generate(prompt, audio_writer)

    # Finalize and write audio file
    audio = audio_writer.finalize()

    point_3 = time.time()

    # Print results
    print(time_report(result['point_1'], result['point_2'], point_3))
    # print(f"\n[DEBUG] First 100 chars of generated text: {result['generated_text'][:100]}")
    # print(f"[DEBUG] Last 100 chars of generated text: {result['generated_text'][-100:]}")


if __name__ == "__main__":
    main()

