"""Main execution module for the TTS system."""

import argparse
import logging
import soundfile as sf
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanitts import Config, TTSFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Text-to-Speech generation')
    parser.add_argument('--p', '--prompt', type=str, 
                       default=('The morning fog rolled across the valley like a gentle gray blanket, slowly revealing the ancient oak trees that had stood sentinel for centuries.'),
                       help='Text prompt to convert to speech')
    args = parser.parse_args()
    
    try:
        config = Config.default()
        kani_model, _ = TTSFactory.create_system(config)
        
        prompt = args.p
        
        audio, text = kani_model.run_model(prompt)
        
        # Save audio as WAV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"basic/generated_audio_{timestamp}.wav"
        
        try:
            # Get sample rate from config (default 22050)
            sample_rate = config.audio.sample_rate
            sf.write(filename, audio, sample_rate)
            logger.info(f"Audio saved as: {filename}")
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            raise
        
        print(f'TEXT: {text}')
        logger.info("Audio generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()


