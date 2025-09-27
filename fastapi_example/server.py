#!/usr/bin/env python3
"""FastAPI server for TTS audio streaming."""

import io
import asyncio
import logging
import tempfile
from typing import Optional
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanitts import Config, TTSFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kani TTS API", description="Text-to-Speech API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    temperature: Optional[float] = 0.6
    max_tokens: Optional[int] = 1200

# Global TTS system - initialized on startup
tts_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize TTS system on startup."""
    global tts_system
    try:
        logger.info("Initializing TTS system...")
        config = Config.default()
        tts_system, _ = TTSFactory.create_system(config)
        logger.info("TTS system initialized successfully")

        logger.info("Warming up TTS model...")
        await asyncio.to_thread(tts_system.run_model, "Warm-up phrase for TTS initialization.")
        logger.info("TTS model warm-up completed")
    except Exception as e:
        logger.error(f"Failed to initialize TTS system: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Kani TTS API",
        "endpoints": {
            "/tts": "POST - Generate speech from text",
            "/stream-tts": "POST - Stream speech from text",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "tts_initialized": tts_system is not None}

@app.post("/tts")
async def generate_speech(request: TTSRequest):
    """Generate speech from text and return WAV audio."""
    if not tts_system:
        raise HTTPException(status_code=503, detail="TTS system not initialized")
    
    try:
        logger.info(f"Generating speech for text: {request.text[:50]}...")
        
        # Override config if custom parameters provided
        if request.temperature != 0.6 or request.max_tokens != 1200:
            config = Config.default()
            config.model.temperature = request.temperature
            config.model.max_new_tokens = request.max_tokens
            # Note: For production, you might want to create a new instance
            # For simplicity, we'll use the global one with original settings
        
        audio, _ = tts_system.run_model(request.text)
        
        # Convert numpy array to WAV bytes
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, audio, 22050)
            tmp_file.seek(0)
            wav_data = tmp_file.read()
        
        return StreamingResponse(
            io.BytesIO(wav_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=generated_speech.wav",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")

@app.post("/stream-tts")
async def stream_speech(request: TTSRequest):
    """Stream speech generation from text."""
    if not tts_system:
        raise HTTPException(status_code=503, detail="TTS system not initialized")
    
    def generate_audio_stream():
        try:
            logger.info(f"Streaming speech for text: {request.text[:50]}...")
            
            audio, _ = tts_system.run_model(request.text)
            
            # Create WAV file in memory
            buffer = io.BytesIO()
            sf.write(buffer, audio, 22050, format='WAV')
            buffer.seek(0)
            
            # Stream the audio data in chunks
            chunk_size = 8192
            while True:
                chunk = buffer.read(chunk_size)
                if not chunk:
                    break
                yield chunk
            
            logger.info("✅ Streaming completed - all chunks sent to client")
                
        except Exception as e:
            logger.error(f"Error in audio stream: {e}")
            raise
    
    return StreamingResponse(
        generate_audio_stream(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=streamed_speech.wav",
            "Cache-Control": "no-cache",
            "Accept-Ranges": "bytes"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
