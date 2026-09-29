"""
VoxCPM2 Backend for Modal — Streaming, Voice Design, and Voice Cloning.
Deploy with: modal deploy modal_backend.py
"""

import modal
import io
import uuid
import re
from pathlib import Path

# --- Configuration ---
APP_NAME = "voxcpm2-voiceover-backend"
MODEL_ID = "openbmb/VoxCPM2"
MODEL_REVISION = "bffb3df5a29440629464e5e839f4d214c8714c3d" # Pin for stability
MODEL_CACHE_PATH = "/models"
REFERENCE_CACHE_PATH = "/voice-references"

# --- Modal Infrastructure ---
app = modal.App(APP_NAME)

# Persistant volumes for caching model weights and voice references
model_volume = modal.Volume.from_name("voxcpm2-models", create_if_missing=True)
reference_volume = modal.Volume.from_name("voxcpm2-voices", create_if_missing=True)

# Custom image with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "clang", "build-essential")
    .uv_pip_install(
        "torch>=2.5.0",
        "torchaudio",
        "voxcpm==2.0.3",
        "soundfile==0.13.1",
        "huggingface-hub[hf-xet]==0.36.0",
        "librosa",
        "fastapi[standard]",
        "websockets"
    )
    .env({
        "HF_HOME": MODEL_CACHE_PATH,
        "HF_HUB_CACHE": MODEL_CACHE_PATH,
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)

# --- Helper Functions ---
def _safe_reference_id(value: str) -> str:
    """Validate the reference ID to prevent path traversal."""
    if not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid voice reference identifier.")
    return value

def _decode_audio_to_mono_16k(audio_bytes: bytes) -> tuple:
    """Decode arbitrary audio bytes into mono 16kHz float32 samples."""
    import librosa
    import soundfile as sf
    samples, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sample_rate != 16000:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
    return samples, 16000

# --- Modal Class: VoxCPM2 Runtime ---
@app.cls(
    image=image,
    gpu="L4", # Recommended for the best balance of speed and memory
    scaledown_window=15, # Terminate 15s after last activity
    enable_memory_snapshot=True, # Dramatically reduces cold-start time
    volumes={
        MODEL_CACHE_PATH: model_volume,
        REFERENCE_CACHE_PATH: reference_volume,
    },
    secrets=[modal.Secret.from_name("voxcpm2-secrets")] # Optional: for HF token
)
class VoxCPM2Runtime:
    
    @modal.enter()
    def load_model(self):
        """Load the model into VRAM on container start."""
        from huggingface_hub import snapshot_download
        from voxcpm import VoxCPM
        
        # Download to Volume (cached across container restarts)
        model_path = snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=f"{MODEL_CACHE_PATH}/VoxCPM2",
        )
        model_volume.commit()
        
        # Load model, keeping it lean
        self.model = VoxCPM.from_pretrained(
            model_path,
            load_denoiser=False, # We'll denoise reference audio manually if needed
            optimize=False,
            device="cuda",
        )
        self.sample_rate = int(self.model.tts_model.sample_rate)

    # --- Voice Reference Management ---
    @modal.method()
    def create_reference(self, audio_bytes: bytes, audio_suffix: str) -> dict:
        """Store a reference audio clip for voice cloning."""
        import soundfile as sf
        
        if not audio_bytes or len(audio_bytes) > 25 * 1024 * 1024:
            raise ValueError("Reference audio must be between 1 byte and 25 MB.")
        if audio_suffix not in {".flac", ".mp3", ".ogg", ".wav"}:
            raise ValueError("Unsupported reference audio format.")
        
        reference_id = uuid.uuid4().hex
        output_path = Path(REFERENCE_CACHE_PATH) / f"{reference_id}.wav"
        
        try:
            samples, sr = _decode_audio_to_mono_16k(audio_bytes)
            sf.write(output_path, samples, sr, format="WAV")
        except Exception as e:
            output_path.unlink(missing_ok=True)
            raise ValueError("VoxCPM2 could not decode the reference audio.") from e
        
        reference_volume.commit()
        return {
            "voice_id": reference_id,
            "provider": "openbmb-voxcpm2-modal",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    # --- Streaming Synthesis (WebSocket) ---
    @modal.method()
    async def synthesize_stream(self, websocket):
        """Bidirectional WebSocket handler for streaming TTS."""
        await websocket.accept()
        try:
            while True:
                # Receive a JSON message from the client
                data = await websocket.receive_json()
                action = data.get("action")
                
                if action == "generate":
                    text = data.get("text", "")
                    voice_id = data.get("voice_id")  # Optional: for cloning
                    design_prompt = data.get("design_prompt")  # Optional: for design
                    
                    # Prepare generation kwargs
                    gen_kwargs = {
                        "cfg_value": data.get("cfg_value", 2.0),
                        "inference_timesteps": data.get("inference_timesteps", 10),
                        "normalize": data.get("normalize", False),
                    }
                    
                    # Handle voice cloning
                    ref_path = None
                    if voice_id:
                        ref_path = str(Path(REFERENCE_CACHE_PATH) / f"{_safe_reference_id(voice_id)}.wav")
                        if not Path(ref_path).exists():
                            await websocket.send_json({"error": f"Voice ID {voice_id} not found."})
                            continue
                        gen_kwargs["reference_wav_path"] = ref_path
                    
                    # Handle voice design (prepend instruction to text)
                    if design_prompt:
                        text = f"({design_prompt}){text}"
                    
                    # Stream the generation
                    await websocket.send_json({"status": "generating"})
                    
                    # VoxCPM2's streaming API yields audio chunks
                    for chunk in self.model.generate_streaming(
                        text=text,
                        **gen_kwargs
                    ):
                        # Send raw PCM bytes (frontend can play via Web Audio API)
                        # We convert to int16 PCM for efficient WebSocket transport
                        import numpy as np
                        pcm_int16 = (chunk * 32767).astype(np.int16)
                        await websocket.send_bytes(pcm_int16.tobytes())
                    
                    await websocket.send_json({"status": "complete"})
                
                elif action == "ping":
                    await websocket.send_json({"status": "pong"})
                
                elif action == "close":
                    break
                    
        except Exception as e:
            try:
                await websocket.send_json({"error": str(e)})
            except:
                pass
        finally:
            await websocket.close()

    # --- REST Synthesis (for simple/non-streaming requests) ---
    @modal.method()
    def synthesize(self, text: str, voice_id: str = None, design_prompt: str = None, **kwargs) -> bytes:
        """Non-streaming synthesis for simple REST requests."""
        import soundfile as sf
        
        gen_kwargs = {
            "cfg_value": kwargs.get("cfg_value", 2.0),
            "inference_timesteps": kwargs.get("inference_timesteps", 10),
            "normalize": kwargs.get("normalize", False),
        }
        
        if voice_id:
            ref_path = str(Path(REFERENCE_CACHE_PATH) / f"{_safe_reference_id(voice_id)}.wav")
            if not Path(ref_path).exists():
                raise ValueError(f"Voice ID {voice_id} not found.")
            gen_kwargs["reference_wav_path"] = ref_path
        
        if design_prompt:
            text = f"({design_prompt}){text}"
        
        wav = self.model.generate(text=text, **gen_kwargs)
        
        # Write to in-memory WAV buffer
        buffer = io.BytesIO()
        sf.write(buffer, wav, self.sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()

# --- FastAPI Web App ---
@app.function(
    image=image,
    volumes={
        MODEL_CACHE_PATH: model_volume,
        REFERENCE_CACHE_PATH: reference_volume,
    },
)
@modal.asgi_app()
def web_app():
    from fastapi import FastAPI, UploadFile, File, Response, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    import uuid
    
    web = FastAPI(title="VoxCPM2 Voiceover API")
    
    # CORS for your Netlify frontend
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten this in production to your Netlify domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Get a handle to the runtime (lazy-loads on first use)
    runtime = VoxCPM2Runtime()
    
    @web.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_ID}
    
    @web.post("/v1/references")
    async def upload_reference(file: UploadFile = File(...)):
        """Upload a reference audio clip for voice cloning."""
        contents = await file.read()
        suffix = Path(file.filename).suffix.lower()
        result = runtime.create_reference.remote(contents, suffix)
        return result
    
    @web.post("/v1/speech")
    async def synthesize_speech(request: dict):
        """Non-streaming synthesis endpoint."""
        audio_bytes = runtime.synthesize.remote(
            text=request.get("text", ""),
            voice_id=request.get("voice_id"),
            design_prompt=request.get("design_prompt"),
            cfg_value=request.get("cfg_value", 2.0),
            inference_timesteps=request.get("inference_timesteps", 10),
            normalize=request.get("normalize", False),
        )
        return Response(content=audio_bytes, media_type="audio/wav")
    
    @web.websocket("/v1/stream")
    async def stream_speech(websocket: WebSocket):
        """WebSocket endpoint for streaming TTS."""
        await runtime.synthesize_stream.remote(websocket)
    
    return web
