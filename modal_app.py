"""VoxCPM2 backend for Modal — streaming TTS with voice design + cloning."""
import asyncio, io, re, threading, uuid
from pathlib import Path

import modal

APP_NAME = "voxcpm2-voiceover"
MODEL_ID = "openbmb/VoxCPM2"
MODEL_REVISION = "main"
MODEL_CACHE_PATH = "/models"
REFERENCE_CACHE_PATH = "/voices"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("voxcpm2-models", create_if_missing=True)
reference_volume = modal.Volume.from_name("voxcpm2-voices", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "clang", "build-essential")
    .uv_pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "voxcpm==2.0.3",
        "soundfile==0.13.1",
        "librosa==0.10.2.post1",
        "huggingface-hub[hf-xet]==0.36.0",
        "fastapi[standard]",
        "numpy<2.2",
    )
    .env({"HF_HOME": MODEL_CACHE_PATH, "HF_HUB_CACHE": MODEL_CACHE_PATH})
)


def _safe_ref(value: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{32}", value or ""):
        raise ValueError("Invalid voice reference id.")
    return value


def _decode_reference(audio_bytes: bytes):
    import librosa, soundfile as sf

    samples, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 16000:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
    return samples, 16000


@app.cls(
    image=image,
    gpu="L4",
    scaledown_window=15,
    enable_memory_snapshot=True,
    volumes={
        MODEL_CACHE_PATH: model_volume,
        REFERENCE_CACHE_PATH: reference_volume,
    },
)
class VoxCPM2Service:
    @modal.enter()
    def load(self):
        from huggingface_hub import snapshot_download
        from voxcpm import VoxCPM

        path = snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=f"{MODEL_CACHE_PATH}/VoxCPM2",
        )
        model_volume.commit()

        self.model = VoxCPM.from_pretrained(path, load_denoiser=False, device="cuda")
        self.sample_rate = int(self.model.tts_model.sample_rate)
        self.gpu_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Streaming generation — runs inside the GPU container
    # ------------------------------------------------------------------
    async def _generate_stream(self, ws, msg):
        import numpy as np

        chunks = [c for c in (msg.get("chunks") or []) if c and c.strip()]
        if not chunks:
            await ws.send_json({"error": "No text provided."})
            return

        voice_id = msg.get("voice_id")
        design = (msg.get("design_prompt") or "").strip()
        kwargs = {
            "cfg_value": float(msg.get("cfg_value", 2.0)),
            "inference_timesteps": int(msg.get("inference_timesteps", 10)),
            "normalize": bool(msg.get("normalize", False)),
        }

        if voice_id:
            ref = Path(REFERENCE_CACHE_PATH) / f"{_safe_ref(voice_id)}.wav"
            if not ref.exists():
                await ws.send_json({"error": f"Voice {voice_id} not found."})
                return
            kwargs["reference_wav_path"] = str(ref)

        await ws.send_json({
            "status": "start",
            "sample_rate": self.sample_rate,
            "chunk_count": len(chunks),
            "voice_id": voice_id,
        })

        loop = asyncio.get_running_loop()

        for index, raw in enumerate(chunks):
            text = f"({design}){raw}" if design else raw
            await ws.send_json({"status": "chunk_start", "index": index})

            queue: asyncio.Queue = asyncio.Queue()
            sentinel = object()

            def produce(text=text, kwargs=kwargs, queue=queue, sentinel=sentinel, loop=loop):
                try:
                    with self.gpu_lock:
                        for audio in self.model.generate_streaming(text=text, **kwargs):
                            loop.call_soon_threadsafe(queue.put_nowait, audio)
                except Exception as exc:  # surfaced to the client
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, sentinel)

            threading.Thread(target=produce, daemon=True).start()

            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                pcm = (
                    np.clip(np.asarray(item, dtype=np.float32), -1.0, 1.0) * 32767.0
                ).astype(np.int16)
                await ws.send_bytes(pcm.tobytes())

            await ws.send_json({"status": "chunk_done", "index": index})

        await ws.send_json({"status": "complete"})

    # ------------------------------------------------------------------
    # ASGI app, mounted on the GPU container
    # ------------------------------------------------------------------
    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, File, Response, UploadFile, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        import soundfile as sf

        api = FastAPI(title="VoxCPM2 Voiceover API")
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # tighten to your Netlify domain in production
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @api.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_ID, "sample_rate": self.sample_rate}

        @api.post("/v1/references")
        async def upload_reference(file: UploadFile = File(...)):
            data = await file.read()
            if not data or len(data) > 25 * 1024 * 1024:
                return Response("Reference must be 1 B – 25 MB.", status_code=400)
            if Path(file.filename or "").suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
                return Response("Unsupported format.", status_code=400)

            voice_id = uuid.uuid4().hex
            try:
                samples, sr = _decode_reference(data)
                sf.write(Path(REFERENCE_CACHE_PATH) / f"{voice_id}.wav", samples, sr)
            except Exception:
                return Response("Could not decode the reference audio.", status_code=400)

            reference_volume.commit()
            return {"voice_id": voice_id, "model_id": MODEL_ID}

        @api.post("/v1/speech")
        async def speech(payload: dict):
            import numpy as np

            text = (payload.get("text") or "").strip()
            if not text:
                return Response("Empty text.", status_code=400)

            design = (payload.get("design_prompt") or "").strip()
            kwargs = {
                "cfg_value": float(payload.get("cfg_value", 2.0)),
                "inference_timesteps": int(payload.get("inference_timesteps", 10)),
                "normalize": bool(payload.get("normalize", False)),
            }
            voice_id = payload.get("voice_id")
            if voice_id:
                kwargs["reference_wav_path"] = str(
                    Path(REFERENCE_CACHE_PATH) / f"{_safe_ref(voice_id)}.wav"
                )
            if design:
                text = f"({design}){text}"

            wav = await asyncio.to_thread(
                lambda: self.model.generate(text=text, **kwargs)
            )
            buf = io.BytesIO()
            sf.write(buf, wav, self.sample_rate, format="WAV")
            return Response(buf.getvalue(), media_type="audio/wav")

        @api.websocket("/v1/stream")
        async def stream(ws: WebSocket):
            await ws.accept()
            try:
                while True:
                    msg = await ws.receive_json()
                    action = msg.get("action")
                    if action == "generate":
                        await self._generate_stream(ws, msg)
                    elif action == "ping":
                        await ws.send_json({"status": "pong"})
                    elif action == "close":
                        break
            except WebSocketDisconnect:
                return
            except Exception as exc:
                try:
                    await ws.send_json({"error": str(exc)})
                except Exception:
                    pass
            finally:
                try:
                    await ws.close()
                except Exception:
                    pass

        return api
