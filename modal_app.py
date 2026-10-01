import base64
import io
import os
import tempfile

import modal

app = modal.App("omnivoice-tts")

# Persist the downloaded model weights so cold starts don't re-download them
hf_cache = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=True)

# Your Netlify site URL(s), no trailing slash. Use ["*"] while testing.
ALLOWED_ORIGINS = ["https://privoice.netlify.app"]

MAX_CHARS = 10000  # cap text length per request to limit GPU cost

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "omnivoice",
        "soundfile",
        "fastapi[standard]",
    )
    .env({"HF_HOME": "/cache"})
)


@app.cls(
    image=image,
    gpu="L4",
    volumes={"/cache": hf_cache},
    scaledown_window=80,  # seconds idle before the GPU shuts down
    max_containers=1,  # hard cap on parallel GPUs = cap on spend
    timeout=1200,
)
class TTS:
    @modal.enter()
    def load(self):
        import torch
        from omnivoice import OmniVoice

        self.model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda:0",
            dtype=torch.float16,
        )
        hf_cache.commit()

    @modal.asgi_app()
    def web(self):
        import soundfile as sf
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import Response
        from pydantic import BaseModel

        class Req(BaseModel):
            text: str
            instruct: str | None = None  # voice design, e.g. "female, british accent"
            ref_audio_b64: str | None = None  # voice cloning: base64 of 3-10s clip
            ref_text: str | None = None
            speed: float = 1.0
            num_step: int = 32  # 16 = faster, 32 = better

        api = FastAPI()
        api.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,
            allow_methods=["POST", "OPTIONS"],
            allow_headers=["*"],
        )

        @api.post("/tts")
        def tts(req: Req):
            if not req.text.strip():
                raise HTTPException(400, "Text is empty")
            if len(req.text) > MAX_CHARS:
                raise HTTPException(400, f"Text too long (max {MAX_CHARS} characters)")

            kwargs = dict(text=req.text, speed=req.speed, num_step=req.num_step)
            tmp_path = None

            if req.ref_audio_b64:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(base64.b64decode(req.ref_audio_b64))
                    tmp_path = f.name
                kwargs["ref_audio"] = tmp_path
                if req.ref_text:
                    kwargs["ref_text"] = req.ref_text
            elif req.instruct:
                kwargs["instruct"] = req.instruct

            try:
                audio = self.model.generate(**kwargs)
            finally:
                if tmp_path:
                    os.unlink(tmp_path)

            buf = io.BytesIO()
            sf.write(buf, audio[0], 24000, format="WAV")
            return Response(buf.getvalue(), media_type="audio/wav")

        return api
