# ============================================================
# Ontic AI — Backend (Modal + FastAPI + Qwen3-TTS 1.7B Base)
# Deploy: modal deploy modal_app.py
#
# GPU:          Nvidia L4 (24GB VRAM, native FlashAttention-2)
# Concurrency:  4 users per L4, max 5 L4s in the workspace
# Caching:      Model weights baked into the image at build time
#               Tester codes + generated WAVs in a persistent Volume
# ============================================================

import modal

# ---------- Modal App & Volumes ----------
app = modal.App("ontic-ai")

data_vol     = modal.Volume.from_name("ontic-ai-data",     create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("ontic-ai-hf-cache", create_if_missing=True)

DATA_DIR   = "/data"
CODES_FILE = f"{DATA_DIR}/codes.json"
OUTPUT_DIR = f"{DATA_DIR}/outputs"
HF_CACHE   = "/root/.cache/huggingface"

# ---------- Base Image ----------
base_image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime")
    .apt_install("ffmpeg", "sox", "libsox-dev")
    .pip_install(
        "qwen-tts",
        "fastapi",
        "python-multipart",
        "soundfile",
        "sox",
        "transformers",
        "uvicorn[standard]",
        "aiofiles",
        "huggingface_hub",
    )
)

# ---------- One-time model download (runs at image build) ----------
def _download_model():
    import os
    from huggingface_hub import snapshot_download
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    print("Pre-downloading Qwen3-TTS 1.7B Base into image cache...")
    snapshot_download(
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        cache_dir=cache_dir,
    )
    print("✅ Model cached at image build time.")

image = base_image.run_function(_download_model)

# ---------- Tester Codes ----------
INITIAL_CODES = {
    "ONTIC-7K3P": {"tts_left": 2, "clone_left": 1},
    "ONTIC-M9XQ": {"tts_left": 2, "clone_left": 1},
    "ONTIC-R4VT": {"tts_left": 2, "clone_left": 1},
    "ONTIC-H8NW": {"tts_left": 2, "clone_left": 1},
    "ONTIC-B2ZF": {"tts_left": 2, "clone_left": 1},
    "ONTIC-D5LC": {"tts_left": 2, "clone_left": 1},
    "ONTIC-J6YR": {"tts_left": 2, "clone_left": 1},
    "ONTIC-P3GS": {"tts_left": 2, "clone_left": 1},
    "ONTIC-T7AM": {"tts_left": 2, "clone_left": 1},
    "ONTIC-W9EB": {"tts_left": 2, "clone_left": 1},
}


# ============================================================
# Modal class-based app (required for concurrency controls)
# ============================================================
@app.cls(
    image=image,
    gpu="L4",                    # 24GB VRAM, native FlashAttention-2
    cpu=2.0,
    memory=8192,
    volumes={
        DATA_DIR: data_vol,
        HF_CACHE: hf_cache_vol,
    },
    timeout=1800,
    allow_concurrent_inputs=4,   # 4 users share a single L4
    concurrency_limit=5,         # cap total L4s at 5 (budget shield)
    scaledown_window=180,        # keep warm 3 min after last request
)
class QwenTTS:
    # Loaded once per container, reused across all requests
    @modal.enter()
    def load_model(self):
        import os, torch
        from qwen_tts import Qwen3TTSModel

        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(HF_CACHE, exist_ok=True)

        print("Loading Qwen3-TTS 1.7B Base from image cache …")
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",   # unlocks FlashAttention-2 on L4
        )
        print("✅ Model ready.")

    # ------------- Storage helpers -------------
    def _load_codes(self):
        import os, json
        if os.path.exists(CODES_FILE):
            try:
                with open(CODES_FILE, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        seed = {k: dict(v) for k, v in INITIAL_CODES.items()}
        return seed

    def _save_codes(self, codes):
        import os, json
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CODES_FILE, "w") as f:
            json.dump(codes, f, indent=2)
        data_vol.commit()

    # ------------- HTTP app -------------
    @modal.asgi_app()
    def web(self):
        import os, io, re, uuid
        import numpy as np
        import soundfile as sf
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.middleware.cors import CORSMiddleware

        # Seed codes file on first boot
        if not os.path.exists(CODES_FILE):
            self._save_codes(self._load_codes())

        fapp = FastAPI()
        fapp.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @fapp.get("/")
        def root():
            return {"status": "ok", "service": "Ontic AI", "gpu": "L4"}

        # -------- Verify tester code --------
        @fapp.post("/api/verify-code")
        async def verify_code(payload: dict):
            code = (payload.get("code") or "").strip().upper()
            codes = self._load_codes()
            if code not in codes:
                raise HTTPException(404, "Invalid code. Check your DM.")
            e = codes[code]
            return {
                "valid": True,
                "tts_left": e["tts_left"],
                "clone_left": e["clone_left"],
            }

        # -------- Text-to-speech --------
        @fapp.post("/api/generate")
        async def generate(
            code: str = Form(...),
            text: str = Form(...),
            voice: str = Form("Eric"),
            language: str = Form("English"),
        ):
            code = code.strip().upper()
            codes = self._load_codes()
            if code not in codes:
                raise HTTPException(404, "Invalid code")
            entry = codes[code]
            if entry["tts_left"] <= 0:
                raise HTTPException(403, "No TTS generations left on this code")

            text = text.strip()
            if not text:
                raise HTTPException(400, "Text cannot be empty")
            if len(text) > 500:
                raise HTTPException(400, "Text exceeds 500 characters")

            try:
                wavs, sr = self.model.generate_custom_voice(
                    text=text, language=language, speaker=voice,
                )
            except Exception as e:
                raise HTTPException(500, f"Generation failed: {e}")

            fname = f"{uuid.uuid4().hex}.wav"
            out_path = os.path.join(OUTPUT_DIR, fname)
            sf.write(out_path, np.asarray(wavs[0]).squeeze(), sr)

            entry["tts_left"] -= 1
            self._save_codes(codes)

            return {"audio_url": f"/api/audio/{fname}", "tts_left": entry["tts_left"]}

        # -------- Voice cloning --------
        @fapp.post("/api/clone")
        async def clone(
            code: str = Form(...),
            ref_audio: UploadFile = File(...),
            ref_text: str = Form(""),
            target_text: str = Form(...),
        ):
            code = code.strip().upper()
            codes = self._load_codes()
            if code not in codes:
                raise HTTPException(404, "Invalid code")
            entry = codes[code]
            if entry["clone_left"] <= 0:
                raise HTTPException(403, "No voice clones left on this code")

            target_text = target_text.strip()
            if not target_text:
                raise HTTPException(400, "Target text cannot be empty")
            if len(target_text) > 500:
                raise HTTPException(400, "Target text exceeds 500 characters")

            data = await ref_audio.read()
            try:
                arr, sr = sf.read(io.BytesIO(data))
            except Exception as e:
                raise HTTPException(400, f"Could not read reference audio: {e}")

            ref_path = f"/tmp/ref_{uuid.uuid4().hex}.wav"
            sf.write(ref_path, arr, sr)

            try:
                wavs, sr = self.model.generate_voice_clone(
                    text=target_text,
                    language="English",
                    ref_audio=ref_path,
                    ref_text=ref_text.strip() if ref_text and ref_text.strip() else None,
                )
            except Exception as e:
                raise HTTPException(500, f"Cloning failed: {e}")
            finally:
                try: os.remove(ref_path)
                except OSError: pass

            fname = f"{uuid.uuid4().hex}.wav"
            out_path = os.path.join(OUTPUT_DIR, fname)
            sf.write(out_path, np.asarray(wavs[0]).squeeze(), sr)

            entry["clone_left"] -= 1
            self._save_codes(codes)

            return {"audio_url": f"/api/audio/{fname}", "clone_left": entry["clone_left"]}

        # -------- Serve generated audio --------
        @fapp.get("/api/audio/{fname}")
        def get_audio(fname: str):
            if not re.match(r"^[a-f0-9]+\.wav$", fname):
                raise HTTPException(400, "Invalid filename")
            p = os.path.join(OUTPUT_DIR, fname)
            if not os.path.exists(p):
                raise HTTPException(404, "Audio not found")
            return FileResponse(p, media_type="audio/wav")

        return fapp
