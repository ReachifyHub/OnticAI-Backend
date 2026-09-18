# ============================================================
# Ontic AI — Backend (Modal + FastAPI + Qwen3-TTS 1.7B Base)
# Deploy: modal deploy modal_app.py
# ============================================================

import modal
import os, io, re, json, uuid
import numpy as np
import soundfile as sf

# ---------- Modal App & Volume ----------
app = modal.App("ontic-ai")
vol = modal.Volume.from_name("ontic-ai-data", create_if_missing=True)

DATA_DIR   = "/data"
CODES_FILE = f"{DATA_DIR}/codes.json"
OUTPUT_DIR = f"{DATA_DIR}/outputs"

# ---------- Container Image ----------
image = (
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
    )
)

# ---------- Tester Codes ----------
# Each tester gets 2 TTS generations + 1 voice clone.
# Send one code to each of your 10 testers.
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


# ---------- Code storage helpers ----------
def _load_codes():
    if os.path.exists(CODES_FILE):
        try:
            with open(CODES_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    # First boot: seed from INITIAL_CODES
    return {k: dict(v) for k, v in INITIAL_CODES.items()}


def _save_codes(codes: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CODES_FILE, "w") as f:
        json.dump(codes, f, indent=2)


# ---------- Modal function ----------
@app.function(
    image=image,
    gpu="T4",
    volumes={DATA_DIR: vol},
    timeout=1800,
    max_containers=1,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    import torch
    from qwen_tts import Qwen3TTSModel

    # Ensure dirs exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Seed codes file if missing
    if not os.path.exists(CODES_FILE):
        _save_codes(_load_codes())
        vol.commit()

    # Load the model once
    print("Loading Qwen3-TTS 1.7B Base …")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print("✅ Model ready.")

    fapp = FastAPI()
    fapp.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],       # Netlify URL — safe for beta
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------- Health check --------
    @fapp.get("/")
    def root():
        return {"status": "ok", "service": "Ontic AI"}

    # -------- Verify tester code --------
    @fapp.post("/api/verify-code")
    async def verify_code(payload: dict):
        code = (payload.get("code") or "").strip().upper()
        codes = _load_codes()
        if code not in codes:
            raise HTTPException(status_code=404, detail="Invalid code. Check your DM.")
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
        codes = _load_codes()
        if code not in codes:
            raise HTTPException(status_code=404, detail="Invalid code")

        entry = codes[code]
        if entry["tts_left"] <= 0:
            raise HTTPException(status_code=403, detail="No TTS generations left on this code")

        text = text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        if len(text) > 500:
            raise HTTPException(status_code=400, detail="Text exceeds 500 characters")

        try:
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=voice,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

        fname = f"{uuid.uuid4().hex}.wav"
        out_path = os.path.join(OUTPUT_DIR, fname)
        sf.write(out_path, np.asarray(wavs[0]).squeeze(), sr)

        entry["tts_left"] -= 1
        _save_codes(codes)
        vol.commit()

        return {
            "audio_url": f"/api/audio/{fname}",
            "tts_left": entry["tts_left"],
        }

    # -------- Voice cloning --------
    @fapp.post("/api/clone")
    async def clone(
        code: str = Form(...),
        ref_audio: UploadFile = File(...),
        ref_text: str = Form(""),
        target_text: str = Form(...),
    ):
        code = code.strip().upper()
        codes = _load_codes()
        if code not in codes:
            raise HTTPException(status_code=404, detail="Invalid code")

        entry = codes[code]
        if entry["clone_left"] <= 0:
            raise HTTPException(status_code=403, detail="No voice clones left on this code")

        target_text = target_text.strip()
        if not target_text:
            raise HTTPException(status_code=400, detail="Target text cannot be empty")
        if len(target_text) > 500:
            raise HTTPException(status_code=400, detail="Target text exceeds 500 characters")

        # Read reference audio
        data = await ref_audio.read()
        try:
            arr, sr = sf.read(io.BytesIO(data))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read reference audio: {e}")

        ref_path = f"/tmp/ref_{uuid.uuid4().hex}.wav"
        sf.write(ref_path, arr, sr)

        try:
            wavs, sr = model.generate_voice_clone(
                text=target_text,
                language="English",
                ref_audio=ref_path,
                ref_text=ref_text.strip() if ref_text and ref_text.strip() else None,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cloning failed: {e}")
        finally:
            try:
                os.remove(ref_path)
            except OSError:
                pass

        fname = f"{uuid.uuid4().hex}.wav"
        out_path = os.path.join(OUTPUT_DIR, fname)
        sf.write(out_path, np.asarray(wavs[0]).squeeze(), sr)

        entry["clone_left"] -= 1
        _save_codes(codes)
        vol.commit()

        return {
            "audio_url": f"/api/audio/{fname}",
            "clone_left": entry["clone_left"],
        }

    # -------- Serve generated audio --------
    @fapp.get("/api/audio/{fname}")
    def get_audio(fname: str):
        if not re.match(r"^[a-f0-9]+\.wav$", fname):
            raise HTTPException(status_code=400, detail="Invalid filename")
        p = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail="Audio not found")
        return FileResponse(p, media_type="audio/wav")

    return fapp
