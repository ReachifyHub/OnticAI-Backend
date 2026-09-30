"""
VoxCPM2 backend for Modal — vLLM-Omni runtime.
Deploy: modal deploy backend/modal_backend.py

Key differences from the previous version:
  - Uses `vllm serve openbmb/VoxCPM2 --omni` instead of the raw voxcpm package.
  - Exposes the OpenAI-compatible /v1/audio/speech endpoint directly.
  - Cold start is ~90s (vLLM init + torch.compile + CUDA Graph warmup),
    but the FIRST user request is ~2s instead of ~28s.
  - scaledown_window is 15s — the container dies quickly when idle.
"""

import modal

APP_NAME = "voxcpm2-voiceover"
MODEL_ID = "openbmb/VoxCPM2"
MODEL_CACHE_PATH = "/models"
REFERENCE_CACHE_PATH = "/voices"
VLLM_PORT = 8000

app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name("voxcpm2-models", create_if_missing=True)
reference_volume = modal.Volume.from_name("voxcpm2-voices", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "ninja-build", "git")
    .uv_pip_install(
        "vllm==0.21.0",
        "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git",
        "voxcpm>=2.0",
        "soundfile",
        "httpx",
        "ninja",
    )
    .env({
        "HF_HOME": MODEL_CACHE_PATH,
        "HF_HUB_CACHE": MODEL_CACHE_PATH,
        "VLLM_LOGGING_LEVEL": "INFO",
        # Let vLLM pick up the bundled voxcpm2.yaml deploy config automatically
        "VLLM_OMNI_DEPLOY_CONFIG": "vllm_omni/deploy/voxcpm2.yaml",
    })
)


@app.function(
    image=image,
    gpu="L4",                          # 24 GB — matches the recipe's minimum
    volumes={
        MODEL_CACHE_PATH: model_volume,
        REFERENCE_CACHE_PATH: reference_volume,
    },
    scaledown_window=15,               # kill 15s after last request
    timeout=600,                       # allow long generations
    startup_timeout=600,               # vLLM cold start can take ~90s
    max_containers=1,                  # one GPU is enough for a personal tool
)
@modal.web_server(port=VLLM_PORT, startup_timeout=600)
def serve():
    """
    Launch the vLLM-Omni OpenAI-compatible TTS server.

    vLLM-Omni auto-loads `vllm_omni/deploy/voxcpm2.yaml` from the model
    registry (HF model_type=voxcpm2). That config sets:
        gpu_memory_utilization: 0.9
        max_num_seqs: 4
        enforce_eager: true
    """
    import subprocess
    import os

    cmd = [
        "vllm", "serve", MODEL_ID,
        "--omni",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", "0.85",   # headroom for the VAE / CFM buffers
        "--max-model-len", "2048",            # generous for any TTS chunk
    ]

    env = os.environ.copy()
    # Point HuggingFace at the persistent volume so the 2B weights are cached
    env["HF_HOME"] = MODEL_CACHE_PATH
    env["HF_HUB_CACHE"] = MODEL_CACHE_PATH

    subprocess.Popen(cmd, env=env)
