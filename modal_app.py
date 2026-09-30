"""
VoxCPM2 TTS backend for Modal using vLLM-Omni.

Deploy:
    modal deploy modal_app.py

The container runs the official vLLM-Omni VoxCPM2 OpenAI-compatible
speech server on a single NVIDIA L4 GPU.

Important:
- Model files are cached on a persistent Modal Volume.
- vLLM stays attached to the Modal web-server process.
- vLLM stdout/stderr are inherited so startup logs appear in Modal.
- The container stays warm for 10 minutes after activity.
"""

from __future__ import annotations

import os
import signal
import subprocess

import modal


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "voxcpm2-voiceover"

MODEL_ID = "openbmb/VoxCPM2"

MODEL_CACHE_PATH = "/models"
REFERENCE_CACHE_PATH = "/voices"

VLLM_PORT = 8000


# ---------------------------------------------------------------------------
# Modal application
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)


# Persistent model cache.
#
# This prevents Hugging Face from downloading the model again whenever
# Modal creates a new container.
model_volume = modal.Volume.from_name(
    "voxcpm2-models",
    create_if_missing=True,
)


# Persistent reference-audio cache.
reference_volume = modal.Volume.from_name(
    "voxcpm2-voices",
    create_if_missing=True,
)


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "ffmpeg",
        "ninja-build",
        "git",
    )
    .uv_pip_install(
        "vllm==0.21.0",
        "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git",
        "voxcpm>=2.0",
        "soundfile",
        "httpx",
        "ninja",
    )
    .env(
        {
            # Hugging Face model cache
            "HF_HOME": MODEL_CACHE_PATH,
            "HF_HUB_CACHE": MODEL_CACHE_PATH,

            # vLLM logging
            "VLLM_LOGGING_LEVEL": "INFO",

            # VoxCPM2 deployment configuration supplied by vLLM-Omni
            "VLLM_OMNI_DEPLOY_CONFIG":
                "vllm_omni/deploy/voxcpm2.yaml",

            # Required/recommended for CUDA multiprocessing
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
)


# ---------------------------------------------------------------------------
# vLLM server
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="L4",

    volumes={
        MODEL_CACHE_PATH: model_volume,
        REFERENCE_CACHE_PATH: reference_volume,
    },

    # IMPORTANT:
    # Do not kill the GPU after 15 seconds.
    #
    # VoxCPM2 has a substantial initialization time. Keeping the container
    # warm means subsequent requests do not repeatedly reload the model.
    scaledown_window=600,

    # Long-running TTS requests are allowed.
    timeout=1800,

    # Give vLLM plenty of time to load VoxCPM2 during a cold start.
    startup_timeout=900,

    # This is a personal voice-generation backend.
    max_containers=1,
)
@modal.web_server(
    port=VLLM_PORT,
    startup_timeout=900,
)
def serve() -> None:
    """
    Start vLLM-Omni as the long-running web server.

    vLLM exposes the OpenAI-compatible API, including:

        POST /v1/audio/speech

    Do not return from this function while vLLM is still running.
    The subprocess must remain alive for the lifetime of the container.
    """

    cmd = [
        "vllm",
        "serve",
        MODEL_ID,

        # Enable vLLM-Omni.
        "--omni",

        # Bind to all interfaces so Modal can reach the server.
        "--host",
        "0.0.0.0",

        "--port",
        str(VLLM_PORT),

        # NVIDIA L4 has 24 GB VRAM.
        #
        # Leave some headroom because VoxCPM2 uses additional buffers
        # outside the main model weights.
        "--gpu-memory-utilization",
        "0.85",

        # More than enough context for normal TTS requests.
        "--max-model-len",
        "2048",
    ]

    env = os.environ.copy()

    # Explicitly guarantee that Hugging Face uses the persistent Modal
    # Volume rather than the container's ephemeral filesystem.
    env["HF_HOME"] = MODEL_CACHE_PATH
    env["HF_HUB_CACHE"] = MODEL_CACHE_PATH

    print("=" * 70, flush=True)
    print("[Modal] Starting VoxCPM2 vLLM-Omni server", flush=True)
    print(f"[Modal] Model: {MODEL_ID}", flush=True)
    print(f"[Modal] GPU: NVIDIA L4", flush=True)
    print(f"[Modal] Port: {VLLM_PORT}", flush=True)
    print("[Modal] Model cache: /models", flush=True)
    print("[Modal] Scaledown window: 600 seconds", flush=True)
    print("=" * 70, flush=True)

    print(
        "[Modal] vLLM startup logs will appear below.",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Start vLLM.
    #
    # stdout/stderr are intentionally NOT captured.
    #
    # This is important because we want vLLM's logs to flow directly into
    # Modal's logs. If vLLM crashes during model initialization, we should
    # see the actual exception instead of an apparently "stuck" invocation.
    # -----------------------------------------------------------------------

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=None,
        stderr=None,
    )

    # -----------------------------------------------------------------------
    # Forward container shutdown signals to vLLM.
    # -----------------------------------------------------------------------

    def forward_signal(
        signum: int,
        _frame: object,
    ) -> None:
        if process.poll() is None:
            print(
                f"[Modal] Forwarding signal {signum} to vLLM...",
                flush=True,
            )

            try:
                process.send_signal(signum)
            except ProcessLookupError:
                pass

    signal.signal(
        signal.SIGTERM,
        forward_signal,
    )

    signal.signal(
        signal.SIGINT,
        forward_signal,
    )

    # -----------------------------------------------------------------------
    # IMPORTANT:
    #
    # Keep the Modal function alive for as long as vLLM is alive.
    #
    # Do NOT use:
    #
    #     subprocess.Popen(...)
    #     return
    #
    # because that detaches the server lifecycle from this function.
    # -----------------------------------------------------------------------

    exit_code = process.wait()

    print(
        f"[Modal] vLLM process exited with code {exit_code}",
        flush=True,
    )

    if exit_code != 0:
        raise RuntimeError(
            f"vLLM exited unexpectedly with code {exit_code}"
        )


# ---------------------------------------------------------------------------
# Optional local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main() -> None:
    print(
        f"Deploy this application with:\n"
        f"    modal deploy modal_app.py"
    )
