"""Deploy a vLLM OpenAI-compatible server on Modal.

--------------------------------------------------------------------------
One-time setup
--------------------------------------------------------------------------
    # 1. Install Modal and authenticate (writes ~/.modal.toml)
    .venv/bin/pip install modal
    .venv/bin/modal token new

    # 2. Hugging Face token — required for gated models (Llama, Mistral
    #    Instruct v0.3, etc.). For an open model like the default Qwen, you
    #    can still create the secret with an empty value; vLLM just won't
    #    use it. Modal requires the secret to EXIST.
    .venv/bin/modal secret create huggingface-secret HF_TOKEN=hf_xxx

    # 3. API key to protect the endpoint (optional but recommended).
    #    Leave empty if you want the server open; vLLM will skip auth.
    .venv/bin/modal secret create vllm-api-key VLLM_API_KEY=choose-a-long-random-string

--------------------------------------------------------------------------
Run
--------------------------------------------------------------------------
    # Ephemeral dev server — streams logs, stops on Ctrl-C
    .venv/bin/modal serve deploy/vllm_modal.py

    # Persistent deployment — stays up and scales to zero when idle
    .venv/bin/modal deploy deploy/vllm_modal.py

Modal prints a public URL like:
    https://<workspace>--repomind-vllm-serve.modal.run

--------------------------------------------------------------------------
Call it
--------------------------------------------------------------------------
    BASE=https://<workspace>--repomind-vllm-serve.modal.run
    KEY=$VLLM_API_KEY

    curl $BASE/v1/chat/completions \\
        -H "Authorization: Bearer $KEY" \\
        -H "Content-Type: application/json" \\
        -d '{"model": "Qwen/Qwen2.5-7B-Instruct",
             "messages": [{"role":"user","content":"hello"}]}'

Or from Python (drop-in with the OpenAI SDK):

    from openai import OpenAI
    client = OpenAI(base_url=f"{BASE}/v1", api_key=KEY)
    client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role":"user","content":"hello"}],
    )
"""

import modal

# --------------------------------------------------------------------------- #
# Model + hardware config. Edit these to change model or GPU.
# --------------------------------------------------------------------------- #
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"      # open, no HF gating
MODEL_REVISION = "main"
MAX_MODEL_LEN = 8192                         # context window

# For gated Llama 3.1 8B: set MODEL_NAME to "meta-llama/Meta-Llama-3.1-8B-Instruct"
# and make sure huggingface-secret contains a real HF_TOKEN with model access.

GPU_TYPE = "A10G"        # "A10G" (24GB) | "L4" (24GB) | "A100-40GB" | "H100"
N_GPU = 1
VLLM_VERSION = "0.6.4.post1"

MINUTES = 60

# --------------------------------------------------------------------------- #
# Container image
# --------------------------------------------------------------------------- #
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Cache model weights + vLLM compilation across cold starts.
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("repomind-vllm")


@app.function(
    image=vllm_image,
    gpu=GPU_TYPE if N_GPU == 1 else f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("vllm-api-key"),
    ],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=10 * MINUTES)
def serve() -> None:
    """Launch vLLM's built-in OpenAI-compatible server on port 8000."""
    import os
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--revision", MODEL_REVISION,
        "--tensor-parallel-size", str(N_GPU),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--served-model-name", MODEL_NAME,
        "--disable-log-requests",
    ]

    api_key = os.environ.get("VLLM_API_KEY", "").strip()
    if api_key:
        cmd += ["--api-key", api_key]

    # web_server starts the process; Modal proxies port 8000 to the public URL.
    subprocess.Popen(cmd)
