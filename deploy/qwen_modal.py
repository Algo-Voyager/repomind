"""Deploy Qwen2.5-7B-Instruct via vLLM on Modal — OpenAI-compatible endpoint.

Unlike rag-learning (which uses transformers directly), repomind needs a full
OpenAI-compatible /v1/chat/completions endpoint because agent.py uses the
openai SDK with VLLM_BASE_URL. vLLM's built-in server provides exactly that.

─── One-time setup ───────────────────────────────────────────────────────────
    # 1. Install Modal and authenticate
    .venv/bin/pip install modal
    .venv/bin/modal token new

    # 2. Hugging Face token (required for gated models; fine to leave empty for Qwen)
    .venv/bin/modal secret create huggingface-secret HF_TOKEN=hf_xxx

    # 3. API key to protect the endpoint
    .venv/bin/modal secret create vllm-api-key VLLM_API_KEY=choose-a-long-random-string

─── Deploy ───────────────────────────────────────────────────────────────────
    # Ephemeral — streams logs, stops on Ctrl-C
    .venv/bin/modal serve deploy/qwen_modal.py

    # Persistent — stays up, scales to zero when idle
    .venv/bin/modal deploy deploy/qwen_modal.py

    Modal prints a public URL, e.g.:
      https://<workspace>--repomind-vllm-serve.modal.run

    Set in .env:
      VLLM_BASE_URL=https://<workspace>--repomind-vllm-serve.modal.run/v1
      VLLM_API_KEY=<your key>
      VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct

─── Test (OpenAI-compatible) ─────────────────────────────────────────────────
    BASE=https://<workspace>--repomind-vllm-serve.modal.run
    KEY=$VLLM_API_KEY

    curl $BASE/v1/chat/completions \\
        -H "Authorization: Bearer $KEY" \\
        -H "Content-Type: application/json" \\
        -d '{"model": "Qwen/Qwen2.5-7B-Instruct",
             "messages": [{"role": "user", "content": "What is a ReAct agent?"}]}'

─── Stop ─────────────────────────────────────────────────────────────────────
    .venv/bin/modal app stop repomind-vllm

    Or run the cleanup script to also delete cached volumes:
    bash cleanup_modal.sh
"""

import modal

# ── Model + hardware ──────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-7B-Instruct"   # open model, no HF gating required
MODEL_REVISION = "main"
MAX_MODEL_LEN  = 8192                          # context window
VLLM_VERSION   = "0.6.4.post1"

GPU_TYPE = "A10G"   # "A10G" (24 GB) | "L4" (24 GB) | "A100-40GB" | "H100"
N_GPU    = 1

MINUTES = 60

# ── Container image ───────────────────────────────────────────────────────────
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ── Volumes — weights cached so restarts are fast ─────────────────────────────
hf_cache   = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache",        create_if_missing=True)

# ── Modal app ─────────────────────────────────────────────────────────────────
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
    """Launch vLLM's OpenAI-compatible server on port 8000.

    Modal proxies the port to the public URL printed after deploy.
    The /v1/chat/completions endpoint is used directly by agent.py via
    openai.OpenAI(base_url=VLLM_BASE_URL).
    """
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

    subprocess.Popen(cmd)
