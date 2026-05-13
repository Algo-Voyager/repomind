"""
Two services deployed together under the "qwen-7b-service" Modal app:

  1. QwenService   — Qwen2.5-7B-Instruct text generation  (GPU: A10G)
  2. embedding_api — BAAI/bge-small-en-v1.5 embeddings    (CPU, OpenAI-compatible)

─── Deploy ───────────────────────────────────────────────────────────────────
    venv/bin/modal deploy qwen_modal.py

    Modal prints two URLs, e.g.:
      Generation : https://iampkumar02--qwen-7b-service-qwenservice-generate.modal.run
      Embeddings : https://iampkumar02--qwen-7b-service-embedding-api.modal.run

    Set in .env:
      VLLM_BASE_URL=https://iampkumar02--qwen-7b-service-embedding-api.modal.run/v1
      VLLM_API_KEY=<your key>

─── Test generation ──────────────────────────────────────────────────────────
    curl -X POST <generation-url> \
         -H "Content-Type: application/json" \
         -H "Authorization: Bearer <key>" \
         -d '{"prompt": "What is a RAG pipeline?"}'

─── Test embeddings (OpenAI-compatible) ──────────────────────────────────────
    curl -X POST <embedding-url>/v1/embeddings \
         -H "Content-Type: application/json" \
         -H "Authorization: Bearer <key>" \
         -d '{"input": ["Hello world"], "model": "BAAI/bge-small-en-v1.5"}'

─── Stop ─────────────────────────────────────────────────────────────────────
    venv/bin/modal app stop qwen-7b-service
"""

import os
import modal
from dotenv import load_dotenv

load_dotenv()

try:
    from fastapi import FastAPI, Request, HTTPException
except ImportError:
    FastAPI = object
    Request = object
    HTTPException = Exception

# ── Generation model config ────────────────────────────────────────────────────
GEN_MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
GEN_MODEL_DIR = "/model-cache"

# ── Embedding model config ─────────────────────────────────────────────────────
EMBED_MODEL_ID  = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_DIR = "/embed-cache"

# ── Images ─────────────────────────────────────────────────────────────────────
gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.45.0",
        "torch>=2.2.0",
        "accelerate>=0.27.0",
        "huggingface_hub>=0.24.0",
        "fastapi[standard]",
    )
)

embed_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0.0",
        "torch>=2.2.0",
        "fastapi[standard]",
    )
)

# ── Volumes — weights cached so restarts are fast ─────────────────────────────
gen_volume   = modal.Volume.from_name("qwen-7b-weights-cache",  create_if_missing=True)
embed_volume = modal.Volume.from_name("bge-small-embed-cache",  create_if_missing=True)

# ── Modal secret — holds VLLM_API_KEY ────────────────────────────────────────
api_secret = modal.Secret.from_dict({"VLLM_API_KEY": os.environ.get("VLLM_API_KEY", "")})

# ── Modal app ──────────────────────────────────────────────────────────────────
app = modal.App("qwen-7b-service")


# ── 1. Generation service ──────────────────────────────────────────────────────
@app.cls(
    gpu="A10G",
    image=gen_image,
    volumes={GEN_MODEL_DIR: gen_volume},
    secrets=[api_secret],
    timeout=300,
    min_containers=1,
)
class QwenService:

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[startup] Loading tokenizer: {GEN_MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            GEN_MODEL_ID, cache_dir=GEN_MODEL_DIR
        )

        print(f"[startup] Loading model: {GEN_MODEL_ID}")
        self.model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=GEN_MODEL_DIR,
        )
        self.model.eval()
        gen_volume.commit()
        print("[startup] Generation model ready ✓")

    @modal.fastapi_endpoint(method="POST")
    def generate(self, req: dict, request: Request) -> dict:
        import torch

        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {os.environ['VLLM_API_KEY']}":
            raise HTTPException(status_code=401, detail="Unauthorized")

        prompt         = req.get("prompt", "")
        max_new_tokens = req.get("max_new_tokens", 256)
        temperature    = req.get("temperature", 0.7)
        top_p          = req.get("top_p", 0.9)

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return {"response": response, "model": GEN_MODEL_ID}


# ── 2. Embedding service (OpenAI-compatible /v1/embeddings) ───────────────────
@app.function(
    image=embed_image,
    volumes={EMBED_MODEL_DIR: embed_volume},
    secrets=[api_secret],
    min_containers=1,
    timeout=120,
)
@modal.asgi_app()
def embedding_api():
    from sentence_transformers import SentenceTransformer

    print(f"[startup] Loading embedding model: {EMBED_MODEL_ID}")
    model = SentenceTransformer(EMBED_MODEL_ID, cache_folder=EMBED_MODEL_DIR)
    embed_volume.commit()
    print("[startup] Embedding model ready ✓")

    web = FastAPI()

    @web.post("/v1/embeddings")
    async def create_embeddings(request: Request):
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {os.environ['VLLM_API_KEY']}":
            raise HTTPException(status_code=401, detail="Unauthorized")

        body  = await request.json()
        texts = body.get("input", [])
        if isinstance(texts, str):
            texts = [texts]

        vecs = model.encode(texts, normalize_embeddings=True).tolist()
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": v, "index": i}
                for i, v in enumerate(vecs)
            ],
            "model": EMBED_MODEL_ID,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    return web
