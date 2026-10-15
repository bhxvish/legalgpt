# Training the local models on Google Colab (3B everyday, 7B heavy)

The laptop GPU (4 GB) can train only the 1.5B model. A free Colab **T4** (16 GB) can train
Qwen2.5-**3B** and **7B** with the same recipe (v0.7). After training, both run **on the laptop,
offline**:

| model | used for | runs on the laptop through | speed (RTX 3050 Ti, 4 GB) |
|---|---|---|---|
| 3B + LoRA | everyday questions ("Local (LoRA)") | PyTorch, 4-bit, fully on the GPU (~2.5 GB) | a few seconds |
| 7B + LoRA | harder questions ("Local heavy (7B)") | llama.cpp: 4-bit GGUF, ~16 layers on the GPU, the rest on the CPU | tens of seconds |

## 1. On the laptop: make the package

```bash
.venv/Scripts/python backend/scripts/colab/make_package.py
```

This writes `data/colab/legalgpt_colab.zip` (~3 MB): code, IPC corpus and index, benchmark, cached
teacher rewrites. It contains **no keys and no judgments**. `--with-judgments` adds the Module 1
judgment corpus, but only do that if you are allowed to upload those texts to Google Drive.

## 2. On Colab

1. Upload `legalgpt_colab.zip` to Google Drive, folder `MyDrive/legalgpt/`.
2. Open `backend/scripts/colab/LegalGPT_LoRA_Colab.ipynb` in Colab (File → Upload notebook).
3. Runtime → Change runtime type → **T4 GPU**.
4. Set `MODEL_SIZE = "3B"` and run all. When it finishes, change it to `"7B"` and run all again.

Checkpoints and benchmark answers are saved to Drive as they are made. If Colab disconnects,
reconnect and run all again: both resume where they stopped.

## 3. Back on the laptop

**3B (everyday):**
- Download `MyDrive/legalgpt/lora/v0.7-3b`, without `_trainer/`, to `data/models/lora/v0.7-3b`.
- In `.env`: `ADAPTER_PATH=data/models/lora/v0.7-3b`.
- The base model (Qwen2.5-3B-Instruct, ~6 GB) downloads once on first use.

**7B (heavy):**

1. Install llama.cpp's Python package into `.venv-gpu`. A CUDA build is needed for GPU offload;
   check https://github.com/abetlen/llama-cpp-python for the current wheel index for your CUDA
   version:
   ```bash
   .venv-gpu/Scripts/pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
   ```
2. Download a 4-bit GGUF of the **base** model (~4.7 GB) into `data/models/gguf/`, e.g.
   `Qwen2.5-7B-Instruct-Q4_K_M.gguf` from the Hugging Face repo `bartowski/Qwen2.5-7B-Instruct-GGUF`
   (or the official `Qwen/Qwen2.5-7B-Instruct-GGUF`). It must be the same model the adapter was
   trained on: Qwen2.5-7B-Instruct.
3. Copy `MyDrive/legalgpt/lora/v0.7-7b.gguf` (the adapter) into `data/models/gguf/`.
4. In `.env`:
   ```
   HEAVY_MODEL_GGUF=data/models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf
   HEAVY_LORA_GGUF=data/models/gguf/v0.7-7b.gguf
   HEAVY_GPU_LAYERS=16
   ```
   Lower `HEAVY_GPU_LAYERS` if the GPU runs out of memory; raise it if there is room.

**Fully offline:** `OFFLINE_MODE=true` in `.env`. The local models then also do the query rewrite
themselves, and nothing is sent to Groq or any other service. Fact-checking (Module 5) still needs
the hosted model.

**Signed answers (ZK):** the published manifest fingerprints one adapter (v0.4 today). After
switching `ADAPTER_PATH`, re-run `backend/scripts/zk_setup.py` so that signed answers carry the new
adapter's fingerprint. Until then they carry none.

**Results:** copy `MyDrive/legalgpt/bench/legalgpt-lora_v0.7-*.jsonl` into
`data/reports/bench/baselines/` and run `backend/scripts/run_baselines.py score` and
`backend/scripts/paired_tests.py`.

## Notes

- The Colab benchmark runs the 7B through PyTorch (4-bit). The laptop runs it through llama.cpp
  (Q4_K_M), a different 4-bit format, so answers can differ slightly. The paper should name the
  configuration it reports.
- Free Colab GPU time is limited per day. The 7B may need two sessions; checkpoints make that safe.
