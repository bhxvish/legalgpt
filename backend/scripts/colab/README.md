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

## 2. On Colab (one session, both models)

1. Upload `legalgpt_colab.zip` to Google Drive, folder `MyDrive/legalgpt/`.
2. Open `backend/scripts/colab/LegalGPT_LoRA_Colab.ipynb` in Colab (File → Upload notebook).
3. Runtime → Change runtime type → **L4 GPU**. A100 also works; T4 is about 3× slower.
4. **Run all.** The notebook:
   - trains the 3B, then the 7B, with the same recipe and 2 passes each;
   - benchmarks each one;
   - converts the 7B adapter to GGUF;
   - **disconnects the GPU** when done, so no units are spent idle.

**Time and cost:** about 4-5 h on an L4, roughly 20-25 compute units. *View resources* shows the
live rate.

**Recipe:** on a 24 GB GPU it trains on 4 examples at a time with 2 accumulation steps. That is the
same effective batch of 8 as the laptop's 1 × 8, only faster. If memory runs out, it retries with
2 × 4, then 1 × 8.

**If the session drops:** run all again. Finished steps are skipped, training resumes from its last
checkpoint, and the benchmark from its last saved answer. Runs go to new `…-l4` folders, so older
partial runs are left untouched.

## 3. Back on the laptop

**3B (everyday):**
- Download `MyDrive/legalgpt/lora/v0.7-3b-l4`, without `_trainer/`, to `data/models/lora/v0.7-3b-l4`.
- In `.env`: `ADAPTER_PATH=data/models/lora/v0.7-3b-l4`.
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
3. Copy `MyDrive/legalgpt/lora/v0.7-7b-l4.gguf` (the adapter) into `data/models/gguf/`.
4. In `.env`:
   ```
   HEAVY_MODEL_GGUF=data/models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf
   HEAVY_LORA_GGUF=data/models/gguf/v0.7-7b-l4.gguf
   HEAVY_GPU_LAYERS=16
   ```
   Lower `HEAVY_GPU_LAYERS` if the GPU runs out of memory; raise it if there is room.

**Fully offline:** `OFFLINE_MODE=true` in `.env`. The local models then also do the query rewrite
themselves, and nothing is sent to Groq or any other service. Fact-checking (Module 5) still needs
the hosted model.

**Signed answers (ZK):** the published manifest fingerprints one adapter (the 3B, v0.7-3b-l4, since 2026-10-15). After
switching `ADAPTER_PATH`, run `backend/scripts/zk_setup.py --adapter-only` so that signed answers carry the
new adapter's fingerprint (circuits and keys are not rebuilt). Until then they carry none.

**Results:** copy `MyDrive/legalgpt/bench/legalgpt-lora_v0.7-*.jsonl` into
`data/reports/bench/baselines/` and run `backend/scripts/run_baselines.py score` and
`backend/scripts/paired_tests.py`.

## Notes

- On an L4 the Colab benchmark runs both models through PyTorch in bfloat16 (the 7B fits in 24 GB).
  On the laptop the 3B runs in 4-bit and the 7B through llama.cpp (Q4_K_M), so answers can differ
  slightly. The paper should name the configuration it reports.
- The free T4 tier ran out of GPU time partway through the 3B. The paid L4 session above fits both models.

## Sentence-role classifier experiments (InLegalBERT)

`LegalGPT_RRL_ablation_Colab.ipynb` runs the context ablation (sentence only / + position /
+ neighbours / + both, 5-fold CV grouped by judgment) and the BiLSTM sequence model with its
ensemble, on the same folds and settings as the laptop. Build the upload with
`python backend/scripts/colab/make_rrl_package.py` (data/colab/legalgpt_rrl.zip: code + the v0.3
annotation snapshot), choose an L4 GPU, Run all, and bring back `rrl_results.zip` (unzip it into
the repo root: it fills data/models/rrl_cv/). About 30–40 minutes on an L4.
