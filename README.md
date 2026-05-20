# ElasticTrack

**Adaptive SAM2 Cell Tracking on Distributed Edge Hardware**

> Paper: *Computers in Biology and Medicine* (under review)

ElasticTrack builds a real-time cell tracker as three progressive architectural layers on SAM2, each addressing a distinct challenge in edge deployment.

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer III — Long-Term Memory Oracle                     │
│  Per-track LTM cache (500 frames) with cosine retrieval │
│  Physically on Orin Nano; retrieved entries sent to NX  │
├─────────────────────────────────────────────────────────┤
│  Layer II — PE-Elastic Controller, Distributed Pipeline  │
│  PE-gated N_mem ∈ {0,2,4,7} · 3-device split inference │
│  Orin Nano (encoder) ──1.83MB──► Jetson NX (decoder)   │
├─────────────────────────────────────────────────────────┤
│  Layer I — LoRA Fine-Tuning                              │
│  SAM2-Hiera-Tiny with LoRA (r=4) on 4 CTC datasets     │
│  SEG: 0.324 → 0.887  ·  157K trainable params (0.4%)   │
└─────────────────────────────────────────────────────────┘
```

Each layer is independently ablatable and builds on the previous.

---

## Key Results

| Layer | Method | Mean SEG | FPS | Compute |
|-------|--------|----------|-----|---------|
| I | SAM2 base (GT-init, offline) | 0.324 | — | 100% |
| I | LoRA on CTC (GT-init, offline) | **0.887** | — | 100% |
| II | Fixed N=7, auto-detect, bidir | 0.705 | 0.073 | 100% |
| II | PE-Elastic controller | 0.709 | 0.073 | **90.9%** |
| II | 3-device pipeline (wired FP16) | 0.705 | **10.05** | 100% |
| III | Clustered short-term bank | **0.714** | 0.072 | 100% |
| III | LTM Oracle (k=5) | 0.706 | 0.072 | 100% |

---

## Repository Structure

```
ElasticTrack/
│
├── layer1_lora/
│   └── train_lora.py           # LoRA fine-tuning on CTC datasets
│
├── layer2_elastic/
│   ├── common.py               # shared constants, model builder, serialisation
│   ├── elastic_controller.py   # PE-gated N_mem controller with hysteresis
│   ├── elastic_bandit.py       # LinUCB online threshold adaptation
│   └── eval_autotrack_bidir.py # main tracker: auto-detect, bidir, elastic
│
├── layer3_ltm/
│   └── eval_autotrack_ltm.py  # Layer III: Long-Term Memory Oracle
│
├── eval/
│   └── paper_results.py        # CTC SEG evaluation and paper tables
│
├── configs/
│   └── default.yaml            # all hyperparameters
│
└── weights/                    # place weights here (not tracked by git)
```

---

## Setup

```bash
# Install SAM2
git clone https://github.com/facebookresearch/segment-anything-2
pip install -e segment-anything-2

# Install dependencies
pip install ultralytics opencv-python tifffile scipy

# On Jetson (Orin/NX), use the NV PyTorch wheel instead:
# pip install torch==2.5.0 --index-url https://developer.download.nvidia.com/...
```

---

## Layer I — LoRA Fine-Tuning

```bash
python layer1_lora/train_lora.py \
  --ctc-dirs /data/DIC-C2DH-HeLa /data/Fluo-C2DL-MSC \
             /data/Fluo-N2DH-GOWT1 /data/Fluo-N2DL-HeLa \
  --ctc-train-seqs 02 \
  --checkpoint weights/sam2.1_hiera_tiny.pt \
  --model_cfg configs/sam2/sam21_hiera_t_512.yaml \
  --epochs 30 --batch_size 2 --lr 3e-4 --rank 4 \
  --encoder-lora \
  --output weights/sam2.1_ctc_v1_enc.pt
```

---

## Layer II — Elastic Controller and Distributed Pipeline

```bash
# Offline bidirectional eval with PE-elastic controller:
SAM2_ROOT=/path/to/segment-anything-2 \
SAM2_FT_CKPT=weights/sam2.1_ctc_v1_enc.pt \
SAM2_CKPT=weights/sam2.1_hiera_tiny.pt \
CTC_DATA_ROOT=/path/to/datasets \
YOLO_CKPT=weights/yolo/best.pt \
python layer2_elastic/eval_autotrack_bidir.py

# Streaming bidirectional (K=5 lookahead, ~500ms delay):
STREAMING_K=5 python layer2_elastic/eval_autotrack_bidir.py

# Disable elastic (fixed N_mem=7):
DISABLE_ELASTIC=1 python layer2_elastic/eval_autotrack_bidir.py
```

---

## Layer III — Long-Term Memory Oracle

```bash
# LTM with k=3 retrieved frames per decode step:
ENABLE_LTM=1 LTM_N_RETRIEVE=3 \
python layer3_ltm/eval_autotrack_ltm.py

# Clustered short-term bank (best accuracy):
ENABLE_LTM=0 ENABLE_CLUSTERED_MEM=1 \
python layer3_ltm/eval_autotrack_ltm.py

# Both combined:
ENABLE_LTM=1 LTM_N_RETRIEVE=3 ENABLE_CLUSTERED_MEM=1 \
python layer3_ltm/eval_autotrack_ltm.py
```

---

## Hardware Topology (Layer II)

```
RPi 5 ──(raw frame)──► Orin Nano ──(1.83 MB/frame)──► Jetson NX ──(mask)──► RPi 5
          ZMQ:5555      [encoder]      ZMQ:5556        [decoder]    ZMQ:5557
                        [LTM cache ← Layer III]
```

Payload: 22 MB (naive FPN) → **1.83 MB** (pos_enc[-1] only) — 12× reduction from SAM2's architecture.

<!--
---

## Citation

```bibtex
@article{elastictrack2026,
  title   = {ElasticTrack: Adaptive SAM2 Cell Tracking on Distributed
             Edge Hardware},
  author  = {},
  journal = {Computers in Biology and Medicine},
  year    = {2026}
}
```
-->
---

## License

MIT
