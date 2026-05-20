"""
Shared constants and serialization utilities for the SAM2 3-device pipeline.

Topology (updated):
  RPi (source)  ──(raw frame and prompt)──▶  Orin:5555  (encoder)
  Orin          ──(backbone features)─────▶  NX:5556    (decoder)
  NX            ──(mask and metrics)──────▶  RPi:5557   (source)

Device roles:
  ORIN_IP  = 192.168.0.230  — Orin AGX   — image encoder (backbone FPN)
  NX_IP    = 192.168.0.18   — Jetson NX  — memory attention, mask decoder, and memory encoder
  RASP_IP  = 192.168.0.177  — RPi 5      — frame source / results sink
"""
import pickle
import numpy as np
import torch

# ── Device addresses ─────────────────────────────────────────────────────────
ORIN_IP  = "192.168.0.230"   # Orin AGX  — encoder node
NX_IP    = "192.168.0.18"    # Jetson NX — decoder node
RASP_IP  = "192.168.0.177"   # RPi 5     — source / sink

# ── ZMQ ports ─────────────────────────────────────────────────────────────────
SRC_PORT = 5555   # source PUSH → NX PULL   (raw frame)
ENC_PORT = 5556   # NX PUSH → Orin PULL     (encoded features)
OUT_PORT = 5557   # Orin PUSH → source PULL (mask and metrics)

# ── SAM2 paths (override via env vars SAM2_ROOT / SAM2_CKPT / SAM2_FT_CKPT) ──
import os as _os
SAM2_ROOT = _os.environ.get(
    "SAM2_ROOT",
    "/home/cps/Documents/biomed/ViT_part/segment-anything-2",
)
SAM2_CFG  = _os.environ.get(
    "SAM2_MODEL_CFG",
    "configs/sam2/sam2_hiera_t_512.yaml",  # default: tiny 512px
)
CKPT_PATH = _os.environ.get(
    "SAM2_CKPT",
    "/home/cps/Documents/biomed/ViT_part/checkpoints/sam2_hiera_tiny.pt",
)
# Fine-tuned macrophage weights (flat OrderedDict, loaded on top of base model)
FT_CKPT = _os.environ.get(
    "SAM2_FT_CKPT",
    "/home/cps/Documents/biomed/ViT_part/weights/tiny/sam2_macrophage_v2_best.pt",
)

# ── Model constants ───────────────────────────────────────────────────────────
IMG_SIZE = 512   # SAM2 input resolution
N_MEM    = 7     # memory bank depth (past frames)
HIDDEN_DIM = 256
MEM_DIM    = 64

# Normalisation (ImageNet)
PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PIXEL_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Serialisation ─────────────────────────────────────────────────────────────

def pack(obj) -> bytes:
    return pickle.dumps(obj, protocol=4)


def unpack(data: bytes):
    return pickle.loads(data)


def tensors_to_np(tensors):
    """List[Tensor] → list of numpy arrays (cpu, float16 for features)."""
    return [t.detach().cpu().half().numpy() for t in tensors]


def np_to_tensors(arrays, device, dtype=torch.float32):
    """list of numpy arrays → List[Tensor] on device."""
    return [torch.from_numpy(a.astype(np.float32)).to(device) for a in arrays]


def preprocess_image(image_np: np.ndarray, device) -> torch.Tensor:
    """uint8 (H,W,3) → float32 (1,3,H,W) normalised, on device."""
    img = image_np.astype(np.float32) / 255.0
    img = (img - PIXEL_MEAN) / PIXEL_STD
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def build_sam2_finetuned(device, ft_ckpt=None):
    """
    Build SAM2 model loaded with fine-tuned macrophage weights.
    Loads base checkpoint first, then overlays the fine-tuned flat state dict.
    Returns the full model in eval mode.
    """
    import sys as _sys
    _sys.path.insert(0, SAM2_ROOT)
    import os as _os2
    _os2.chdir(SAM2_ROOT)
    from sam2.build_sam import build_sam2

    cfg_name = SAM2_CFG.removesuffix(".yaml")
    model = build_sam2(cfg_name, ckpt_path=CKPT_PATH,
                       device=device, mode="eval", apply_postprocessing=False)

    ft = ft_ckpt or FT_CKPT
    if ft and _os.path.exists(ft):
        sd = torch.load(ft, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        if missing or unexpected:
            print(f"[WARN] Fine-tuned ckpt mismatch — missing:{len(missing)} unexpected:{len(unexpected)}")
        else:
            print(f"[INFO] Fine-tuned macrophage weights loaded from {ft}")
    else:
        print(f"[WARN] Fine-tuned checkpoint not found at {ft}, using base weights")

    model.eval()
    return model
