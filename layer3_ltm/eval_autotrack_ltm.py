#!/usr/bin/env python3
"""
eval_autotrack_ltm.py — Dense-grid bidirectional autonomous cell tracking
                         with Long-Term Memory Oracle (Experiment 3).

Architecture: each track maintains two memory stores:
  - Short-term bank  (N_MEM=7): the standard SAM2 FIFO memory
  - Long-term cache (LTM_SIZE): ALL past memory entries, never evicted
    On every frame, the LTM retrieves LTM_N_RETRIEVE entries most similar
    to the current short-term bank (cosine sim in feature space) and injects
    them into memory_attention alongside the short-term entries.
    This gives the decoder access to the pre-division parent appearance even
    50+ frames after division — directly closing the HeLa bottleneck.

Original algorithm:

Algorithm:
  1. Forward pass  : dense 13×13×2 grid detects + tracks cells frame 0 → N
  2. Backward pass : same grid on reversed frames (N → 0), indices remapped
  3. Merge         : combine both track pools (offset IDs to avoid collision)
  4. Evaluate      : Hungarian assignment picks best track per GT cell from
                     combined pool — fwd tracks cover early frames, bwd tracks
                     cover cells missed early but found in later frames.

Fixes the eval_autoprompt_bidir.py bug where cells absent at frame 0 were
tracked from frame 0 with a GT fallback box, causing SAM2 to segment noise.
"""

import os, sys, json, glob, collections, time
import numpy as np
import cv2
import tifffile
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.optimize import linear_sum_assignment

_YOLO_CKPT = os.environ.get("YOLO_CKPT", "")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.environ.get("PIPELINE_DIR",
    "/home/cps/Documents/biomed/ViT_part_nano/pipeline"))
_SAM2_ROOT = os.environ.get(
    "SAM2_ROOT", "/home/cps/Documents/biomed/ViT_part/segment-anything-2")
sys.path.insert(0, _SAM2_ROOT)
os.chdir(_SAM2_ROOT)

from common import build_sam2_finetuned, preprocess_image, N_MEM, IMG_SIZE
from elastic_controller import ElasticController
from elastic_bandit import ElasticBandit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DATA_ROOT = os.environ.get("CTC_DATA_ROOT",
                            "/home/cps/Documents/biomed/datasets")
_MAX_TRACKS     = int(os.environ.get("MAX_ACTIVE_TRACKS_GLOBAL", "35"))
DISABLE_ELASTIC = os.environ.get("DISABLE_ELASTIC", "0") == "1"
DISABLE_MEMORY  = os.environ.get("DISABLE_MEMORY",  "0") == "1"  # SAM2 memoryless: empty bank, YOLO every frame
FORWARD_ONLY    = os.environ.get("FORWARD_ONLY",    "0") == "1"
STREAMING_K     = int(os.environ.get("STREAMING_K", "0"))  # >0 → streaming bidir with K-frame lookahead
ENABLE_BANDIT   = os.environ.get("ENABLE_BANDIT",  "0") == "1"  # LinUCB online threshold adaptation

SEQUENCES = [
    {"name": "DIC-C2DH-HeLa/01",
     "ctc_dir": f"{_DATA_ROOT}/DIC-C2DH-HeLa", "seq": "01",
     "yolo_init": True,    # YOLO frame-0 init (trained on HeLa seq02)
     "yolo_reprompt": True},  # YOLO re-prompting for degraded tracks
    {"name": "Fluo-N2DH-GOWT1/01",
     "ctc_dir": f"{_DATA_ROOT}/Fluo-N2DH-GOWT1", "seq": "01",
     "yolo_reprompt": True,
     # Small fluorescent dots: relax spawn distance, mask area, detection thresholds
     "overrides": {
         "MIN_SPAWN_DIST": 20, "MIN_MASK_AREA": 30, "MAX_MASK_AREA": 8000,
         "DET_IOU_THRESH": 0.30, "SPAWN_IOU_THRESH": 0.30,
         "MAX_ACTIVE_TRACKS": _MAX_TRACKS, "MATCH_CENTROID_DIST": 40,
         "MEM_MIN_IOU": 0.15,  # dim fluorescent cells → lower quality predictions
         "REPROMPT_IOU_THRESH": 0.20,  # lower rescue threshold for dim cells
     }},
    {"name": "Fluo-C2DL-MSC/01",
     "ctc_dir": f"{_DATA_ROOT}/Fluo-C2DL-MSC", "seq": "01",
     "yolo_reprompt": True},
]
OUT_DIR = Path(os.environ.get(
    "RESULTS_DIR",
    "/home/cps/Documents/biomed/experiment_three/results/autotrack_ltm"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Innovation flags ─────────────────────────────────────────────────────────
ENABLE_OF_PROMPT     = os.environ.get("ENABLE_OF_PROMPT",     "0") == "1"
ENABLE_TCADAPT       = os.environ.get("ENABLE_TCADAPT",       "0") == "1"
ENABLE_CLUSTERED_MEM = os.environ.get("ENABLE_CLUSTERED_MEM", "0") == "1"
OF_K_DETECT          = int(os.environ.get("OF_K_DETECT",      "15"))
TCADAPT_LR           = float(os.environ.get("TCADAPT_LR",     "1e-5"))
TCADAPT_STEPS        = int(os.environ.get("TCADAPT_STEPS",    "3"))
TCADAPT_INTERVAL     = int(os.environ.get("TCADAPT_INTERVAL", "5"))

# ── Long-Term Memory Oracle flags ─────────────────────────────────────────────
ENABLE_LTM       = os.environ.get("ENABLE_LTM",       "1") == "1"
LTM_SIZE         = int(os.environ.get("LTM_SIZE",     "500"))  # max entries per track
LTM_N_RETRIEVE   = int(os.environ.get("LTM_N_RETRIEVE", "3"))  # entries to inject per frame

# ── Hyper-parameters (same as AMG for fair comparison) ───────────────────────
K_DETECT            = 2
DENSE_GRID          = 13
DET_IOU_THRESH      = 0.50
SPAWN_IOU_THRESH    = 0.50
MIN_MASK_AREA       = 200
MAX_MASK_AREA       = 30_000
NMS_IOU_THRESH      = 0.40
MAX_DETECTIONS      = 60
MAX_ACTIVE_TRACKS   = _MAX_TRACKS
MATCH_IOU_THRESH    = 0.08
MATCH_CENTROID_DIST = 100
MIN_SPAWN_DIST      = 80
KILL_FRAMES         = 9999
LOW_IOU_KILL        = 0.001
EMPTY_KILL_FRAMES   = 9999
MEM_MIN_IOU         = 0.20   # skip memory update when SAM2 predicts IoU below this
YOLO_REPROMPT_INTERVAL = (1 if DISABLE_MEMORY else
                          0 if os.environ.get("DISABLE_YOLO_REPROMPT") else 10)
REPROMPT_IOU_THRESH    = 0.30 # tracks with last_iou below this are rescue candidates

_DEFAULTS = {k: v for k, v in list(globals().items())
             if k == k.upper() and isinstance(v, (int, float))}


def _apply_overrides(overrides: dict):
    """Temporarily override module-level hyperparameters for one sequence."""
    import sys
    mod = sys.modules[__name__]
    for k, v in overrides.items():
        setattr(mod, k, v)


def _reset_defaults():
    import sys
    mod = sys.modules[__name__]
    for k, v in _DEFAULTS.items():
        setattr(mod, k, v)


# ── CellTrack ─────────────────────────────────────────────────────────────────

class CellTrack:
    __slots__ = ("track_id", "memory_bank", "ltm", "prev_logits", "last_mask",
                 "last_iou", "predictions", "frames_alive",
                 "frames_without_match", "reanchor_det", "alive",
                 "last_valid_centroid", "consecutive_empty")

    def __init__(self, track_id, memory_bank, prev_logits, last_mask, last_iou, fi):
        self.track_id             = track_id
        self.memory_bank          = memory_bank
        self.prev_logits          = prev_logits
        self.last_mask            = last_mask
        self.last_iou             = last_iou
        self.predictions          = {fi: last_mask}
        self.frames_alive         = 1
        self.frames_without_match = 0
        self.reanchor_det         = None
        self.alive                = True
        self.consecutive_empty    = 0
        ys, xs = np.where(last_mask)
        self.last_valid_centroid  = (
            (float(xs.mean()), float(ys.mean())) if len(xs) > 0
            else (IMG_SIZE / 2, IMG_SIZE / 2))


# ── Option 3: Clustered Memory Bank ──────────────────────────────────────────

class ClusteredMemoryBank:
    """
    Drop-in replacement for collections.deque(maxlen=N_MEM).
    Maintains a pool of up to pool_size frames; when the pool overflows it
    uses greedy farthest-point sampling in feature space to keep maxlen
    maximally-diverse representatives.  This gives O(maxlen) memory with
    O(T) temporal coverage and naturally handles cell division: pre- and
    post-division appearances form distinct clusters so the backward pass
    links to the correct temporal context.
    """
    def __init__(self, maxlen=7, pool_size=None):
        self.maxlen    = maxlen
        self.pool_size = pool_size or max(maxlen * 3, 21)
        self._pool: list = []

    def append(self, entry):
        self._pool.append(entry)
        if len(self._pool) > self.pool_size:
            self._pool = self._select()

    def _select(self):
        if len(self._pool) <= self.maxlen:
            return list(self._pool)
        feats = torch.stack(
            [e["feat"].flatten().float() for e in self._pool])
        feats = F.normalize(feats, dim=1)
        # Always keep first (oldest) and last (newest)
        selected = [0, len(self._pool) - 1]
        remaining = list(range(1, len(self._pool) - 1))
        while len(selected) < self.maxlen and remaining:
            sel_f = feats[selected]
            rem_f = feats[remaining]
            # Maximum dissimilarity: pick frame least similar to any selected
            max_sim = (rem_f @ sel_f.T).max(dim=1).values
            pick    = int(max_sim.argmin())
            selected.append(remaining.pop(pick))
        return [self._pool[i] for i in sorted(selected)]

    def _effective(self):
        return self._select() if len(self._pool) > self.maxlen else self._pool

    def __len__(self):
        return len(self._effective())

    def __iter__(self):
        return iter(self._effective())

    def __getitem__(self, idx):
        return self._effective()[idx]

    def __bool__(self):
        return len(self._pool) > 0


def _make_memory_bank(maxlen):
    if ENABLE_CLUSTERED_MEM:
        return ClusteredMemoryBank(maxlen=maxlen)
    return collections.deque(maxlen=maxlen)


# ── Long-Term Memory Oracle ───────────────────────────────────────────────────

class LongTermMemoryCache:
    """
    Per-track long-term memory cache for Experiment 3.

    Stores ALL past memory entries (feat + pos tensors from memory_encoder).
    On each decode step, retrieves the LTM_N_RETRIEVE entries most similar
    to the current short-term bank (cosine similarity in flattened feature
    space), which are then concatenated with the short-term entries before
    memory_attention.

    Key property: after a cell divides, the daughter track's short-term bank
    only has post-division frames.  The LTM holds the pre-division parent
    appearance, and cosine retrieval surfaces it when the daughter's mask
    starts drifting — giving SAM2 the "what did this cell look like before"
    context that a FIFO bank permanently loses after 7 frames.
    """
    def __init__(self, max_size=500, n_retrieve=3):
        self._pool: list = []
        self.max_size  = max_size
        self.n_retrieve = n_retrieve

    def add(self, entry):
        self._pool.append(entry)
        if len(self._pool) > self.max_size:
            self._pool.pop(0)

    def retrieve(self, short_term_entries):
        """Return up to n_retrieve entries from LTM not in the short-term bank.

        Query = mean of short-term feat vectors.
        Returns empty list if LTM has nothing beyond what short-term already holds.
        """
        n_st = len(short_term_entries)
        # Only look at frames older than the short-term window
        candidates = self._pool[:-n_st] if n_st < len(self._pool) else []
        if not candidates or not short_term_entries:
            return []

        q_feats = torch.stack(
            [e["feat"].flatten().float() for e in short_term_entries])
        q = F.normalize(q_feats.mean(0, keepdim=True), dim=1)

        c_feats = torch.stack([e["feat"].flatten().float() for e in candidates])
        c_feats = F.normalize(c_feats, dim=1)

        sims = (c_feats @ q.T).squeeze(1)
        k    = min(self.n_retrieve, len(candidates))
        idxs = sims.topk(k).indices.tolist()
        return [candidates[i] for i in idxs]


# ── Option 1: Optical Flow Prompt Propagation ─────────────────────────────────

def compute_lk_flow(prev_gray, curr_gray, points):
    """Propagate 2D (x,y) points via Lucas-Kanade pyramid optical flow.
    Returns (new_points, status) — status[i]==1 means point i was tracked."""
    if len(points) == 0:
        return np.empty((0, 2), np.float32), np.empty(0, np.uint8)
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    return new_pts.reshape(-1, 2), status.flatten()


def warp_mask_with_flow(mask_np, flow):
    """Warp a boolean H×W mask to the next frame using a dense flow field."""
    h, w  = mask_np.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(mask_np.astype(np.float32), map_x, map_y,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped > 0.5


def compute_dense_flow(prev_gray, curr_gray):
    """Farneback dense optical flow; returns (H,W,2) float32 displacement."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


# ── Option 2: Temporal Consistency Adaptation ─────────────────────────────────

def setup_tcadapt(model):
    """Identify mask-decoder attention params and create optimizer."""
    params = [p for n, p in model.named_parameters()
              if "sam_mask_decoder" in n
              and any(k in n for k in ("transformer", "upscal", "output_hyper"))]
    for p in params:
        p.requires_grad_(True)
    optim = torch.optim.AdamW(params, lr=TCADAPT_LR, weight_decay=0.0)
    print(f"[TCADAPT] {sum(p.numel() for p in params)/1e3:.1f}K params registered")
    return optim


def adapt_step_temporal(model, tracks, fpn, curr_pos, flow, optim, n_steps):
    """
    For each alive track with a valid previous mask, warp that mask to the
    current frame and use it as a self-supervised target for the mask decoder.
    Loss: Dice(decoder_pred, warped_prev_mask).
    """
    if flow is None or not tracks:
        return
    for track in tracks:
        if not track.alive or track.last_mask.sum() < 50:
            continue
        warped = warp_mask_with_flow(track.last_mask, flow)
        if warped.sum() < 50:
            continue
        warped_t = torch.tensor(warped, dtype=torch.float32,
                                device=DEVICE).unsqueeze(0).unsqueeze(0)

        B, C, H, W    = fpn[-1].shape
        curr_feat     = fpn[-1].flatten(2).permute(2, 0, 1)
        curr_pos_flat = curr_pos.flatten(2).permute(2, 0, 1)

        if len(track.memory_bank) > 0:
            mem_feats = torch.cat(
                [m["feat"].flatten(2).permute(2, 0, 1) for m in track.memory_bank], 0)
            mem_pos   = torch.cat(
                [m["pos"].flatten(2).permute(2, 0, 1) for m in track.memory_bank], 0)
            attended  = model.memory_attention(
                curr=curr_feat, curr_pos=curr_pos_flat,
                memory=mem_feats, memory_pos=mem_pos)
        else:
            attended = curr_feat
        att_2d = attended.permute(1, 2, 0).view(B, C, H, W).detach()

        for _ in range(n_steps):
            optim.zero_grad()
            with torch.enable_grad():
                sparse_emb, dense_emb = model.sam_prompt_encoder(
                    points=None, boxes=None,
                    masks=track.prev_logits.detach())
                lr, _, _, _ = model.sam_mask_decoder(
                    image_embeddings=att_2d,
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False, repeat_image=False,
                    high_res_features=[fpn[0], fpn[1]])
                pred = torch.sigmoid(F.interpolate(
                    lr.float(), size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear", align_corners=False))
                inter = (pred * warped_t).sum()
                loss  = 1.0 - 2.0 * inter / (pred.sum() + warped_t.sum() + 1e-6)
                loss.backward()
            optim.step()


# ── Image / GT loading ────────────────────────────────────────────────────────

def load_sequence(seq_dir, jpg_dir):
    Path(jpg_dir).mkdir(parents=True, exist_ok=True)
    tifs  = sorted(glob.glob(os.path.join(seq_dir, "t*.tif")))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    for tif in tifs:
        idx = int(Path(tif).stem[1:])
        img = tifffile.imread(tif)
        if img.dtype != np.uint8:
            img = ((img.astype(np.float32) - img.min()) /
                   max(img.max() - img.min(), 1) * 255).astype(np.uint8)
        if img.ndim == 2:
            img = clahe.apply(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(jpg_dir, f"{idx:05d}.jpg"), img)
    return sorted(glob.glob(os.path.join(jpg_dir, "*.jpg")))


def load_gt(gt_seg_dir):
    gt = {}
    for f in sorted(glob.glob(os.path.join(gt_seg_dir, "man_seg*.tif"))):
        idx = int(Path(f).stem.replace("man_seg", ""))
        m   = tifffile.imread(f)
        cells = {}
        for cid in np.unique(m):
            if cid == 0:
                continue
            cm = cv2.resize((m == cid).astype(np.uint8), (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            cells[int(cid)] = cm
        gt[idx] = cells
    return gt


def iou_score(pred, gt_m):
    p, g = pred.astype(bool), gt_m.astype(bool)
    return float((p & g).sum()) / float((p | g).sum() + 1e-8)


# ── Dense dual-offset grid detection (same as AMG) ───────────────────────────

@torch.no_grad()
def detect_from_fpn(model, fpn, device):
    step = IMG_SIZE // (DENSE_GRID + 1)
    half = step // 2
    raw  = []

    for ox, oy in [(0, 0), (half, half)]:
        for row in range(1, DENSE_GRID + 1):
            for col in range(1, DENSE_GRID + 1):
                px = min(col * step + ox, IMG_SIZE - 1)
                py = min(row * step + oy, IMG_SIZE - 1)
                pts_t = torch.tensor([[[px, py]]], dtype=torch.float32, device=device)
                lbs_t = torch.tensor([[1]],        dtype=torch.int32,   device=device)

                sparse_emb, dense_emb = model.sam_prompt_encoder(
                    points=(pts_t, lbs_t), boxes=None, masks=None)
                lr, iou_p, _, _ = model.sam_mask_decoder(
                    image_embeddings=fpn[-1],
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=[fpn[0], fpn[1]])

                best_idx = iou_p.squeeze().argmax().item()
                best_iou = float(iou_p.squeeze()[best_idx])
                if best_iou < DET_IOU_THRESH:
                    continue

                mask = (F.interpolate(
                    lr[:, best_idx:best_idx+1].float(),
                    size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
                ).squeeze() > 0.0).cpu().numpy()

                area = int(mask.sum())
                if area < MIN_MASK_AREA or area > MAX_MASK_AREA:
                    continue

                ys, xs = np.where(mask)
                cx, cy = float(xs.mean()), float(ys.mean())
                raw.append({
                    "mask":     mask,
                    "bbox":     [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "iou":      best_iou,
                    "centroid": (cx, cy),
                    "logits":   lr[:, best_idx:best_idx+1].detach(),
                })

    return _nms(raw)


def _nms(dets):
    dets = sorted(dets, key=lambda x: -x["iou"])
    keep = []
    for d in dets:
        if len(keep) >= MAX_DETECTIONS:
            break
        dup = any(
            (d["mask"] & k["mask"]).sum() /
            max((d["mask"] | k["mask"]).sum(), 1) > NMS_IOU_THRESH
            for k in keep)
        if not dup:
            keep.append(d)
    return keep


# ── YOLO-based detection (frame-0 initialisation) ────────────────────────────

def detect_from_yolo(yolo_model, frame_bgr):
    """
    Run YOLO on a BGR frame; return dets compatible with spawn_track.
    No pre-computed SAM2 mask — spawn_track computes it via SAM2 box+point prompt.
    """
    results = yolo_model(frame_bgr, verbose=False, imgsz=IMG_SIZE, conf=0.15)
    dets = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [float(c) for c in box.xyxy[0].tolist()]
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(IMG_SIZE - 1), x2), min(float(IMG_SIZE - 1), y2)
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dets.append({
                "centroid": (cx, cy),
                "bbox":     [x1, y1, x2, y2],
                "iou":      SPAWN_IOU_THRESH,  # guaranteed to pass spawn threshold
                "mask":     None,              # computed inside spawn_track
            })
    return dets


# ── Hungarian matching ────────────────────────────────────────────────────────

def match_dets_to_tracks(dets, active_tracks):
    alive = [t for t in active_tracks if t.alive]
    if not alive or not dets:
        return list(dets)

    score_mat = np.zeros((len(alive), len(dets)), dtype=np.float32)
    for i, track in enumerate(alive):
        tcx, tcy = track.last_valid_centroid
        for j, det in enumerate(dets):
            dcx, dcy = det["centroid"]
            dist = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5
            if det["mask"] is not None:
                inter = (track.last_mask & det["mask"]).sum()
                union = (track.last_mask | det["mask"]).sum()
                iou   = inter / max(union, 1)
            else:
                iou = 0.0
            centroid_score = max(0.0, 1.0 - dist / MATCH_CENTROID_DIST)
            score_mat[i, j] = max(iou, 0.5 * centroid_score)

    row_ind, col_ind = linear_sum_assignment(-score_mat)
    matched_det_idxs = set()
    for ri, ci in zip(row_ind, col_ind):
        det  = dets[ci]
        dcx, dcy = det["centroid"]
        tcx, tcy = alive[ri].last_valid_centroid
        dist = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5
        if det["mask"] is not None:
            iou = (alive[ri].last_mask & det["mask"]).sum() / max(
                   (alive[ri].last_mask | det["mask"]).sum(), 1)
        else:
            iou = 0.0
        if iou >= MATCH_IOU_THRESH or dist < MATCH_CENTROID_DIST:
            alive[ri].reanchor_det         = det
            alive[ri].frames_without_match = 0
            matched_det_idxs.add(ci)

    return [d for j, d in enumerate(dets) if j not in matched_det_idxs]


# ── Spawn / Update tracks ─────────────────────────────────────────────────────

@torch.no_grad()
def spawn_track(model, fpn, curr_pos, det, track_id, n_mem, fi, device):
    mem_bank   = _make_memory_bank(max(1, n_mem))
    B, C, H, W = fpn[-1].shape
    curr_feat  = fpn[-1].flatten(2).permute(2, 0, 1)
    att_2d     = curr_feat.permute(1, 2, 0).view(B, C, H, W)

    cx, cy = det["centroid"]
    pts_t  = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=device)
    lbs_t  = torch.tensor([[1]],        dtype=torch.int32,   device=device)
    box_arr = np.array(det["bbox"], dtype=np.float32).reshape(1, 4)
    box_t   = torch.from_numpy(box_arr).to(dtype=torch.float32, device=device).unsqueeze(0)

    sparse_emb, dense_emb = model.sam_prompt_encoder(
        points=(pts_t, lbs_t), boxes=box_t, masks=None)
    lr, iou_p, _, _ = model.sam_mask_decoder(
        image_embeddings=att_2d,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=True, repeat_image=False,
        high_res_features=[fpn[0], fpn[1]])

    best_idx = iou_p.squeeze().argmax().item()
    mask_lr  = lr[:, best_idx:best_idx+1]
    mask_np  = (F.interpolate(
        mask_lr.float(), size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear", align_corners=False).squeeze() > 0.0).cpu().numpy()

    pred_up = F.interpolate(mask_lr, size=(IMG_SIZE, IMG_SIZE),
                             mode="bilinear", align_corners=False)
    mem_out = model.memory_encoder(fpn[-1], pred_up, skip_mask_sigmoid=False)
    entry = {"feat": mem_out["vision_features"], "pos": mem_out["vision_pos_enc"][0]}
    mem_bank.append(entry)

    track = CellTrack(track_id=track_id, memory_bank=mem_bank,
                      prev_logits=mask_lr.detach(), last_mask=mask_np,
                      last_iou=float(iou_p.squeeze()[best_idx]), fi=fi)
    track.ltm = LongTermMemoryCache(max_size=LTM_SIZE, n_retrieve=LTM_N_RETRIEVE)
    if ENABLE_LTM:
        track.ltm.add(entry)
    return track


@torch.no_grad()
def reprompt_track(model, track, fpn, det, n_mem, fi, device):
    """Reset a degraded track using a fresh YOLO box prompt.
    Clears the memory bank and re-initialises from the YOLO detection,
    keeping the same track_id so assignment history is preserved.
    """
    B, C, H, W = fpn[-1].shape
    att_2d     = fpn[-1]  # (B, C, H, W) — same layout as spawn_track

    cx, cy = det["centroid"]
    pts_t  = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=device)
    lbs_t  = torch.tensor([[1]],        dtype=torch.int32,   device=device)
    box_arr = np.array(det["bbox"], dtype=np.float32).reshape(1, 4)
    box_t   = torch.from_numpy(box_arr).to(dtype=torch.float32, device=device).unsqueeze(0)

    sparse_emb, dense_emb = model.sam_prompt_encoder(
        points=(pts_t, lbs_t), boxes=box_t, masks=None)
    lr, iou_p, _, _ = model.sam_mask_decoder(
        image_embeddings=att_2d,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=True, repeat_image=False,
        high_res_features=[fpn[0], fpn[1]])

    best_idx = iou_p.squeeze().argmax().item()
    new_iou  = float(iou_p.squeeze()[best_idx])

    # Only reprompt if the fresh mask is meaningfully better than current state
    if new_iou <= track.last_iou + 0.05:
        return

    mask_lr  = lr[:, best_idx:best_idx+1]
    mask_np  = (F.interpolate(
        mask_lr.float(), size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear", align_corners=False).squeeze() > 0.0).cpu().numpy()

    # Reset short-term bank — LTM is preserved (retains pre-reprompt history)
    track.memory_bank = _make_memory_bank(max(1, n_mem))
    pred_up = F.interpolate(mask_lr, size=(IMG_SIZE, IMG_SIZE),
                             mode="bilinear", align_corners=False)
    mem_out = model.memory_encoder(fpn[-1], pred_up, skip_mask_sigmoid=False)
    entry = {"feat": mem_out["vision_features"], "pos": mem_out["vision_pos_enc"][0]}
    track.memory_bank.append(entry)
    if ENABLE_LTM:
        track.ltm.add(entry)

    track.prev_logits  = mask_lr.detach()
    track.last_mask    = mask_np
    track.last_iou     = new_iou
    track.frames_without_match = 0
    track.consecutive_empty    = 0
    xs, ys = mask_np.nonzero()
    if xs.size:
        track.last_valid_centroid = (float(ys.mean()), float(xs.mean()))


@torch.no_grad()
def update_track(model, track, fpn, curr_pos, n_mem, fi, device):
    if n_mem > 0 and n_mem != track.memory_bank.maxlen:
        entries = list(track.memory_bank)[-n_mem:]
        track.memory_bank = _make_memory_bank(n_mem)
        for e in entries:
            track.memory_bank.append(e)
    elif n_mem == 0:
        track.memory_bank = _make_memory_bank(1)

    B, C, H, W    = fpn[-1].shape
    curr_feat     = fpn[-1].flatten(2).permute(2, 0, 1)
    curr_pos_flat = curr_pos.flatten(2).permute(2, 0, 1)

    if len(track.memory_bank) == 0:
        attended = curr_feat
    else:
        short_entries = list(track.memory_bank)
        # Retrieve relevant distant-past frames from LTM and append
        ltm_entries = (track.ltm.retrieve(short_entries)
                       if ENABLE_LTM and hasattr(track, "ltm") else [])
        all_entries = short_entries + ltm_entries

        mem_feats = torch.cat(
            [m["feat"].flatten(2).permute(2, 0, 1) for m in all_entries], 0)
        mem_pos   = torch.cat(
            [m["pos"].flatten(2).permute(2, 0, 1) for m in all_entries], 0)
        attended  = model.memory_attention(
            curr=curr_feat, curr_pos=curr_pos_flat,
            memory=mem_feats, memory_pos=mem_pos)

    att_2d = attended.permute(1, 2, 0).view(B, C, H, W)
    det    = track.reanchor_det
    track.reanchor_det = None

    if det is not None:
        cx, cy = det["centroid"]
        pts_t   = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=device)
        lbs_t   = torch.tensor([[1]],        dtype=torch.int32,   device=device)
        box_arr = np.array(det["bbox"], dtype=np.float32).reshape(1, 4)
        box_t   = torch.from_numpy(box_arr).to(dtype=torch.float32, device=device).unsqueeze(0)
        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=(pts_t, lbs_t), boxes=box_t, masks=None)
        multimask = True
    else:
        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=None, boxes=None, masks=track.prev_logits)
        multimask = False

    lr, iou_p, _, _ = model.sam_mask_decoder(
        image_embeddings=att_2d,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=multimask, repeat_image=False,
        high_res_features=[fpn[0], fpn[1]])

    best_idx = int(iou_p.squeeze().reshape(-1).argmax())
    mask_lr  = lr[:, best_idx:best_idx+1]
    mask_np  = (F.interpolate(
        mask_lr.float(), size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear", align_corners=False).squeeze() > 0.0).cpu().numpy()

    cur_iou = float(iou_p.squeeze().reshape(-1)[best_idx])
    # Only store good-quality frames in memory — prevents cascade degradation
    # where bad masks pollute the bank, making subsequent predictions worse.
    # Always update when re-anchored (det is not None); gate otherwise.
    if det is not None or cur_iou >= MEM_MIN_IOU:
        pred_up = F.interpolate(mask_lr, size=(IMG_SIZE, IMG_SIZE),
                                 mode="bilinear", align_corners=False)
        mem_out = model.memory_encoder(fpn[-1], pred_up, skip_mask_sigmoid=False)
        entry = {"feat": mem_out["vision_features"], "pos": mem_out["vision_pos_enc"][0]}
        track.memory_bank.append(entry)
        if ENABLE_LTM and hasattr(track, "ltm"):
            track.ltm.add(entry)

    track.prev_logits  = mask_lr.detach()
    track.last_mask    = mask_np
    track.last_iou     = cur_iou
    track.predictions[fi] = mask_np
    track.frames_alive += 1
    if mask_np.sum() > 0:
        ys, xs = np.where(mask_np)
        track.last_valid_centroid = (float(xs.mean()), float(ys.mean()))


# ── One-direction tracking pass ───────────────────────────────────────────────

@torch.no_grad()
def run_pass(model, frame_paths, direction="fwd", id_offset=0,
             yolo_model=None, yolo_reprompt=False):
    """
    Run one tracking pass through frame_paths in order.
    Returns all_preds: {track_id+id_offset: {fi_in_frame_paths: mask}}
    fi_in_frame_paths is the LOCAL index (0..N-1), caller remaps to original.
    If yolo_model is provided: frame-0 init (forward pass) + periodic
    re-prompting of degraded tracks every YOLO_REPROMPT_INTERVAL frames
    when yolo_reprompt=True.
    """
    n_frames      = len(frame_paths)
    active_tracks = []
    all_preds     = {}
    next_id       = id_offset
    elastic       = None if DISABLE_ELASTIC else ElasticController()
    bandit        = (ElasticBandit() if (elastic and ENABLE_BANDIT
                                          and not DISABLE_ELASTIC) else None)
    prev_gray     = None          # for optical-flow innovations
    dense_flow    = None
    tc_optim      = setup_tcadapt(model) if ENABLE_TCADAPT else None
    k_detect_eff  = OF_K_DETECT if ENABLE_OF_PROMPT else K_DETECT

    for fi, fp in enumerate(frame_paths):
        img_bgr  = cv2.imread(fp)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_t    = preprocess_image(img_rgb, DEVICE)

        # ── Optical-flow update ──────────────────────────────────────────────
        if ENABLE_OF_PROMPT and prev_gray is not None:
            dense_flow = compute_dense_flow(prev_gray, img_gray)
            # Propagate alive track centroids with LK for lightweight refresh
            alive_tracks = [t for t in active_tracks if t.alive]
            if alive_tracks:
                pts_in = [t.last_valid_centroid for t in alive_tracks]
                new_pts, status = compute_lk_flow(prev_gray, img_gray, pts_in)
                for t, np2, ok in zip(alive_tracks, new_pts, status):
                    if ok:
                        nx, ny = float(np2[0]), float(np2[1])
                        if 0 <= nx < IMG_SIZE and 0 <= ny < IMG_SIZE:
                            t.last_valid_centroid = (nx, ny)
        prev_gray = img_gray

        backbone_out = model.forward_image(img_t)
        fpn      = backbone_out["backbone_fpn"]
        curr_pos = backbone_out["vision_pos_enc"][-1]
        n_mem    = 0 if DISABLE_MEMORY else (elastic.update(fpn[-1]) if elastic else N_MEM)

        # Bandit: feed quality signals, adapt thresholds (no GT needed)
        if bandit is not None and elastic is not None:
            alive_ious = [t.last_iou for t in active_tracks if t.alive]
            mean_iou   = float(np.mean(alive_ious)) if alive_ious else 0.5
            bandit.observe(mean_iou, track_churn=0.0)   # churn updated after spawning
            eco_r, boost_r, skip_r = bandit.step(elastic)
            elastic.update_ratios(eco_r, boost_r, skip_r)

        deaths_this_frame = 0
        for track in active_tracks:
            if not track.alive:
                continue
            was_matched = track.reanchor_det is not None
            update_track(model, track, fpn, curr_pos, n_mem, fi, DEVICE)
            if not was_matched:
                track.frames_without_match += 1
            all_preds.setdefault(track.track_id, {})[fi] = track.last_mask
            if track.last_mask.sum() == 0:
                track.consecutive_empty += 1
            else:
                track.consecutive_empty = 0
            if (track.frames_without_match >= KILL_FRAMES and
                    track.last_iou < LOW_IOU_KILL):
                track.alive = False
                deaths_this_frame += 1
            elif track.consecutive_empty >= EMPTY_KILL_FRAMES:
                track.alive = False
                deaths_this_frame += 1

        # Periodic YOLO re-prompting (forward pass only): rescue degraded tracks
        if (yolo_reprompt and yolo_model is not None and direction == "fwd"
                and YOLO_REPROMPT_INTERVAL > 0
                and fi > 0 and fi % YOLO_REPROMPT_INTERVAL == 0):
            yolo_dets = detect_from_yolo(yolo_model, img_bgr)
            degraded  = [t for t in active_tracks
                         if t.alive and t.last_iou < REPROMPT_IOU_THRESH]
            for track in degraded:
                tcx, tcy = track.last_valid_centroid
                # Find closest YOLO det to this track's last known centroid
                best_det, best_d2 = None, MATCH_CENTROID_DIST ** 2
                for d in yolo_dets:
                    dcx, dcy = d["centroid"]
                    d2 = (dcx - tcx) ** 2 + (dcy - tcy) ** 2
                    if d2 < best_d2:
                        best_det, best_d2 = d, d2
                if best_det is not None:
                    reprompt_track(model, track, fpn, best_det, n_mem, fi, DEVICE)
                    all_preds.setdefault(track.track_id, {})[fi] = track.last_mask

        # ── Temporal consistency adaptation ──────────────────────────────────
        if (ENABLE_TCADAPT and tc_optim is not None and fi > 0
                and fi % TCADAPT_INTERVAL == 0 and dense_flow is not None):
            adapt_step_temporal(model, active_tracks, fpn, curr_pos,
                                dense_flow, tc_optim, TCADAPT_STEPS)

        births_this_frame = 0
        if fi == 0 or fi % k_detect_eff == 0:
            if fi == 0 and yolo_model is not None and direction == "fwd":
                # Use YOLO for frame-0 init (forward pass only)
                dets = detect_from_yolo(yolo_model, img_bgr)
            else:
                dets = detect_from_fpn(model, fpn, DEVICE)
            unmatched_dets = match_dets_to_tracks(dets, active_tracks)

            for det in unmatched_dets:
                if det["iou"] < SPAWN_IOU_THRESH:
                    continue
                n_alive = sum(1 for t in active_tracks if t.alive)
                if n_alive >= MAX_ACTIVE_TRACKS:
                    break
                dcx, dcy = det["centroid"]
                too_close = any(
                    ((dcx - t.last_valid_centroid[0]) ** 2 +
                     (dcy - t.last_valid_centroid[1]) ** 2) < MIN_SPAWN_DIST ** 2
                    for t in active_tracks if t.alive)
                if too_close:
                    continue
                new_t = spawn_track(model, fpn, curr_pos, det, next_id, n_mem, fi, DEVICE)
                active_tracks.append(new_t)
                all_preds[next_id] = {fi: new_t.last_mask}
                next_id += 1
                births_this_frame += 1

        # Update bandit churn signal: (births + deaths) / alive_before_prune
        if bandit is not None:
            n_alive_pre = max(sum(1 for t in active_tracks if t.alive), 1)
            churn = (births_this_frame + deaths_this_frame) / n_alive_pre
            bandit.observe(mean_iou=0.0, track_churn=churn)   # mean_iou already sent above

        active_tracks = [t for t in active_tracks if t.alive]

        if fi % 10 == 0 or fi == n_frames - 1:
            n_alive  = len(active_tracks)
            n_spawned = next_id - id_offset
            mode_str = (f"  mode={elastic.mode}" if elastic else "")
            print(f"  [{direction}] f={fi:3d}/{n_frames}  alive={n_alive:3d}  "
                  f"spawned={n_spawned:3d}  n_mem={n_mem}{mode_str}", flush=True)

    elastic_stats = elastic.stats() if elastic else {"mean_n_mem": N_MEM, "eco_pct": 0, "normal_pct": 0, "boost_pct": 0, "skip_pct": 0}
    bandit_stats  = bandit.stats() if bandit else {}
    return all_preds, next_id - id_offset, elastic_stats, bandit_stats


# ── Evaluation functions ──────────────────────────────────────────────────────

def _score_matrix(all_preds, gt, gt_cell_list, gt_frames, track_ids):
    """Mean IoU over ALL annotated frames for every (GT-cell, track) pair.
    Frames where the track has no prediction count as 0 (penalises short tracks
    that happen to have high IoU on the few frames they're active).
    """
    scores = np.zeros((len(gt_cell_list), len(track_ids)), dtype=np.float32)
    for i, cid in enumerate(gt_cell_list):
        gt_fis = [fi for fi in gt_frames if cid in gt[fi]]
        if not gt_fis:
            continue
        n_annot = len(gt_fis)
        for j, tid in enumerate(track_ids):
            total = sum(iou_score(all_preds[tid][fi], gt[fi][cid])
                        for fi in gt_fis if fi in all_preds[tid])
            scores[i, j] = total / n_annot   # denominator = all annotated frames
    return scores


def evaluate(all_preds, gt, cell_ids):
    """Single best-track assignment per GT cell (original metric)."""
    gt_frames    = sorted(gt.keys())
    gt_cell_list = list(cell_ids)
    track_ids    = list(all_preds.keys())

    if not track_ids:
        total = sum(1 for fi in gt_frames for _ in gt[fi].values())
        return {"seg": 0.0, "mean_iou": 0.0, "n": total}

    scores      = _score_matrix(all_preds, gt, gt_cell_list, gt_frames, track_ids)
    all_ious    = []
    used_tracks = set()
    for i, cid in enumerate(gt_cell_list):
        row    = scores[i].copy()
        row[[j for j, tid in enumerate(track_ids) if tid in used_tracks]] = -1
        best_j = int(np.argmax(row))
        gt_fis = [fi for fi in gt_frames if cid in gt[fi]]
        if row[best_j] <= 0:
            all_ious.extend([0.0] * len(gt_fis))
        else:
            tid = track_ids[best_j]
            used_tracks.add(tid)
            for fi in gt_fis:
                if fi in all_preds[tid]:
                    all_ious.append(iou_score(all_preds[tid][fi], gt[fi][cid]))
                else:
                    all_ious.append(0.0)

    seg      = float(np.median(all_ious)) if all_ious else 0.0
    mean_iou = float(np.mean(all_ious))   if all_ious else 0.0
    return {"seg": seg, "mean_iou": mean_iou, "n": int(len(all_ious))}


def evaluate_multi(all_preds, gt, cell_ids, n_fwd, k_tracks=None):
    """
    Multi-track assignment: K rounds of Hungarian matching over the combined
    fwd+bwd track pool. Each GT cell receives up to K tracks (from either
    direction); per-frame IoU = max over all assigned tracks.

    k_tracks defaults to floor(n_tracks / n_cells), minimum 2. This gives
    each cell roughly the same number of tracks as resources allow while
    ensuring globally-optimal assignment in each round.
    """
    gt_frames    = sorted(gt.keys())
    gt_cell_list = list(cell_ids)
    all_tids     = list(all_preds.keys())
    n_cells      = len(gt_cell_list)

    if not all_tids or not n_cells:
        total = sum(1 for fi in gt_frames for _ in gt[fi].values())
        return {"seg": 0.0, "mean_iou": 0.0, "n": total, "cell_details": [],
                "k_tracks": 0}

    if k_tracks is None:
        k_tracks = max(2, len(all_tids) // n_cells)

    # Full score matrix: shape (n_cells, n_tracks)
    scores = _score_matrix(all_preds, gt, gt_cell_list, gt_frames, all_tids)

    # K rounds of Hungarian assignment over the combined pool
    cell_tracks   = {cid: [] for cid in gt_cell_list}
    remaining_idx = list(range(len(all_tids)))  # indices into all_tids

    for _ in range(k_tracks):
        if not remaining_idx:
            break
        sub = scores[:, remaining_idx]          # (n_cells, n_remaining)
        n_rem = len(remaining_idx)
        if n_rem < n_cells:
            # Fewer tracks than cells — pad cost matrix so Hungarian can run
            pad  = np.zeros((n_cells, n_cells - n_rem), dtype=np.float32)
            sub  = np.hstack([sub, pad])
        row_ind, col_ind = linear_sum_assignment(-sub)
        used = set()
        for r, c in zip(row_ind, col_ind):
            if c >= n_rem:          # padding column
                continue
            real = remaining_idx[c]
            if scores[r, real] > 0:
                cell_tracks[gt_cell_list[r]].append(all_tids[real])
            used.add(c)
        remaining_idx = [remaining_idx[c] for c in range(n_rem) if c not in used]

    # Score each cell: per annotated frame, take max IoU over assigned tracks
    all_ious     = []
    cell_details = []
    for cid in gt_cell_list:
        gt_fis = [fi for fi in gt_frames if cid in gt[fi]]
        tids   = cell_tracks[cid]
        if not tids:
            all_ious.extend([0.0] * len(gt_fis))
            cell_details.append({"cid": cid, "n_gt": len(gt_fis),
                                  "n_covered": 0, "mean_iou": 0.0})
            continue
        cell_ious = []
        n_covered = 0
        for fi in gt_fis:
            best = max(
                (iou_score(all_preds[tid][fi], gt[fi][cid])
                 for tid in tids if fi in all_preds[tid]),
                default=0.0)
            cell_ious.append(best)
            n_covered += best > 0
        all_ious.extend(cell_ious)
        cell_details.append({"cid": cid, "n_gt": len(gt_fis),
                              "n_covered": n_covered,
                              "mean_iou": float(np.mean(cell_ious))})

    seg      = float(np.median(all_ious)) if all_ious else 0.0
    mean_iou = float(np.mean(all_ious))   if all_ious else 0.0
    n_fwd_used = sum(1 for ts in cell_tracks.values() for t in ts if t < n_fwd)
    n_bwd_used = sum(1 for ts in cell_tracks.values() for t in ts if t >= n_fwd)
    return {"seg": seg, "mean_iou": mean_iou, "n": len(all_ious),
            "cell_details": cell_details, "k_tracks": k_tracks,
            "n_fwd_used": n_fwd_used, "n_bwd_used": n_bwd_used}


def evaluate_oracle(all_preds, gt, cell_ids):
    """
    Per-frame oracle: for each (GT cell, annotated frame), take max IoU over
    ALL tracks active at that frame.  Upper bound — shows detection quality
    independent of tracking identity consistency.
    """
    gt_frames = sorted(gt.keys())
    all_ious  = []
    for fi in gt_frames:
        for cid, gt_mask in sorted(gt[fi].items()):
            if cid not in cell_ids:
                continue
            best_iou = max(
                (iou_score(preds[fi], gt_mask)
                 for preds in all_preds.values() if fi in preds),
                default=0.0)
            all_ious.append(best_iou)

    seg      = float(np.median(all_ious)) if all_ious else 0.0
    mean_iou = float(np.mean(all_ious))   if all_ious else 0.0
    return {"seg": seg, "mean_iou": mean_iou, "n": len(all_ious)}


# ── Streaming bidirectional helper ───────────────────────────────────────────

# Max tracks per backward-decode batch. Larger = faster but needs more GPU memory.
# 16 avoids OOM on Jetson NX when all 60 tracks have full T=7 banks (GOWT1 case).
BIDIR_CHUNK = int(os.environ.get("BIDIR_CHUNK", "16"))


@torch.no_grad()
def _backward_decode_frame(model, old_entry, alive_tracks_dict):
    """
    Re-decode an old (buffered) frame using CURRENT memory banks.
    The current memory banks contain K frames of future context the forward
    pass had not yet seen when the old frame was processed.

    Returns {tid: fused_mask} where fused = forward_mask | backward_mask.
    Tracks that died or have empty banks fall back to the forward mask.

    Processes tracks in chunks of BIDIR_CHUNK to avoid GPU OOM when all tracks
    have full memory banks (worst case: 60 tracks × T=7 on Jetson NX).
    """
    fpn_old      = old_entry["fpn"]
    curr_pos_old = old_entry["curr_pos"]
    image_pe     = model.sam_prompt_encoder.get_dense_pe()
    B, C, H, W   = fpn_old[-1].shape

    decode_tids = [
        tid for tid in old_entry["fwd_logits"]
        if tid in alive_tracks_dict and len(alive_tracks_dict[tid].memory_bank) > 0
    ]

    fused = dict(old_entry["fwd_masks"])   # fallback = forward mask
    if not decode_tids:
        return fused

    curr_feat     = fpn_old[-1].flatten(2).permute(2, 0, 1)  # (HW, 1, C)
    curr_pos_flat = curr_pos_old.flatten(2).permute(2, 0, 1)

    # Process in chunks to cap GPU memory usage
    for chunk_start in range(0, len(decode_tids), BIDIR_CHUNK):
        chunk = decode_tids[chunk_start : chunk_start + BIDIR_CHUNK]

        # Group within chunk by bank length T for exact batching (no padding)
        from collections import defaultdict
        groups = defaultdict(list)
        for tid in chunk:
            groups[len(alive_tracks_dict[tid].memory_bank)].append(tid)

        att_2ds = {}
        for T, tids_g in groups.items():
            Bg         = len(tids_g)
            curr_b     = curr_feat.expand(-1, Bg, -1)
            curr_pos_b = curr_pos_flat.expand(-1, Bg, -1)

            per_t_feats, per_t_pos = [], []
            for ti in range(T):
                per_t_feats.append(torch.cat(
                    [alive_tracks_dict[tid].memory_bank[ti]["feat"].flatten(2).permute(2, 0, 1)
                     for tid in tids_g], dim=1))
                per_t_pos.append(torch.cat(
                    [alive_tracks_dict[tid].memory_bank[ti]["pos"].flatten(2).permute(2, 0, 1)
                     for tid in tids_g], dim=1))

            attended = model.memory_attention(
                curr=curr_b, curr_pos=curr_pos_b,
                memory=torch.cat(per_t_feats, dim=0),
                memory_pos=torch.cat(per_t_pos, dim=0))  # (HW, Bg, C)

            for bi, tid in enumerate(tids_g):
                att_2ds[tid] = attended[:, bi:bi+1, :].permute(1, 2, 0).view(B, C, H, W)

        # Batched mask decoder for this chunk
        Nc        = len(chunk)
        att_b     = torch.cat([att_2ds[tid] for tid in chunk], dim=0)
        prev_b    = torch.cat([old_entry["fwd_logits"][tid] for tid in chunk], dim=0)

        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=None, boxes=None, masks=prev_b)
        lr, _, _, _ = model.sam_mask_decoder(
            image_embeddings=att_b,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False, repeat_image=False,
            high_res_features=[
                fpn_old[0].expand(Nc, -1, -1, -1).contiguous(),
                fpn_old[1].expand(Nc, -1, -1, -1).contiguous(),
            ])

        masks_up = F.interpolate(lr.float(), size=(IMG_SIZE, IMG_SIZE),
                                  mode="bilinear", align_corners=False)
        masks_np = (masks_up.squeeze(1) > 0.0).cpu().numpy()

        for i, tid in enumerate(chunk):
            fused[tid] = (old_entry["fwd_masks"].get(
                tid, np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)) | masks_np[i])

    return fused


@torch.no_grad()
def _spawn_backward_tracks(model, old_entry, buf_list, active_tracks,
                            all_preds, start_id, n_mem):
    """
    Find cells at old_entry["fi"] that the forward pass missed, spawn them,
    and propagate forward through buf_list so they rejoin the live tracker.

    Detection runs on old_entry["fpn"] (the buffered frame's FPN features).
    Matching uses existing predictions at old_fi — any detection with <NMS_IOU_THRESH
    overlap to all existing masks is a new cell.

    Each new track is:
      1. Spawned at old_fi  → prediction stored in all_preds
      2. Propagated forward through buf_list frames  → predictions stored
      3. Added to active_tracks  → continues tracking from current frame

    Returns (new_track_list, n_spawned).
    """
    fpn_old = old_entry["fpn"]
    old_fi  = old_entry["fi"]

    dets = detect_from_fpn(model, fpn_old, DEVICE)
    if not dets:
        return [], 0

    # Existing masks at old_fi (forward + already-fused backward decodes)
    existing_masks = [
        preds[old_fi]
        for preds in all_preds.values()
        if old_fi in preds
    ]

    new_tracks = []
    new_tid = start_id

    for det in dets:
        if det["iou"] < SPAWN_IOU_THRESH:
            continue
        budget = MAX_ACTIVE_TRACKS - len(active_tracks) - len(new_tracks)
        if budget <= 0:
            break

        det_mask = det["mask"]

        # Skip if detection overlaps an already-tracked cell at old_fi
        if any((det_mask & em).sum() / max((det_mask | em).sum(), 1) > NMS_IOU_THRESH
               for em in existing_masks):
            continue

        # Skip if spawn point is too close to a current live track centroid
        dcx, dcy = det["centroid"]
        if any(((dcx - t.last_valid_centroid[0]) ** 2 +
                 (dcy - t.last_valid_centroid[1]) ** 2) < MIN_SPAWN_DIST ** 2
               for t in active_tracks if t.alive):
            continue

        # Spawn at old frame
        track = spawn_track(model, fpn_old, old_entry["curr_pos"],
                            det, new_tid, n_mem, old_fi, DEVICE)
        all_preds[new_tid] = {old_fi: track.last_mask.copy()}

        # Propagate forward through buffered frames to catch up to current time
        for buf_entry in buf_list:
            update_track(model, track, buf_entry["fpn"], buf_entry["curr_pos"],
                         n_mem, buf_entry["fi"], DEVICE)
            all_preds[new_tid][buf_entry["fi"]] = track.last_mask.copy()

        # Guard against duplicate: add mask to existing set
        existing_masks.append(track.last_mask)
        new_tracks.append(track)
        new_tid += 1

    return new_tracks, new_tid - start_id


@torch.no_grad()
def run_streaming_bidir_pass(model, frame_paths, K,
                              yolo_model=None, yolo_reprompt=False):
    """
    Online bidirectional tracking with K-frame output delay.

    For each emitted frame t-K:
      - Forward mask  : recorded during the online forward pass at time t-K
      - Backward mask : re-decode frame t-K using memory built up to frame t
                        (K frames of future context not available at t-K)
      - Fused mask    : forward | backward

    Latency vs forward-only: K additional frames (K/FPS seconds).
    Accuracy vs offline bidir: approaches offline as K → T; saturates ~K=10.
    """
    n_frames      = len(frame_paths)
    active_tracks = []
    all_preds     = {}   # {tid: {fi: mask}} — forward masks stored first, overwritten by fused
    next_id       = 0
    elastic       = None if DISABLE_ELASTIC else ElasticController()
    buf           = collections.deque()   # sliding window of K+1 entries

    for fi, fp in enumerate(frame_paths):
        img_bgr = cv2.imread(fp)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_t   = preprocess_image(img_rgb, DEVICE)

        backbone_out = model.forward_image(img_t)
        fpn      = backbone_out["backbone_fpn"]
        curr_pos = backbone_out["vision_pos_enc"][-1]
        n_mem    = 0 if DISABLE_MEMORY else (elastic.update(fpn[-1]) if elastic else N_MEM)

        # ── Forward step (identical to run_pass) ─────────────────────────────
        for track in active_tracks:
            if not track.alive:
                continue
            was_matched = track.reanchor_det is not None
            update_track(model, track, fpn, curr_pos, n_mem, fi, DEVICE)
            if not was_matched:
                track.frames_without_match += 1
            if track.last_mask.sum() == 0:
                track.consecutive_empty += 1
            else:
                track.consecutive_empty = 0
            if track.frames_without_match >= KILL_FRAMES and track.last_iou < LOW_IOU_KILL:
                track.alive = False
            elif track.consecutive_empty >= EMPTY_KILL_FRAMES:
                track.alive = False

        if (yolo_reprompt and yolo_model is not None
                and YOLO_REPROMPT_INTERVAL > 0
                and fi > 0 and fi % YOLO_REPROMPT_INTERVAL == 0):
            yolo_dets = detect_from_yolo(yolo_model, img_bgr)
            degraded  = [t for t in active_tracks
                         if t.alive and t.last_iou < REPROMPT_IOU_THRESH]
            for track in degraded:
                tcx, tcy = track.last_valid_centroid
                best_det, best_d2 = None, MATCH_CENTROID_DIST ** 2
                for d in yolo_dets:
                    dcx, dcy = d["centroid"]
                    d2 = (dcx - tcx) ** 2 + (dcy - tcy) ** 2
                    if d2 < best_d2:
                        best_det, best_d2 = d, d2
                if best_det is not None:
                    reprompt_track(model, track, fpn, best_det, n_mem, fi, DEVICE)

        if fi == 0 or fi % K_DETECT == 0:
            dets = (detect_from_yolo(yolo_model, img_bgr)
                    if fi == 0 and yolo_model is not None
                    else detect_from_fpn(model, fpn, DEVICE))
            unmatched = match_dets_to_tracks(dets, active_tracks)
            for det in unmatched:
                if det["iou"] < SPAWN_IOU_THRESH:
                    continue
                if sum(1 for t in active_tracks if t.alive) >= MAX_ACTIVE_TRACKS:
                    break
                dcx, dcy = det["centroid"]
                if any(((dcx - t.last_valid_centroid[0]) ** 2 +
                         (dcy - t.last_valid_centroid[1]) ** 2) < MIN_SPAWN_DIST ** 2
                        for t in active_tracks if t.alive):
                    continue
                new_t = spawn_track(model, fpn, curr_pos, det, next_id, n_mem, fi, DEVICE)
                active_tracks.append(new_t)
                next_id += 1

        active_tracks = [t for t in active_tracks if t.alive]

        # Store forward masks (may be overwritten by fused masks K frames later)
        fwd_masks  = {t.track_id: t.last_mask.copy()      for t in active_tracks}
        fwd_logits = {t.track_id: t.prev_logits.clone()   for t in active_tracks}
        for tid, mask in fwd_masks.items():
            all_preds.setdefault(tid, {})[fi] = mask

        buf.append({
            "fi":         fi,
            "fpn":        [f.clone() for f in fpn],
            "curr_pos":   curr_pos.clone(),
            "fwd_logits": fwd_logits,
            "fwd_masks":  fwd_masks,
        })

        # Emit oldest buffered frame once buffer has K+1 entries
        if len(buf) > K:
            old       = buf.popleft()
            alive_now = {t.track_id: t for t in active_tracks}

            # Step 1: refine existing forward tracks with K-frame future context
            fused = _backward_decode_frame(model, old, alive_now)
            for tid, mask in fused.items():
                all_preds.setdefault(tid, {})[old["fi"]] = mask

            # Step 2: spawn new tracks for cells missed by the forward pass,
            # propagate them forward through the K buffered frames so they
            # join active_tracks at the current time step
            new_tracks, n_bwd = _spawn_backward_tracks(
                model, old, list(buf), active_tracks, all_preds, next_id, n_mem)
            active_tracks.extend(new_tracks)
            next_id += n_bwd

        if fi % 10 == 0 or fi == n_frames - 1:
            print(f"  [stream K={K}] f={fi:3d}/{n_frames}  alive={len(active_tracks):3d}  "
                  f"spawned={next_id:3d}  n_mem={n_mem}", flush=True)

    # Flush remaining K frames with the final (fully informed) memory state
    # Also attempt backward spawning — late-appearing cells may be visible now
    alive_now = {t.track_id: t for t in active_tracks}
    buf_list  = list(buf)
    while buf:
        old   = buf.popleft()
        fused = _backward_decode_frame(model, old, alive_now)
        for tid, mask in fused.items():
            all_preds.setdefault(tid, {})[old["fi"]] = mask
        new_tracks, n_bwd = _spawn_backward_tracks(
            model, old, [e for e in buf_list if e["fi"] > old["fi"]],
            active_tracks, all_preds, next_id, n_mem)
        active_tracks.extend(new_tracks)
        next_id += n_bwd
        alive_now = {t.track_id: t for t in active_tracks}

    elastic_stats = elastic.stats() if elastic else {"mean_n_mem": N_MEM, "pct": {}}
    return all_preds, next_id, elastic_stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    model = build_sam2_finetuned(DEVICE)
    model.eval()
    all_results = {}

    # Optional YOLO model for frame-0 cell initialisation
    _yolo = None
    if _YOLO_CKPT:
        from ultralytics import YOLO as _YOLOCLS
        _yolo = _YOLOCLS(_YOLO_CKPT)
        print(f"[YOLO] Loaded frame-0 detector: {_YOLO_CKPT}")

    for seq_cfg in SEQUENCES:
        name    = seq_cfg["name"]
        ctc_dir = seq_cfg["ctc_dir"]
        seq     = seq_cfg["seq"]

        print(f"\n{'='*60}")
        print(f"  {name}  [BIDIRECTIONAL dense-grid tracking]")
        print(f"{'='*60}")

        seq_dir    = os.path.join(ctc_dir, seq)
        gt_seg_dir = os.path.join(ctc_dir, f"{seq}_GT", "SEG")
        jpg_dir    = str(OUT_DIR / name.replace("/", "_") / "frames")

        _reset_defaults()
        if seq_cfg.get("overrides"):
            _apply_overrides(seq_cfg["overrides"])

        t0          = time.perf_counter()
        frame_paths = load_sequence(seq_dir, jpg_dir)
        gt          = load_gt(gt_seg_dir)
        cell_ids    = sorted({cid for cells in gt.values() for cid in cells})
        n_frames    = len(frame_paths)

        print(f"  {n_frames} frames  {len(cell_ids)} GT cells  "
              f"{sum(len(v) for v in gt.values())} annotations")

        # Per-sequence YOLO usage flags
        seq_yolo      = _yolo if (_yolo is not None and
                                   (seq_cfg.get("yolo_init") or
                                    seq_cfg.get("yolo_reprompt"))) else None
        seq_reprompt  = bool(seq_cfg.get("yolo_reprompt") and seq_yolo is not None)

        # ── Forward pass (+ optional bidir) ──────────────────────────────────
        if STREAMING_K > 0:
            # Streaming bidir: single forward pass with K-frame delayed output
            print(f"\n  --- Streaming bidir (K={STREAMING_K}) ---")
            combined, n_fwd, fwd_elastic = run_streaming_bidir_pass(
                model, frame_paths, STREAMING_K,
                yolo_model=seq_yolo, yolo_reprompt=seq_reprompt)
            n_bwd       = 0
            bwd_elastic = {"mean_n_mem": N_MEM, "pct": {}}
            fwd_bandit  = {}
            bwd_bandit  = {}
        else:
            print(f"\n  --- Forward pass ---")
            fwd_raw, n_fwd, fwd_elastic, fwd_bandit = run_pass(
                model, frame_paths, direction="fwd",
                id_offset=0, yolo_model=seq_yolo, yolo_reprompt=seq_reprompt)

            # ── Backward pass (skip if FORWARD_ONLY=1) ────────────────────
            if FORWARD_ONLY:
                bwd_raw, n_bwd, bwd_elastic, bwd_bandit = {}, 0, {"mean_n_mem": N_MEM, "pct": {}}, {}
                combined = dict(fwd_raw)
            else:
                print(f"\n  --- Backward pass ---")
                bwd_raw, n_bwd, bwd_elastic, bwd_bandit = run_pass(
                    model, frame_paths[::-1], direction="bwd",
                    id_offset=n_fwd, yolo_model=None)
                bwd_preds = {}
                for tid, preds in bwd_raw.items():
                    bwd_preds[tid] = {n_frames - 1 - fi: mask for fi, mask in preds.items()}
                combined = {**fwd_raw, **bwd_preds}

        m_single = evaluate(combined, gt, cell_ids)
        m_multi  = evaluate_multi(combined, gt, cell_ids, n_fwd)
        m_oracle = evaluate_oracle(combined, gt, cell_ids)
        elapsed  = time.perf_counter() - t0
        K        = m_multi["k_tracks"]

        print(f"\n  --- Results ---")
        print(f"  single-track:    SEG={m_single['seg']:.4f}  mean_iou={m_single['mean_iou']:.4f}")
        print(f"  multi-track(K={K}): SEG={m_multi['seg']:.4f}  mean_iou={m_multi['mean_iou']:.4f}"
              f"  (fwd={m_multi['n_fwd_used']} bwd={m_multi['n_bwd_used']})")
        print(f"  oracle:          SEG={m_oracle['seg']:.4f}  mean_iou={m_oracle['mean_iou']:.4f}")
        print(f"  fwd_tracks={n_fwd}  bwd_tracks={n_bwd}  combined={len(combined)}  "
              f"GT_cells={len(cell_ids)}")
        print(f"  Elapsed: {elapsed:.0f}s")
        mean_n_mem = (fwd_elastic.get("mean_n_mem", N_MEM) + bwd_elastic.get("mean_n_mem", N_MEM)) / 2
        fwd_pct    = fwd_elastic.get("pct", {})
        print(f"  Elastic: mean_N_MEM={mean_n_mem:.2f}  "
              f"(fwd skip={fwd_pct.get('SKIP',0):.0f}%  "
              f"eco={fwd_pct.get('ECO',0):.0f}%  "
              f"normal={fwd_pct.get('NORMAL',0):.0f}%  "
              f"boost={fwd_pct.get('BOOST',0):.0f}%)")
        if fwd_bandit:
            print(f"  Bandit:  eco_ratio={fwd_bandit.get('eco_ratio','?'):.3f}  "
                  f"boost_ratio={fwd_bandit.get('boost_ratio','?'):.3f}  "
                  f"iou_ema={fwd_bandit.get('iou_ema','?'):.3f}  "
                  f"steps={fwd_bandit.get('steps','?')}")
        print(f"\n  Per-cell multi-track coverage (K={K}):")
        for cd in m_multi.get("cell_details", []):
            print(f"    cell {cd['cid']:3d}: {cd['n_covered']:2d}/{cd['n_gt']:2d} frames  "
                  f"mean_iou={cd['mean_iou']:.3f}")

        all_results[name] = {
            "seg_single":    m_single["seg"],
            "seg_multi":     m_multi["seg"],
            "seg_oracle":    m_oracle["seg"],
            "mean_iou_multi": m_multi["mean_iou"],
            "k_tracks":      K,
            "n":             m_multi["n"],
            "n_fwd_tracks":  int(n_fwd),
            "n_bwd_tracks":  int(n_bwd),
            "n_gt_cells":    int(len(cell_ids)),
            "elapsed_s":     round(elapsed, 1),
            "cell_details":  m_multi.get("cell_details", []),
            "elastic_fwd":   fwd_elastic,
            "elastic_bwd":   bwd_elastic,
            "mean_n_mem":    round(mean_n_mem, 2),
            "bandit_fwd":    fwd_bandit,
            "bandit_bwd":    bwd_bandit,
        }

    print(f"\n{'='*76}")
    print(f"  {'Sequence':<28}  {'SEG_single':>10}  {'SEG_multi':>9}  {'K':>3}  {'SEG_oracle':>10}")
    print(f"  {'-'*74}")
    for name, r in all_results.items():
        print(f"  {name:<28}  {r['seg_single']:>10.4f}  {r['seg_multi']:>9.4f}  "
              f"{r['k_tracks']:>3}  {r['seg_oracle']:>10.4f}")
    print(f"{'='*76}")

    out_json = OUT_DIR / "autotrack_bidir.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved → {out_json}")


if __name__ == "__main__":
    main()
