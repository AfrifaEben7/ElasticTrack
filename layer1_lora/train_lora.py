#!/usr/bin/env python3
"""
train_lora.py — LoRA fine-tuning of SAM2 (sam2_hiera_tiny) for macrophage segmentation.

Key difference vs train.py:
  - ALL base weights are frozen (image encoder, prompt encoder, mask decoder).
  - LoRA adapters (rank-r decomposition) are injected into every attention
    projection in the mask decoder transformer (q/k/v/out_proj in all 3 blocks).
  - Only the LoRA parameters are trained (~57 K vs ~7 M in the full decoder).
  - Result: macrophage specialisation WITHOUT catastrophic forgetting of the
    base model's general segmentation capability.

LoRA formulation:
    W_out = W_base @ x + (alpha/r) * B @ A @ x
    where A ∈ R^{r × in}, B ∈ R^{out × r}, alpha/r = 1 (no extra scaling).
    A initialised with N(0, 0.02); B initialised to zero so adapter output
    starts at zero and training is stable.

Usage (on a GPU machine):
    python3 train_lora.py \
        --data_root /home/cps/Documents/biomed/ViT_part/microphage-4 \
        --checkpoint /home/cps/Documents/biomed/ViT_part/checkpoints/sam2_hiera_tiny.pt \
        --epochs 30 \
        --rank 4 \
        --output /home/cps/Documents/biomed/ViT_part/weights/tiny/sam2_macrophage_lora.pt
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from monai.losses import DiceLoss, FocalLoss
from scipy import ndimage

# Reuse dataset + prompt helpers from train.py in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from train import (
    MacrophageDataset,
    load_model,
    forward_pass,
    count_params,
)
from ctc_dataset import CTCDataset


# ===========================================================================
# LoRA
# ===========================================================================

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank adapter.

    W_out = W_base(x) + (alpha/r) * B(A(x))
    A: (r, in_features)   — initialised N(0, 0.02)
    B: (out_features, r)  — initialised 0  → zero output at init
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.base  = base
        self.rank  = rank
        self.scale = alpha / rank

        in_f, out_f = base.in_features, base.out_features
        dev  = base.weight.device
        dtyp = base.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(rank, in_f, device=dev, dtype=dtyp))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=dtyp))
        nn.init.normal_(self.lora_A, std=0.02)

        # Base weight and bias are frozen — no grad
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.lora_A.T @ self.lora_B.T)

    def merged_weight(self) -> torch.Tensor:
        """Return base weight + LoRA delta for exporting a merged checkpoint."""
        return self.base.weight + self.scale * (self.lora_B @ self.lora_A)


def _apply_lora_to_module(root_module: nn.Module, module_path: str,
                          rank: int, alpha: float,
                          target_suffix: tuple) -> list[str]:
    """
    Inject LoRA into all nn.Linear layers inside `root_module` whose name
    ends with one of `target_suffix`.  Returns list of patched paths.
    """
    patched = []
    replacements = {}
    for name, mod in root_module.named_modules():
        if isinstance(mod, nn.Linear):
            if any(name.endswith(s) for s in target_suffix):
                replacements[name] = LoRALinear(mod, rank=rank, alpha=alpha)

    for name, lora_mod in replacements.items():
        parts = name.split(".")
        parent = root_module
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], lora_mod)
        patched.append(f"{module_path}.{name}")

    return patched


def apply_lora(model: nn.Module, rank: int, alpha: float = 1.0,
               encoder_lora: bool = False) -> list[str]:
    """
    Inject LoRA adapters into SAM2.

    Always patches mask decoder (q/k/v/out_proj in all transformer layers).
    Optionally patches image encoder backbone (qkv and proj in all Hiera blocks).

    Returns a list of all patched module paths for logging.
    """
    patched = []

    # 1. Mask decoder — separate q/k/v/out_proj projections
    patched += _apply_lora_to_module(
        model.sam_mask_decoder, "sam_mask_decoder", rank, alpha,
        target_suffix=("q_proj", "k_proj", "v_proj", "out_proj"),
    )

    # 2. Image encoder backbone — fused qkv and attention output proj
    if encoder_lora:
        patched += _apply_lora_to_module(
            model.image_encoder.trunk, "image_encoder.trunk", rank, alpha,
            target_suffix=("attn.qkv", "attn.proj"),
        )

    return patched


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA parameters (lora_A, lora_B) — small checkpoint."""
    return {k: v for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


def _merge_lora_in_module(root_module: nn.Module) -> None:
    """Replace every LoRALinear inside root_module with a merged nn.Linear."""
    for name, mod in list(root_module.named_modules()):
        if isinstance(mod, LoRALinear):
            merged = nn.Linear(mod.base.in_features, mod.base.out_features,
                               bias=mod.base.bias is not None)
            merged.weight.data.copy_(mod.merged_weight())
            if mod.base.bias is not None:
                merged.bias.data.copy_(mod.base.bias.data)
            parts = name.split(".")
            parent = root_module
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], merged)


def merge_lora_into_base(model: nn.Module) -> nn.Module:
    """
    Replace every LoRALinear with a plain nn.Linear that has the merged weight.
    Handles both mask decoder and image encoder. Returns model modified in-place.
    """
    _merge_lora_in_module(model.sam_mask_decoder)
    _merge_lora_in_module(model.image_encoder)
    return model


# ===========================================================================
# Argument parser
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="LoRA fine-tune SAM2 hiera_tiny — no catastrophic forgetting.")
    p.add_argument("--data_root",   default="",
                   help="Macrophage dataset root (COCO-RLE format). Ignored if --ctc-dirs is set.")
    p.add_argument("--ctc-dirs",    nargs="+", default=None,
                   help="One or more CTC dataset roots. If set, trains on CTC data instead of macrophage.")
    p.add_argument("--ctc-train-seqs", nargs="+", default=None,
                   help="Which sequence IDs to use for training (e.g. 02). Default: all. "
                        "Use '02' to hold out seq01 for fair evaluation.")
    p.add_argument("--checkpoint",
                   default="/home/cps/Documents/biomed/ViT_part/checkpoints/sam2_hiera_tiny.pt")
    p.add_argument("--model_cfg",   default="sam2_hiera_t_512.yaml")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=3e-4,
                   help="LoRA adapters can use a higher lr than full fine-tuning")
    p.add_argument("--lr_min",      type=float, default=1e-5)
    p.add_argument("--warmup_steps",type=int,   default=50)
    p.add_argument("--rank",        type=int,   default=4,
                   help="LoRA rank r (4–16 typical; higher = more capacity)")
    p.add_argument("--alpha",       type=float, default=4.0,
                   help="LoRA alpha scaling (set equal to rank → scale=1.0)")
    p.add_argument("--l2_reg",      type=float, default=0.0,
                   help="Extra L2 penalty on LoRA params (0 = off, try 1e-4)")
    p.add_argument("--n_points",    type=int,   default=3)
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--image_size",  type=int,   default=512)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--output",
                   default="/home/cps/Documents/biomed/ViT_part/weights/tiny/sam2_macrophage_lora.pt",
                   help="Path to save the merged full-model weights for inference")
    p.add_argument("--lora_only_out",
                   default="/home/cps/Documents/biomed/ViT_part/weights/tiny/sam2_macrophage_lora_adapters.pt",
                   help="Path to save the LoRA adapters only (small, for research)")
    p.add_argument("--encoder-lora", action="store_true", default=False,
                   help="Also inject LoRA into image encoder backbone (attn.qkv and attn.proj). "
                        "Adds ~110K params at rank=4. Improves cell-type generalisation.")
    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for SAM2 LoRA fine-tuning.")
    device = "cuda"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[INFO] Device: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Training dtype: {dtype}")

    # ---- Dataset -----------------------------------------------------------
    if args.ctc_dirs:
        train_seqs = args.ctc_train_seqs  # e.g. ["02"] or None for all
        print(f"\n[INFO] Loading CTC datasets from: {args.ctc_dirs}")
        if train_seqs:
            print(f"[INFO] Training sequences: {train_seqs}  (val uses all GT)")
        train_dataset = CTCDataset(args.ctc_dirs, split="train",
                                   image_size=args.image_size, augment=True,
                                   sequences=train_seqs)
        val_dataset   = CTCDataset(args.ctc_dirs, split="val",
                                   image_size=args.image_size, augment=False)
    else:
        data_root = Path(args.data_root).resolve()
        print(f"\n[INFO] Loading macrophage dataset from: {data_root}")
        train_dataset = MacrophageDataset(str(data_root), split="train",
                                          image_size=args.image_size,
                                          n_points=args.n_points, augment=True)
        val_dataset   = MacrophageDataset(str(data_root), split="valid",
                                          image_size=args.image_size,
                                          n_points=1, augment=False)

    def collate_fn(batch):
        images, masks, coords, labels, boxes = zip(*batch)
        images = torch.stack(images)
        masks  = torch.stack(masks)
        max_n  = max(c.shape[2] for c in coords)
        def pad_pts(t):
            n = t.shape[2]
            if n < max_n:
                t = torch.cat([t, torch.zeros(1, 1, max_n-n, 2)], dim=2)
            return t
        def pad_lbs(t):
            n = t.shape[2]
            if n < max_n:
                t = torch.cat([t, torch.full((1,1,max_n-n), -1, dtype=torch.int32)], dim=2)
            return t
        coords = torch.cat([pad_pts(c) for c in coords], dim=0)
        labels = torch.cat([pad_lbs(l) for l in labels], dim=0)
        boxes_out = None if any(b is None for b in boxes) else torch.cat(boxes, dim=0)
        return images, masks, coords, labels, boxes_out

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=False, collate_fn=collate_fn)

    # ---- Model + LoRA injection --------------------------------------------
    print(f"\n[INFO] Loading SAM2 base model ...")
    sam2_model = load_model(args.model_cfg, str(Path(args.checkpoint).resolve()),
                            device, dtype)

    # Freeze everything first
    for p in sam2_model.parameters():
        p.requires_grad_(False)

    # Inject LoRA into mask decoder (always) and image encoder (if --encoder-lora)
    patched = apply_lora(sam2_model, rank=args.rank, alpha=args.alpha,
                         encoder_lora=args.encoder_lora)
    enc_str = " and image encoder" if args.encoder_lora else ""
    print(f"\n[INFO] LoRA rank={args.rank} alpha={args.alpha} — "
          f"injected into {len(patched)} layers (mask decoder{enc_str}):")
    for p in patched:
        print(f"       {p}")

    total, trainable = count_params(sam2_model)
    print(f"\n[INFO] Total params     : {total:,}")
    print(f"[INFO] Trainable params : {trainable:,}  ({100.0*trainable/total:.3f}%)")

    sam2_model.train()
    sam2_model.image_encoder.eval()        # keep BN/dropout in eval mode

    # ---- Optimizer (only LoRA params) -------------------------------------
    lora_params = [p for p in sam2_model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(lora_params, lr=args.lr_min, weight_decay=0.0)

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return (args.lr_min + (args.lr - args.lr_min) * step / warmup_steps) / args.lr
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos_val  = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (args.lr_min + (args.lr - args.lr_min) * cos_val) / args.lr

    scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0

    dice_loss_fn  = DiceLoss(sigmoid=True, reduction="mean")
    focal_loss_fn = FocalLoss(reduction="mean")

    # ---- Training loop -----------------------------------------------------
    print(f"\n[INFO] Training {args.epochs} epochs, "
          f"lr={args.lr_min:.0e}→{args.lr:.0e}→{args.lr_min:.0e}, "
          f"rank={args.rank}, batch={args.batch_size}\n")

    best_val_loss = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(args.lora_only_out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        sam2_model.train()
        sam2_model.image_encoder.eval()

        epoch_loss = 0.0
        t0 = time.time()

        for bi, (images, gt_masks, point_coords, point_labels, boxes) in enumerate(train_loader):
            images       = images.to(device)
            gt_masks     = gt_masks.to(device, dtype=torch.float32)
            point_coords = point_coords.to(device, dtype=torch.float32)
            point_labels = point_labels.to(device, dtype=torch.int32)
            boxes        = boxes.to(device, dtype=torch.float32) if boxes is not None else None

            loss, _ = forward_pass(sam2_model, images, gt_masks, point_coords,
                                   point_labels, dtype, device,
                                   dice_loss_fn, focal_loss_fn, boxes=boxes)

            # Optional L2 regularisation on LoRA adapter weights
            if args.l2_reg > 0:
                l2 = sum(p.pow(2).sum() for p in lora_params)
                loss = loss + args.l2_reg * l2

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss  += loss.item()

            if (bi + 1) % 10 == 0 or (bi + 1) == len(train_loader):
                avg    = epoch_loss / (bi + 1)
                cur_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                print(f"  Epoch [{epoch:>3}/{args.epochs}] "
                      f"Step [{bi+1:>4}/{len(train_loader)}]  "
                      f"Loss: {loss.item():.4f}  Avg: {avg:.4f}  "
                      f"LR: {cur_lr:.2e}  Time: {elapsed:.1f}s")

        train_loss = epoch_loss / len(train_loader)

        # ---- Validation ----------------------------------------------------
        sam2_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, gt_masks, point_coords, point_labels, boxes in val_loader:
                images       = images.to(device)
                gt_masks     = gt_masks.to(device, dtype=torch.float32)
                point_coords = point_coords.to(device, dtype=torch.float32)
                point_labels = point_labels.to(device, dtype=torch.int32)
                boxes        = boxes.to(device, dtype=torch.float32) if boxes is not None else None
                loss, _ = forward_pass(sam2_model, images, gt_masks, point_coords,
                                       point_labels, dtype, device,
                                       dice_loss_fn, focal_loss_fn, boxes=boxes)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch:>3}/{args.epochs} — "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  LR: {cur_lr:.2e}\n")

        # ---- Save LoRA adapters checkpoint every 5 epochs -----------------
        if epoch % 5 == 0 or epoch == args.epochs:
            adapters_path = Path(args.lora_only_out).with_suffix(f".e{epoch:03d}.pt")
            torch.save(lora_state_dict(sam2_model), adapters_path)
            print(f"[CKPT] LoRA adapters saved: {adapters_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best LoRA adapters
            torch.save(lora_state_dict(sam2_model), args.lora_only_out)
            print(f"[BEST] Val loss {val_loss:.4f} — LoRA adapters → {args.lora_only_out}")

    # ---- Final: merge LoRA into base and save full model for inference ----
    print("\n[INFO] Merging LoRA weights into base model ...")
    sam2_model = merge_lora_into_base(sam2_model)
    torch.save(sam2_model.state_dict(), args.output)
    print(f"[DONE] Merged model saved → {args.output}")
    print(f"       LoRA adapters only → {args.lora_only_out}")
    print(f"       Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
