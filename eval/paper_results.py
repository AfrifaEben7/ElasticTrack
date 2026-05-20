"""
paper_results.py — Complete paper results table combining:
  1. CTC SEG accuracy (5-config offline eval)
  2. Pipeline FPS (4-variant and elastic/bidir benchmarks)
  3. Power measurements
  4. Baseline comparisons

Usage:
    python3 paper_results.py
"""
import json, math
from pathlib import Path

BIDIR_BASE     = Path("/home/cps/Documents/biomed/ViT_part_nano/pipeline/benchmark_results/autotrack_bidir")
BIDIR_BASE_SAM21 = Path("/home/cps/Documents/biomed/ViT_part_nano/pipeline/benchmark_results/autotrack_bidir_sam21")
BIDIR_BASE_INN = Path("/home/cps/Documents/biomed/ViT_part_nano/pipeline/benchmark_results/autotrack_bidir_innovations")
BM_BASE        = Path("/home/cps/Documents/biomed/ViT_part_nano/pipeline/benchmark_results")
METRICS    = Path("/home/cps/Documents/biomed/ViT_part_nano/pipeline/benchmark_results/paper_metrics_final.json")

SEQS = ["DIC-C2DH-HeLa/01", "Fluo-N2DH-GOWT1/01", "Fluo-C2DL-MSC/01"]

BIDIR_CONFIGS = [
    ("c1_fixed",    "Fixed N_MEM=7 (offline bidir)"),
    ("c2_elastic",  "PE-Elastic (offline bidir)"),
    ("c3_bandit",   "PE-Elastic with LinUCB (offline bidir)"),
    ("c4_streaming","PE-Elastic Streaming-K5"),
    ("c5_full",     "Full System (stream, elastic, bandit)"),
]

PIPELINE_VARIANTS = [
    ("benchmark_wifi_fp32",         "WiFi FP32"),
    ("benchmark_wifi_fp16",         "WiFi FP16"),
    ("benchmark_wired_fp32",        "Wired FP32"),
    ("benchmark_wired_fp16",        "Wired FP16 (baseline)"),
    ("benchmark_wired_fp16_elastic","Wired FP16 with Elastic"),
    ("benchmark_wired_fp16_bidir",  "Wired FP16 with Bidir-K5"),
    ("benchmark_wired_fp16_elastic_bidir", "Wired FP16 with Elastic and Bidir-K5"),
]


INNOVATIONS = [
    ("i0_baseline",   "Baseline (SAM2.1 + YOLO, no innovations)"),
    ("i1_of_prompt",  "Opt.1: OF-prompt propagation (K_det=15)"),
    ("i1_of_k7",      "Opt.1: OF-prompt propagation (K_det=7)"),
    ("i3_clustered",  "Opt.3: Clustered memory bank"),
    ("i13_of_cluster","Opt.1+3: OF-prompt + clustered memory"),
    ("i2_tcadapt",    "Opt.2: Temporal consistency adaptation"),
    ("i123_stream",   "Clustered mem + Bidir-K5 streaming"),
]


def load_bidir(cfg: str, sam21: bool = False, inn: bool = False) -> dict:
    if inn:
        base = BIDIR_BASE_INN
    elif sam21:
        base = BIDIR_BASE_SAM21
    else:
        base = BIDIR_BASE
    p = base / cfg / "autotrack_bidir.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def load_bm(label: str) -> dict:
    p = BM_BASE / f"{label}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def get_seq(d: dict, seq: str) -> dict:
    k2 = seq.replace("/", "_").replace("-", "_")
    return d.get(seq) or d.get(k2) or {}


def mean3(d: dict, key: str) -> float:
    vals = [get_seq(d, s).get(key) for s in SEQS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def mean_nmem(d: dict) -> float:
    vals = []
    for s in SEQS:
        e = get_seq(d, s)
        nm = e.get("mean_n_mem")
        if nm is None:
            nm = e.get("elastic_fwd", {}).get("mean_n_mem")
        if nm is not None:
            vals.append(nm)
    return sum(vals) / len(vals) if vals else float("nan")


def nan_str(v, fmt=".4f", dflt="  —   "):
    return dflt if math.isnan(v) else format(v, fmt)


def section(title):
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")


# ──────────────────────────────────────────────────────────────────────────────
def print_accuracy_table():
    section("ACCURACY TABLE  (CTC SEG metric, mean over 3 sequences)")
    ref = 7.0
    print(f"  {'Method':<44} {'SEG_multi':>9} {'SEG_oracle':>10} "
          f"{'N_MEM':>6} {'Compute%':>9}")
    print("  " + "-" * 80)

    # Baselines
    rows = [
        ("SAM2 Hiera-Tiny base (offline, GT-init)", 0.324, float("nan"), 7.0, False),
        ("SAM2+LoRA CTC ft (offline eval_ctc, GT-init)", 0.887, float("nan"), 7.0, False),
        ("Pipeline fwd-only, auto-detect (10 FPS)", 0.000, float("nan"), 7.0, True),
    ]
    for lbl, sm, so, nm, stream in rows:
        s = "[S] " if stream else "    "
        print(f"  {s}{lbl:<44}  {nan_str(sm):>9}  {nan_str(so):>10}  "
              f"{nm:>6.1f}  {100*nm/ref:>8.1f}%")

    print("  " + "· " * 40)

    for cfg, lbl in BIDIR_CONFIGS:
        d = load_bidir(cfg)
        if not d:
            print(f"  (auto-detect bidir) {lbl:<36}  {'[pending]':>9}")
            continue
        sm  = mean3(d, "seg_multi")
        so  = mean3(d, "seg_oracle")
        nm  = mean_nmem(d)
        s   = "[S] " if "stream" in cfg else "    "
        comp = f"{100*nm/ref:.1f}%" if not math.isnan(nm) else "  —"
        print(f"  {s}(auto-detect bidir) {lbl:<36}  "
              f"{nan_str(sm):>9}  {nan_str(so):>10}  "
              f"{nan_str(nm, '.2f'):>6}  {comp:>9}")


def print_per_seq_table():
    section("PER-SEQUENCE BREAKDOWN")
    for seq in SEQS:
        print(f"\n  {seq}")
        print(f"    {'Config':<40} {'SEG_multi':>9} {'SEG_oracle':>10} "
              f"{'SEG_single':>10} {'N_MEM':>6}")
        print("    " + "-" * 78)
        for cfg, lbl in BIDIR_CONFIGS:
            d = load_bidir(cfg)
            e = get_seq(d, seq)
            if not e:
                print(f"    {lbl:<40}  {'[pending]':>9}")
                continue
            nm = e.get("mean_n_mem")
            if nm is None:
                nm = e.get("elastic_fwd", {}).get("mean_n_mem", float("nan"))
            print(f"    {lbl:<40}  "
                  f"{nan_str(e.get('seg_multi',float('nan'))):>9}  "
                  f"{nan_str(e.get('seg_oracle',float('nan'))):>10}  "
                  f"{nan_str(e.get('seg_single',float('nan'))):>10}  "
                  f"{nan_str(nm if nm else float('nan'), '.2f'):>6}")


def print_elastic_table():
    section("ELASTIC CONTROLLER: MODE DISTRIBUTION (mean across 3 sequences)")
    print(f"  {'Config':<40} {'SKIP%':>6} {'ECO%':>6} {'NORM%':>6} "
          f"{'BOOST%':>7} {'N_MEM':>6} {'Saved%':>7}")
    print("  " + "-" * 80)
    for cfg, lbl in BIDIR_CONFIGS:
        d = load_bidir(cfg)
        if not d:
            continue
        skip_v, eco_v, norm_v, bst_v = [], [], [], []
        for seq in SEQS:
            e  = get_seq(d, seq)
            ef = e.get("elastic_fwd", {})
            pc = ef.get("pct", {})
            if pc:
                skip_v.append(pc.get("SKIP", 0))
                eco_v.append(pc.get("ECO", 0))
                norm_v.append(pc.get("NORMAL", 0))
                bst_v.append(pc.get("BOOST", 0))
        if not skip_v:
            continue
        skip = sum(skip_v)/len(skip_v)
        eco  = sum(eco_v)/len(eco_v)
        norm = sum(norm_v)/len(norm_v)
        bst  = sum(bst_v)/len(bst_v)
        nm   = mean_nmem(d)
        saved = 100 * (7.0 - nm) / 7.0
        print(f"  {lbl:<40}  {skip:>6.1f}  {eco:>6.1f}  {norm:>6.1f}  "
              f"{bst:>7.1f}  {nan_str(nm, '.2f'):>6}  {saved:>6.1f}%")


def print_pipeline_table():
    section("PARTITION PIPELINE: FPS & LATENCY (3-device, DIC-C2DH-HeLa/01)")
    print(f"  {'Configuration':<36} {'FPS':>6} {'E2E_ms':>8} "
          f"{'Enc_ms':>8} {'Dec_ms':>8} {'Bidir':>6}")
    print("  " + "-" * 76)
    for label, name in PIPELINE_VARIANTS:
        d = load_bm(label)
        if not d:
            print(f"  {name:<36}  [pending]")
            continue
        fps    = d.get("throughput_fps", float("nan"))
        e2e    = d.get("e2e_ms", {}).get("mean", float("nan"))
        enc    = d.get("enc_ms", {}).get("mean", float("nan"))
        dec    = d.get("orin_ms", d.get("dec_ms", {})).get("mean", float("nan"))
        bidir  = "  K=5" if "bidir" in label else "   no"
        print(f"  {name:<36}  {fps:>6.2f}  {e2e:>8.0f}  "
              f"{enc:>8.1f}  {dec:>8.1f}  {bidir}")


def print_bandit_convergence():
    section("LINUCB BANDIT: CONVERGENCE (final adaptive ratios)")
    print(f"  {'Config':<36} {'Sequence':<28} "
          f"{'eco_r':>6} {'bst_r':>6} {'iou_ema':>8} {'steps':>6}")
    print("  " + "-" * 88)
    for cfg, lbl in BIDIR_CONFIGS:
        d = load_bidir(cfg)
        if not d:
            continue
        printed = False
        for seq in SEQS:
            e = get_seq(d, seq)
            b = e.get("bandit_fwd", {})
            if b and b.get("steps", 0) > 0:
                print(f"  {lbl:<36}  {seq:<28}  "
                      f"{b.get('eco_ratio',float('nan')):>6.3f}  "
                      f"{b.get('boost_ratio',float('nan')):>6.3f}  "
                      f"{b.get('iou_ema',float('nan')):>8.4f}  "
                      f"{b.get('steps',0):>6d}")
                printed = True
        if printed:
            print()


def print_sam21_table():
    section("SAM2.1-tiny v1 + YOLO REPROMPT  (5-config ablation, mean over 3 sequences)")
    ref = 7.0
    any_result = any(load_bidir(cfg, sam21=True) for cfg, _ in BIDIR_CONFIGS)
    if not any_result:
        print("  [pending — ablation still running on NX]")
        return
    print(f"  {'Method':<44} {'SEG_multi':>9} {'SEG_oracle':>10} "
          f"{'N_MEM':>6} {'Compute%':>9}")
    print("  " + "-" * 80)
    for cfg, lbl in BIDIR_CONFIGS:
        d = load_bidir(cfg, sam21=True)
        if not d:
            print(f"  {lbl:<44}  {'[pending]':>9}")
            continue
        sm   = mean3(d, "seg_multi")
        so   = mean3(d, "seg_oracle")
        nm   = mean_nmem(d)
        s    = "[S] " if "stream" in cfg else "    "
        comp = f"{100*nm/ref:.1f}%" if not math.isnan(nm) else "  —"
        print(f"  {s}{lbl:<44}  {nan_str(sm):>9}  {nan_str(so):>10}  "
              f"{nan_str(nm, '.2f'):>6}  {comp:>9}")

    print()
    print(f"  {'Per-sequence breakdown':}")
    print(f"  {'Config':<40} {'HeLa':>8} {'GOWT1':>8} {'MSC':>8}")
    print("  " + "-" * 66)
    for cfg, lbl in BIDIR_CONFIGS:
        d = load_bidir(cfg, sam21=True)
        if not d:
            continue
        hela  = get_seq(d, "DIC-C2DH-HeLa/01").get("seg_multi", float("nan"))
        gowt1 = get_seq(d, "Fluo-N2DH-GOWT1/01").get("seg_multi", float("nan"))
        msc   = get_seq(d, "Fluo-C2DL-MSC/01").get("seg_multi", float("nan"))
        s = "[S] " if "stream" in cfg else "    "
        print(f"  {s}{lbl:<40}  {nan_str(hela, '.4f'):>8}  "
              f"{nan_str(gowt1, '.4f'):>8}  {nan_str(msc, '.4f'):>8}")


def print_innovations_table():
    section("INNOVATIONS ABLATION  (SAM2.1-tiny v1, mean over 3 sequences)")
    ref = 7.0
    any_result = any(load_bidir(cfg, inn=True) for cfg, _ in INNOVATIONS)
    if not any_result:
        print("  [pending — innovations ablation running on NX]")
        return
    print(f"  {'Method':<48} {'SEG_multi':>9} {'SEG_oracle':>10} "
          f"{'HeLa':>8} {'GOWT1':>8} {'MSC':>8}")
    print("  " + "-" * 98)
    for cfg, lbl in INNOVATIONS:
        d = load_bidir(cfg, inn=True)
        if not d:
            print(f"  {lbl:<48}  {'[pending]':>9}")
            continue
        sm    = mean3(d, "seg_multi")
        so    = mean3(d, "seg_oracle")
        hela  = get_seq(d, "DIC-C2DH-HeLa/01").get("seg_multi", float("nan"))
        gowt1 = get_seq(d, "Fluo-N2DH-GOWT1/01").get("seg_multi", float("nan"))
        msc   = get_seq(d, "Fluo-C2DL-MSC/01").get("seg_multi", float("nan"))
        s     = "[S] " if "stream" in cfg else "    "
        print(f"  {s}{lbl:<48}  {nan_str(sm):>9}  {nan_str(so):>10}  "
              f"{nan_str(hela, '.4f'):>8}  {nan_str(gowt1, '.4f'):>8}  "
              f"{nan_str(msc, '.4f'):>8}")

    # Timing: sum elapsed_s across sequences
    print()
    print(f"  {'Method':<48} {'total_s':>8} {'fps_equiv':>10}")
    print("  " + "-" * 70)
    for cfg, lbl in INNOVATIONS:
        d = load_bidir(cfg, inn=True)
        if not d:
            continue
        total_s = sum(get_seq(d, s).get("elapsed_s", 0.0) for s in SEQS)
        # HeLa: 84 frames, GOWT1: 92 frames, MSC: 48 frames  (approx)
        total_frames = 84 + 92 + 48
        fps = total_frames / total_s if total_s > 0 else float("nan")
        s = "[S] " if "stream" in cfg else "    "
        print(f"  {s}{lbl:<48}  {nan_str(total_s, '.0f'):>8}s  "
              f"{nan_str(fps, '.3f'):>9} fps")


def main():
    print_accuracy_table()
    print_per_seq_table()
    print_elastic_table()
    print_pipeline_table()
    print_bandit_convergence()
    print_sam21_table()
    print_innovations_table()
    print()


if __name__ == "__main__":
    main()
