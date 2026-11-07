#!/usr/bin/env python3
"""
auto_label.py - automatic pseudo-label generation for the black phone casing.

Reads  dataset/undistorted/*.jpg
Writes dataset/labels/masks/<name>.png        binary 255/0 mask, same size as input
       dataset/labels/overlays/<name>.jpg     boundary overlay (flags/status in banner)
       dataset/labels/reports/report_<ts>.csv per-image machine-readable record
       dataset/labels/reports/summary_<ts>.json   run metadata + QC counters
       dataset/labels/reports/PSEUDO_LABELS_NOTICE.txt

THE OUTPUTS ARE AUTOMATICALLY GENERATED PSEUDO-LABELS. They are NOT manually
verified ground truth and need quality assessment before use.

Pipeline (CPU only, OpenCV + NumPy):
  1. Find undistortion-invalid black regions (near-pure-black components that
     touch the image border) and exclude them.
  2. Locate the red/white playing card (red-pixel clusters -> hull) and exclude it.
  3. Dark-region mask at several adaptive thresholds (Otsu on valid pixels,
     clamped, then scaled down to separate casing from shadows).
  4. Morphology + hole filling + connected components -> candidates.
  5. Hard geometric filters, then ranking by rectangularity, solidity, darkness,
     aspect ratio, centrality, border contact and card overlap.
  6. GrabCut refinement (accepted only if it stays close to the candidate).
  7. Plausibility checks -> status ok / review / failed. Failed images get NO mask.

Usage:
  python dataset/auto_label.py                      # whole dataset
  python dataset/auto_label.py --limit 5 --overwrite
  python dataset/auto_label.py --only img_003.jpg
  python dataset/auto_label.py --config my_cfg.json # override Config fields
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ALGORITHM = "auto_label_v1: invalid-region + card exclusion, multi-threshold dark mask, geometric ranking, GrabCut"
VALID_EXT = {".jpg", ".jpeg", ".png"}
# Flags that are informational only and do not force manual review.
INFO_FLAGS = {"grabcut_rejected", "grabcut_skipped"}

log = logging.getLogger("auto_label")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    seed: int = 0
    work_max_side: int = 800            # analysis resolution (long side); masks are upscaled
    min_image_side: int = 200           # reject tiny/invalid images
    # undistortion border
    invalid_max_value: int = 3          # max(B,G,R) <= this counts as "pure black"
    invalid_min_area_frac: float = 0.002
    invalid_dilate_px: int = 3
    # card detection (HSV, OpenCV hue 0-179)
    card_red_hue_lo: int = 10
    card_red_hue_hi: int = 170
    card_red_sat_min: int = 90
    card_red_val_min: int = 70
    card_min_area_frac: float = 0.0005
    card_close_frac: float = 0.03       # closing kernel, fraction of short side
    card_margin_frac: float = 0.012     # extra margin to cover white card border
    # dark mask
    blur_ksize: int = 5
    dark_min_thresh: int = 35
    dark_max_thresh: int = 90
    threshold_scales: tuple = (1.0, 0.75, 0.55)
    open_frac: float = 0.004
    close_frac: float = 0.012
    # candidate hard filters
    min_area_frac: float = 0.02
    max_area_frac: float = 0.75
    min_solidity: float = 0.82
    min_rectangularity: float = 0.65
    aspect_min: float = 1.3
    aspect_max: float = 3.3
    max_card_overlap_reject: float = 0.5
    max_card_overlap_review: float = 0.15
    # ranking / ambiguity / stability
    ambiguity_ratio: float = 0.85
    stability_area_ratio: float = 0.75
    fragment_area_frac: float = 0.15
    fragment_gap_frac: float = 0.03
    min_score: float = 0.45
    # grabcut
    use_grabcut: bool = True
    grabcut_iters: int = 3
    grabcut_min_iou: float = 0.80
    # final mask QC (percent of image area)
    final_min_area_pct: float = 2.0
    final_max_area_pct: float = 75.0

    @classmethod
    def load(cls, path: Path | None) -> "Config":
        cfg = cls()
        if path:
            data = json.loads(Path(path).read_text())
            names = {f.name for f in dataclasses.fields(cls)}
            unknown = set(data) - names
            if unknown:
                raise ValueError(f"Unknown config keys: {sorted(unknown)}")
            for k, v in data.items():
                setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
        return cfg


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def odd(n: float, lo: int = 3) -> int:
    n = max(lo, int(round(n)))
    return n if n % 2 == 1 else n + 1


def ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, thickness=cv2.FILLED)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero((a > 0) & (b > 0))
    union = np.count_nonzero((a > 0) | (b > 0))
    return inter / union if union else 0.0


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# --------------------------------------------------------------------------- #
# Scene analysis: invalid border and playing card
# --------------------------------------------------------------------------- #
def find_invalid_region(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Pure-black components touching the image border = undistortion fill."""
    h, w = img.shape[:2]
    black = (img.max(axis=2) <= cfg.invalid_max_value).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=4)
    invalid = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches and area >= cfg.invalid_min_area_frac * h * w:
            invalid[lab == i] = 255
    if cfg.invalid_dilate_px > 0 and invalid.any():
        invalid = cv2.dilate(invalid, ellipse(2 * cfg.invalid_dilate_px + 1))
    return invalid


def find_card_region(hsv: np.ndarray, cfg: Config) -> np.ndarray:
    """Hull of the red pattern clusters (+ margin for the white border)."""
    h, w = hsv.shape[:2]
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((hh <= cfg.card_red_hue_lo) | (hh >= cfg.card_red_hue_hi))
           & (ss >= cfg.card_red_sat_min) & (vv >= cfg.card_red_val_min)).astype(np.uint8) * 255
    short = min(h, w)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, ellipse(odd(cfg.card_close_frac * short)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        red, connectivity=8
    )

    selected = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.card_min_area_frac * h * w:
            selected[lab == i] = 255

    card = np.zeros((h, w), dtype=np.uint8)

    if selected.any():
        pts = cv2.findNonZero(selected)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(card, hull, 255)
    if card.any():
        card = cv2.dilate(
            card,
            ellipse(odd(cfg.card_margin_frac * short * 2))
        )

    return card


# --------------------------------------------------------------------------- #
# Candidate generation and scoring
# --------------------------------------------------------------------------- #
def dark_threshold(v_blur: np.ndarray, usable: np.ndarray, cfg: Config) -> int:
    vals = v_blur[usable > 0]
    if vals.size < 100:
        return cfg.dark_max_thresh
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(min(cfg.dark_max_thresh, max(cfg.dark_min_thresh, t)))


def describe(mask: np.ndarray, v_blur: np.ndarray, hsv: np.ndarray, zone: np.ndarray,
             card: np.ndarray, thr: int, cfg: Config) -> dict | None:
    """Geometric/colour evidence for one filled candidate mask (work scale)."""
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = float(np.count_nonzero(mask))
    hull = cv2.convexHull(c)
    hull_area = max(cv2.contourArea(hull), 1.0)
    (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
    long_s, short_s = max(rw, rh), max(min(rw, rh), 1.0)
    hull_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    hull_n = max(np.count_nonzero(hull_mask), 1)
    sel = mask > 0
    d = dict(
        mask=mask, area=area, area_frac=area / (h * w),
        solidity=area / hull_area,
        rectangularity=area / max(rw * rh, 1.0),
        aspect=long_s / short_s,
        med_v=float(np.median(v_blur[sel])),
        mean_sat=float(hsv[..., 1][sel].mean()),
        touches=bool(np.count_nonzero(sel & (zone > 0)) > 0),
        card_overlap=np.count_nonzero((hull_mask > 0) & (card > 0)) / hull_n,
        centre_dist=float(np.hypot(cx - w / 2, cy - h / 2) / (0.5 * np.hypot(w, h))),
        bbox=cv2.boundingRect(c), thr=thr,
    )
    return d


def plausible(d: dict, cfg: Config) -> bool:
    return (cfg.min_area_frac <= d["area_frac"] <= cfg.max_area_frac
            and d["solidity"] >= cfg.min_solidity
            and d["rectangularity"] >= cfg.min_rectangularity
            and cfg.aspect_min <= d["aspect"] <= cfg.aspect_max
            and d["card_overlap"] <= cfg.max_card_overlap_reject)


def score(d: dict) -> float:
    rect = clip01((d["rectangularity"] - 0.6) / 0.4)
    solid = clip01((d["solidity"] - 0.7) / 0.3)
    dark = clip01(1.0 - d["med_v"] / max(d["thr"], 1))
    aspect = clip01(1.0 - abs(np.log(d["aspect"] / 1.95)) / np.log(1.8))
    central = clip01(1.0 - d["centre_dist"])
    s = 0.30 * rect + 0.20 * solid + 0.20 * dark + 0.15 * aspect + 0.15 * central
    s -= 0.25 * d["touches"] + 0.40 * clip01(d["card_overlap"])
    return float(s)


def generate_candidates(img_w: np.ndarray, cfg: Config):
    """Return (candidates, context) at work resolution."""
    h, w = img_w.shape[:2]
    short = min(h, w)
    hsv = cv2.cvtColor(img_w, cv2.COLOR_BGR2HSV)
    v_blur = cv2.GaussianBlur(hsv[..., 2], (odd(cfg.blur_ksize), odd(cfg.blur_ksize)), 0)

    invalid = find_invalid_region(img_w, cfg)
    card = find_card_region(hsv, cfg)
    usable = ((invalid == 0) & (card == 0)).astype(np.uint8) * 255

    zone = invalid.copy()
    zone[0, :] = zone[-1, :] = 255
    zone[:, 0] = zone[:, -1] = 255

    base_t = dark_threshold(v_blur, usable, cfg)
    open_k = ellipse(odd(cfg.open_frac * short))
    close_k = ellipse(odd(cfg.close_frac * short))

    cands, raw_masks = [], []
    for scale in cfg.threshold_scales:
        t = max(5, int(base_t * scale))
        dark = ((v_blur <= t) & (usable > 0)).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, open_k)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, close_k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        raw_masks.append((t, dark, lab, stats))
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 0.5 * cfg.min_area_frac * h * w:
                continue
            comp = fill_holes((lab == i).astype(np.uint8) * 255)
            d = describe(comp, v_blur, hsv, zone, card, t, cfg)
            if d is None:
                continue
            d["raw_area"] = float(stats[i, cv2.CC_STAT_AREA])
            d["fill_ratio"] = d["raw_area"] / max(d["area"], 1.0)
            d["score"] = score(d)
            d["plausible"] = plausible(d, cfg)
            cands.append(d)
    ctx = dict(hsv=hsv, invalid=invalid, card=card, zone=zone, base_t=base_t, raw=raw_masks)
    return cands, ctx


def dedupe(plaus: list[dict]) -> list[dict]:
    kept = []
    for d in sorted(plaus, key=lambda x: -x["score"]):
        if all(iou(d["mask"], k["mask"]) < 0.5 for k in kept):
            kept.append(d)
    return kept


# --------------------------------------------------------------------------- #
# Refinement
# --------------------------------------------------------------------------- #
def grabcut_refine(img_w: np.ndarray, cand: np.ndarray, invalid: np.ndarray,
                   card: np.ndarray, cfg: Config) -> np.ndarray | None:
    h, w = cand.shape
    x, y, bw, bh = cv2.boundingRect(cand)
    m = int(0.10 * max(bw, bh)) + 4
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(w, x + bw + m), min(h, y + bh + m)
    crop = np.ascontiguousarray(img_w[y0:y1, x0:x1])
    cm = cand[y0:y1, x0:x1]
    r = max(3, int(0.04 * max(bw, bh)))
    gc = np.full(cm.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(cm, ellipse(2 * r + 1)) > 0] = cv2.GC_PR_BGD
    gc[cm > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(cm, ellipse(2 * r + 1)) > 0] = cv2.GC_FGD
    gc[(invalid[y0:y1, x0:x1] > 0) | (card[y0:y1, x0:x1] > 0)] = cv2.GC_BGD
    if not (gc == cv2.GC_FGD).any():
        return None
    cv2.setRNGSeed(cfg.seed)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, gc, None, bgd, fgd, cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n < 2:
        return None
    best = max(range(1, n), key=lambda i: np.count_nonzero((lab == i) & (cm > 0)))
    comp = fill_holes((lab == best).astype(np.uint8) * 255)
    out = np.zeros_like(cand)
    out[y0:y1, x0:x1] = comp
    return out


# --------------------------------------------------------------------------- #
# Per-image processing
# --------------------------------------------------------------------------- #
def segment_image(img: np.ndarray, cfg: Config) -> dict:
    """Segment one image. Returns dict with status, flags, full-res mask (or None)."""
    H, W = img.shape[:2]
    scale = min(1.0, cfg.work_max_side / max(H, W))
    img_w = cv2.resize(img, (int(round(W * scale)), int(round(H * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    h, w = img_w.shape[:2]

    cands, ctx = generate_candidates(img_w, cfg)

    if not isinstance(ctx, dict):
        raise RuntimeError("generate_candidates returned invalid context")

    card = ctx.get("card")
    invalid = ctx.get("invalid")

    if card is None or invalid is None:
        raise RuntimeError(
            "generate_candidates did not return valid card/invalid masks"
        )

    flags: list[str] = []
    res = dict(
        flags=flags,
        mask=None,
        card_mask=None,
        invalid_mask=None,
        rejected=None,
        n_plausible=0,
        score=None,
        second_score=None,
        thr=ctx["base_t"],
        grabcut_applied=False,
        rect_aspect=None,
        solidity=None,
        rectangularity=None,
        bbox=None,
        card_found=bool(card.any())
    )
    if not card.any():
        flags.append("card_not_found")

    def up(m: np.ndarray) -> np.ndarray:
        if m.shape[:2] == (H, W):
            return m
        return cv2.resize(
            m,
            (W, H),
            interpolation=cv2.INTER_NEAREST
        )

    res["card_mask"], res["invalid_mask"] = up(card), up(invalid)

    plaus = [c for c in cands if c["plausible"] and c["score"] >= cfg.min_score]
    ranked = dedupe(plaus)
    res["n_plausible"] = len(ranked)

    if not ranked:
        flags.append("no_plausible_casing")
        leftovers = sorted(cands, key=lambda c: -c["score"])
        if leftovers:
            res["rejected"] = up(leftovers[0]["mask"])
        res["status"] = "failed"
        return res

    best = ranked[0]
    res["score"] = round(best["score"], 4)
    if len(ranked) > 1:
        res["second_score"] = round(ranked[1]["score"], 4)
        if ranked[1]["score"] >= cfg.ambiguity_ratio * best["score"]:
            flags.append("ambiguous_multiple_candidates")

    # stability across thresholds: does the region keep its size?
    same = [c["area"] for c in plaus if iou(c["mask"], best["mask"]) > 0.5]
    if len(same) > 1 and min(same) / max(same) < cfg.stability_area_ratio:
        flags.append("threshold_unstable")

    # fragmentation: raw dark pixels with many holes, or sizeable neighbours nearby
    if best["fill_ratio"] < 0.70:
        flags.append("fragmented_contour")
    else:
        gap = ellipse(odd(cfg.fragment_gap_frac * min(h, w) * 2))
        near = cv2.dilate(best["mask"], gap)
        for t, dark, lab, stats in ctx["raw"]:
            if t != best["thr"]:
                continue
            for i in np.unique(lab[(near > 0) & (best["mask"] == 0)]):
                if i and stats[i, cv2.CC_STAT_AREA] >= cfg.fragment_area_frac * best["area"]:
                    if np.count_nonzero((lab == i) & (best["mask"] > 0)) == 0:
                        flags.append("fragmented_contour")
                        break

    if best["card_overlap"] > cfg.max_card_overlap_review:
        flags.append("may_include_card")
    if best["touches"]:
        flags.append("touches_boundary")

    final = best["mask"]
    if cfg.use_grabcut:
        try:
            refined = grabcut_refine(img_w, best["mask"], invalid, card, cfg)
        except cv2.error as exc:
            log.debug("grabcut error: %s", exc)
            refined = None
        if refined is None:
            flags.append("grabcut_skipped")
        elif iou(refined, best["mask"]) >= cfg.grabcut_min_iou:
            final = refined
            res["grabcut_applied"] = True
        else:
            flags.append("grabcut_rejected")

    full = up(final)
    area_pct = 100.0 * np.count_nonzero(full) / (H * W)
    if np.count_nonzero(full) == 0:
        flags.append("empty_mask")
    elif not (cfg.final_min_area_pct <= area_pct <= cfg.final_max_area_pct):
        flags.append("implausible_area")

    # final boundary check on the refined mask
    zone_full = up(ctx["zone"])
    if np.any((full > 0) & (zone_full > 0)) and "touches_boundary" not in flags:
        flags.append("touches_boundary")

    d = describe(final, cv2.GaussianBlur(ctx["hsv"][..., 2], (5, 5), 0), ctx["hsv"],
                 ctx["zone"], card, best["thr"], cfg)
    if d:
        res.update(rect_aspect=round(d["aspect"], 3), solidity=round(d["solidity"], 4),
                   rectangularity=round(d["rectangularity"], 4),
                   bbox=[int(v / scale) for v in d["bbox"]])

    flags[:] = list(dict.fromkeys(flags))
    res["mask"] = full if np.count_nonzero(full) else None
    if res["mask"] is None:
        res["status"] = "failed"
    elif any(f not in INFO_FLAGS for f in flags):
        res["status"] = "review"
    else:
        res["status"] = "ok"
    return res


# --------------------------------------------------------------------------- #
# Overlay
# --------------------------------------------------------------------------- #
def make_overlay(img: np.ndarray, res: dict, name: str) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    t = max(2, int(round(min(H, W) / 300)))
    if res["invalid_mask"] is not None and res["invalid_mask"].any():
        tint = out.copy()
        tint[res["invalid_mask"] > 0] = (255, 0, 0)
        out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
    if res["card_mask"] is not None and res["card_mask"].any():
        cs, _ = cv2.findContours(res["card_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (255, 0, 255), t)
    m = res["mask"] if res["mask"] is not None else res["rejected"]
    if m is not None:
        colour = (0, 255, 0) if res["status"] == "ok" else (0, 165, 255) if res["status"] == "review" else (0, 0, 255)
        fill = out.copy()
        fill[m > 0] = colour
        out = cv2.addWeighted(fill, 0.25, out, 0.75, 0)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, colour, t)
    border = {"ok": (0, 160, 0), "review": (0, 140, 255), "failed": (0, 0, 255)}.get(res["status"], (128, 128, 128))
    cv2.rectangle(out, (0, 0), (W - 1, H - 1), border, 3 * t)
    text = f"{name} | {res['status'].upper()} | " + (",".join(res["flags"]) or "no flags") + " | AUTO PSEUDO-LABEL"
    fs = max(0.5, min(H, W) / 1200)
    cv2.rectangle(out, (0, 0), (W, int(40 * fs) + 14), (0, 0, 0), -1)
    cv2.putText(out, text, (8, int(30 * fs) + 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, t // 2), cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Dataset plumbing
# --------------------------------------------------------------------------- #
def read_split(path: Path) -> list[str]:
    if not path.exists():
        log.warning("Split file missing: %s", path)
        return []
    items = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            items.append(Path(line).stem)
    return items


def discover(src: Path) -> list[Path]:
    return sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT)


def archive_existing(paths: list[Path], archive_dir: Path) -> None:
    for p in paths:
        if p.exists():
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(archive_dir / p.name))


def process_one(path: Path, dirs: dict, cfg: Config, split: str, overwrite: bool, stamp: str) -> dict:
    rec = dict(image=path.name, split=split, status="failed", flags="", error="", area_pct="",
               n_plausible=0, score="", second_score="", threshold="", grabcut_applied=False,
               card_found="", rect_aspect="", solidity="", rectangularity="", bbox="",
               width="", height="", touches_boundary=False, mask_file="", overlay_file="", seconds="")
    t0 = time.time()
    mask_path = dirs["masks"] / f"{path.stem}.png"
    ov_path = dirs["overlays"] / f"{path.stem}.jpg"
    try:
        if mask_path.exists() or ov_path.exists():
            if not overwrite:
                rec.update(status="skipped_existing", flags="existing_label_preserved")
                return rec
            archive_existing([mask_path, ov_path], dirs["archive"] / stamp)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("unreadable image")
        H, W = img.shape[:2]
        if min(H, W) < cfg.min_image_side:
            raise ValueError(f"image too small: {W}x{H}")
        rec.update(width=W, height=H)
        res = segment_image(img, cfg)
        rec.update(status=res["status"], flags=";".join(res["flags"]), n_plausible=res["n_plausible"],
                   score=res["score"] if res["score"] is not None else "",
                   second_score=res["second_score"] if res["second_score"] is not None else "",
                   threshold=res["thr"], grabcut_applied=res["grabcut_applied"],
                   card_found=res["card_found"], rect_aspect=res["rect_aspect"] or "",
                   solidity=res["solidity"] or "", rectangularity=res["rectangularity"] or "",
                   bbox=res["bbox"] or "", touches_boundary="touches_boundary" in res["flags"])
        if res["mask"] is not None:
            assert res["mask"].shape[:2] == (H, W)
            cv2.imwrite(str(mask_path), res["mask"])
            rec["mask_file"] = str(mask_path.relative_to(dirs["root"]))
            rec["area_pct"] = round(100.0 * np.count_nonzero(res["mask"]) / (H * W), 3)
        cv2.imwrite(str(ov_path), make_overlay(img, res, path.name), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rec["overlay_file"] = str(ov_path.relative_to(dirs["root"]))
    except Exception as exc:  # keep the batch alive
        log.exception("Failed on %s", path.name)
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", flags="processing_error")
    rec["seconds"] = round(time.time() - t0, 3)
    return rec


def summarise(records: list[dict], n_discovered: int) -> dict:
    ok = [r for r in records if r["status"] in ("ok", "review")]
    failed = [r for r in records if r["status"] == "failed"]
    areas = [r["area_pct"] for r in ok if r["area_pct"] != ""]
    has = lambda r, f: f in r["flags"].split(";")
    by_split: dict = {}
    for r in records:
        s = by_split.setdefault(r["split"] or "none", dict(total=0, ok=0, review=0, failed=0, skipped=0))
        s["total"] += 1
        key = r["status"] if r["status"] in ("ok", "review", "failed") else "skipped"
        s[key] += 1
    return dict(
        images_discovered=n_discovered,
        images_processed_successfully=len(ok),
        images_ok_no_flags_requiring_review=sum(r["status"] == "ok" for r in records),
        images_flagged_for_review=sum(r["status"] == "review" for r in records),
        images_failed=len(failed),
        images_skipped_existing=sum(r["status"] == "skipped_existing" for r in records),
        images_no_casing_detected=sum(has(r, "no_plausible_casing") for r in records),
        images_multiple_plausible_candidates=sum(r["n_plausible"] > 1 for r in records),
        images_ambiguous=sum(has(r, "ambiguous_multiple_candidates") for r in records),
        empty_or_invalid_masks=sum(has(r, "empty_mask") or has(r, "implausible_area") for r in records),
        masks_touching_boundary=sum(bool(r["touches_boundary"]) for r in records),
        mask_area_pct=dict(min=min(areas) if areas else None, max=max(areas) if areas else None,
                           mean=round(float(np.mean(areas)), 3) if areas else None,
                           median=round(float(np.median(areas)), 3) if areas else None),
        per_split=by_split,
    )


def write_reports(records, summary, cfg, dirs, stamp, argv):
    rep = dirs["reports"]
    rep.mkdir(parents=True, exist_ok=True)
    csv_path = rep / f"report_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["image"])
        w.writeheader()
        w.writerows(records)
    meta = dict(
        label_type="AUTOMATICALLY GENERATED PSEUDO-LABELS",
        manually_verified=False,
        notice="Not ground truth. Requires visual quality assessment; review/failed items must be checked or excluded.",
        algorithm=ALGORITHM,
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        command=" ".join(argv), config=dataclasses.asdict(cfg),
        versions=dict(python=sys.version.split()[0], opencv=cv2.__version__, numpy=np.__version__),
        summary=summary, images=records,
    )
    (rep / f"summary_{stamp}.json").write_text(json.dumps(meta, indent=2, default=str))
    notice = rep / "PSEUDO_LABELS_NOTICE.txt"
    if not notice.exists():
        notice.write_text(
            "The masks in dataset/labels/masks were generated AUTOMATICALLY by dataset/auto_label.py.\n"
            "They are pseudo-labels, NOT manually verified ground-truth annotations.\n"
            "Known limitations: dark shadows or dark objects near the casing can distort the outline;\n"
            "undistortion borders are detected heuristically; GrabCut edges are approximate.\n"
            "Images with status 'review' need inspection; 'failed' images have no mask.\n"
            "Disclose this in the project report and assess label quality before training/evaluation.\n"
            "Test-set images/labels must not be used for training, tuning or checkpoint selection.\n")
    return csv_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    ap = argparse.ArgumentParser(description="Automatic casing segmentation (pseudo-labels).")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                    help="dataset root containing undistorted/, splits/ (default: script directory)")
    ap.add_argument("--config", type=Path, help="JSON file overriding Config fields")
    ap.add_argument("--limit", type=int, help="process only the first N images")
    ap.add_argument("--only", nargs="+", help="process only these filenames")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate existing labels (old ones are archived, never deleted)")
    ap.add_argument("--no-grabcut", action="store_true")
    ap.add_argument("--expected-count", type=int, default=129)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.load(args.config)
    if args.no_grabcut:
        cfg.use_grabcut = False
    cv2.setRNGSeed(cfg.seed)
    np.random.seed(cfg.seed)

    root = args.root.resolve()
    src = root / "undistorted"
    if not src.is_dir():
        log.error("Source directory not found: %s", src)
        return 2
    labels = root / "labels"
    dirs = dict(root=root, masks=labels / "masks", overlays=labels / "overlays",
                reports=labels / "reports", archive=labels / "archive")
    for k in ("masks", "overlays", "reports"):
        dirs[k].mkdir(parents=True, exist_ok=True)

    images = discover(src)
    n_discovered = len(images)
    log.info("Discovered %d images in %s", n_discovered, src)
    if n_discovered != args.expected_count:
        log.warning("Expected %d images but found %d", args.expected_count, n_discovered)

    # Splits are read-only inputs; membership is only reported, never altered.
    split_of: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for stem in read_split(root / "splits" / f"{name}.txt"):
            if stem in split_of:
                log.warning("%s appears in both %s and %s", stem, split_of[stem], name)
            split_of[stem] = name
    stems = {p.stem for p in images}
    for stem in sorted(set(split_of) - stems):
        log.warning("Split entry without image: %s", stem)
    for stem in sorted(stems - set(split_of)):
        log.warning("Image not in any split: %s", stem)

    if args.only:
        images = [p for p in images if p.name in set(args.only)]
    if args.limit:
        images = images[: args.limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(dirs["reports"] / f"auto_label_{stamp}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    records = []
    for i, p in enumerate(images, 1):
        rec = process_one(p, dirs, cfg, split_of.get(p.stem, ""), args.overwrite, stamp)
        records.append(rec)
        log.info("[%d/%d] %s -> %s %s", i, len(images), p.name, rec["status"], rec["flags"])

    if not records:
        log.error("No images processed.")
        return 1
    summary = summarise(records, n_discovered)
    csv_path = write_reports(records, summary, cfg, dirs, stamp, argv)
    log.info("Summary:\n%s", json.dumps(summary, indent=2))
    log.info("Report: %s", csv_path)
    log.warning("All masks are AUTOMATIC PSEUDO-LABELS, not verified ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

