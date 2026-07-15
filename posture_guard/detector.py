"""
Posture detection logic.
All checks are delta-based against a calibrated baseline.
Uses a rolling smoothing window to reduce jitter and catch subtle slouches.
"""
import numpy as np
import cv2
from collections import deque
from . import config as cfg
from .import settings

# -- Smoothing window
# Stores the last N frames of raw metric deltas.
# We report bad posture only when the "averaeg" delta exceeds the threshold, which filters out single-frame noise while still catching sustained slouches.
_SMOOTH_WINDOW = 8
_history: deque = deque(maxlen=_SMOOTH_WINDOW)

def reset_history():
    """Call after recalibration so stale frames don't pollute new baseline."""
    _history.clear()

# Geometry
def _angle_from_horizontal(p1, p2) -> float:
    """Returns angle in [0, 90] — 0 means perfectly level."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    a = abs(np.degrees(np.arctan2(dy, dx)))
    return 180 - a if a > 90 else a

def _visible(lm, *idxs):
    return all(lm[i].visibility >= cfg.VISIBILITY_MIN for i in idxs)

def pt(lm, idx, w, h) -> tuple:
    return lm[idx].x * w, lm[idx].y * h

def collect_sample(lm, w, h):
    sample = {}
    if _visible(lm, cfg.LEFT_SHOULDER, cfg.RIGHT_SHOULDER):
        ls = pt(lm, cfg.LEFT_SHOULDER, w, h)
        rs = pt(lm, cfg.RIGHT_SHOULDER, w, h)
        
        sample["sholder_mid_y"] = (lm[cfg.LEFT_SHOULDER].y + lm[cfg.RIGHT_SHOULDER]) / 2
        sample["sholder_width"] = abs(lm[cfg.LEFT_SHOULDER].x - lm[cfg.RIGHT_SHOULDER].x)
        sample["sholder_tilt"] = _angle_from_horizontal(ls, rs)
        sample["lean_offset"] = lm[cfg.NOSE].x - (lm[cfg.LEFT_SHOULDER].x + lm[cfg.RIGHT_SHOULDER].x) / 2
        sample["neck_y_gap"]     = lm[cfg.NOSE].y - lm[cfg.LEFT_SHOULDER].y
    if _visible(lm, cfg.LEFT_EAR, cfg.RIGHT_EAR):
        sample["head_tilt"] = _angle_from_horizontal(
            pt(lm, cfg.LEFT_EAR, w, h), pt(lm, cfg.RIGHT_EAR, w, h)
        )

    return sample

def build_baseline(samples) -> dict:
    """Average a list of samples into a single baseline."""
    keys = {k for s in samples for k in s}
    return {k: float(np.mean([s[k] for s in samples if k in s])) for k in keys}

def _extract_metrics(lm, w: int, h: int, baseline: dict) -> dict:
    """Extract raw per-frame delta values for every check."""
    metrics = {}
    
    if _visible(lm, cfg.LEFT_SHOULDER, cfg.RIGHT_SHOULDER):
        ls = pt(lm, cfg.LEFT_SHOULDER, w, h)
        rs = pt(lm, cfg.RIGHT_SHOULDER, w, h)
        mid_y = (lm[cfg.LEFT_SHOULDER].y + lm[cfg.RIGHT_SHOULDER].y) / 2
        mid_x = (lm[cfg.LEFT_SHOULDER].x + lm[cfg.RIGHT_SHOULDER].x) / 2
        width = abs(lm[cfg.LEFT_SHOULDER].x - lm[cfg.RIGHT_SHOULDER].x)
        tilt = _angle_from_horizontal(ls, rs)
        
        metrics["shoulder_drop"] = mid_y - baseline.get("sholder_mid_y", mid_y)
        metrics["shoulder_width"] = baseline.get("shoulder_width", width) - width
        metrics["shoulder_tilt"] = tilt - baseline.get("shoulder_tilt", 0)
        
        if _visible(lm, cfg.NOSE):
            neck_gap = lm[cfg.NOSE].y - lm[cfg.LEFT_SHOULDER].y
            metrics["neck_gap"] = neck_gap - baseline.get("neck_y_gap", neck_gap)
            metrics["lean"] = abs((lm[cfg.NOSE].x - mid_x) - baseline.get("lean_offset", 0))
            
    if _visible(lm, cfg.LEFT_EAR, cfg.RIGHT_EAR):
        tilt = _angle_from_horizontal(pt(lm, cfg.LEFT_EAR, w, h), pt(lm, cfg.RIGHT_EAR, w, h))
        
        metrics["head_tilt"] = tilt - baseline.get("head_tilt", 0)
        
    return metrics

def _smoothed_metrics() -> dict:
    """Average metrics across the history window."""
    if not _history:
        return {}
    
    keys = {k for frame in _history for k in frame}
    
    return {
        k: float(np.mean([f[k] for f in _history if k in f])) for k in keys
    }

# Live check
def check_posture(lm, w, h, baseline: dict) -> tuple:
    """
    Returns (is_good: bool, reason: str).
    Uses smoothed averages over the last N frames to reduce jitter.
    Thresholds are scaled by the user's sensitivity setting.
    """
    raw = _extract_metrics(lm, w, h, baseline)
    _history.append(raw)
    smoothed = _smoothed_metrics()
    
    # Apply sensivity multiplier from user settings
    user_settings = settings.load()
    mult = settings.SENSIVITY_MULTIPLIERS.get(user_settings.get("sensivity", "medium"), 1.0)
    
    issues = []
    
    


# Debug overlay
def draw_debug_overlay(frame, lm, w, h, baseline):
    """Show live metric values vs baseline for tuning."""
    if not _visible(lm, cfg.LEFT_SHOULDER, cfg.RIGHT_SHOULDER):
        return

    mid_y  = (lm[cfg.LEFT_SHOULDER].y + lm[cfg.RIGHT_SHOULDER].y) / 2
    width  = abs(lm[cfg.LEFT_SHOULDER].x - lm[cfg.RIGHT_SHOULDER].x)
    neck_y = lm[cfg.NOSE].y - lm[cfg.LEFT_SHOULDER].y if _visible(lm, cfg.NOSE) else 0

    metrics = [
        (f"shld_drop:  {mid_y - baseline.get('shoulder_mid_y', mid_y):+.3f}",  cfg.SHOULDER_DROP_THRESHOLD),
        (f"shld_width: {baseline.get('shoulder_width', width) - width:+.3f}",  cfg.SHOULDER_WIDTH_THRESHOLD),
        (f"neck_gap:   {neck_y - baseline.get('neck_y_gap', neck_y):+.3f}",    cfg.HEAD_RISE_THRESHOLD),
    ]
    for i, (label, threshold) in enumerate(metrics):
        color = (100, 100, 220) if abs(float(label.split()[-1])) > threshold else (180, 180, 180)
        cv2.putText(frame, label, (10, h - 20 - i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    # Draw skeleton
    def p(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)
    
    for a, b in cfg.SKELETON_CONNECTIONS:
        try:
            if _visible(lm, a, b):
                cv2.line(frame, p(a), p(b), (245, 66, 230), 2)
                
        except IndexError:
            pass
    
    for i in range(len(lm)):
        try:
            if lm[i].visibility >= cfg.VISIBILITY_MIN:
                cv2.circle(frame, p(i), 4, (245, 117, 66), -1)
                
        except IndexError:
            pass
