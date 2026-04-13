"""
Posture detection logic.
All checks are delta-based against a calibrated baseline,
so body proportions and camera angle don't affect accuracy.
"""
import numpy as np
import cv2
from . import config as cfg

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
        mid_y = (lm[cfg.LEFT_SHOULDER].y + lm[cfg.RIGHT_SHOULDER].y) / 2
        mid_x = (lm[cfg.LEFT_SHOULDER].x + lm[cfg.RIGHT_SHOULDER].x) / 2
        width  = abs(lm[cfg.LEFT_SHOULDER].x - lm[cfg.RIGHT_SHOULDER].x)
        tilt   = _angle_from_horizontal(ls, rs)

        sample["shoulder_mid_y"] = mid_y          # normalized Y of shoulder midpoint
        sample["shoulder_width"] = width          # normalized X distance between shoulders
        sample["shoulder_tilt"]  = tilt           # angle
        sample["lean_offset"]    = lm[cfg.NOSE].x - mid_x
        sample["neck_y_gap"]     = lm[cfg.NOSE].y - lm[cfg.LEFT_SHOULDER].y  # nose above/below shoulder

    if _visible(lm, cfg.LEFT_EAR, cfg.RIGHT_EAR):
        le = pt(lm, cfg.LEFT_EAR, w, h)
        re = pt(lm, cfg.RIGHT_EAR, w, h)
        sample["head_tilt"] = _angle_from_horizontal(le, re)

    return sample

def build_baseline(samples) -> dict:
    """Average a list of samples into a single baseline."""
    keys = {k for s in samples for k in s}
    return {k: float(np.mean([s[k] for s in samples if k in s])) for k in keys}

# Live check
def check_posture(lm, w, h, baseline):
    issues = []

    if _visible(lm, cfg.LEFT_SHOULDER, cfg.RIGHT_SHOULDER):
        ls = pt(lm, cfg.LEFT_SHOULDER, w, h)
        rs = pt(lm, cfg.RIGHT_SHOULDER, w, h)
        mid_y = (lm[cfg.LEFT_SHOULDER].y + lm[cfg.RIGHT_SHOULDER].y) / 2
        mid_x = (lm[cfg.LEFT_SHOULDER].x + lm[cfg.RIGHT_SHOULDER].x) / 2
        width  = abs(lm[cfg.LEFT_SHOULDER].x - lm[cfg.RIGHT_SHOULDER].x)
        tilt   = _angle_from_horizontal(ls, rs)

        # 1. Shoulders dropped (Y increased = lower in frame for most cameras)
        drop = mid_y - baseline.get("shoulder_mid_y", mid_y)
        if drop > cfg.SHOULDER_DROP_THRESHOLD:
            issues.append(f"Shoulders dropped ({drop:.3f})")

        # 2. Shoulders narrowed = hunching forward
        width_loss = baseline.get("shoulder_width", width) - width
        if width_loss > cfg.SHOULDER_WIDTH_THRESHOLD:
            issues.append(f"Hunching forward — shoulders narrowed ({width_loss:.3f})")

        # 3. Shoulder tilt — one side higher
        tilt_delta = tilt - baseline.get("shoulder_tilt", 0)
        if tilt_delta > cfg.SHOULDER_TILT_THRESHOLD:
            issues.append(f"Shoulders uneven (+{tilt_delta:.1f}°)")

        # 4. Head-to-shoulder gap changed (shoulders dropped away from head)
        if _visible(lm, cfg.NOSE):
            neck_gap   = lm[cfg.NOSE].y - lm[cfg.LEFT_SHOULDER].y
            gap_delta  = neck_gap - baseline.get("neck_y_gap", neck_gap)
            if gap_delta > cfg.HEAD_RISE_THRESHOLD:
                issues.append(f"Shoulders dropped from head ({gap_delta:.3f})")

        # 5. Lateral lean
        if _visible(lm, cfg.NOSE):
            lean = (lm[cfg.NOSE].x - mid_x) - baseline.get("lean_offset", 0)
            if abs(lean) > cfg.FORWARD_LEAN_THRESHOLD:
                issues.append(f"Leaning {'right' if lean > 0 else 'left'} ({abs(lean):.3f})")

    # 6. Head tilt
    if _visible(lm, cfg.LEFT_EAR, cfg.RIGHT_EAR):
        le = pt(lm, cfg.LEFT_EAR, w, h)
        re = pt(lm, cfg.RIGHT_EAR, w, h)
        tilt  = _angle_from_horizontal(le, re)
        delta = tilt - baseline.get("head_tilt", 0)
        if delta > cfg.HEAD_TILT_THRESHOLD:
            issues.append(f"Head tilted ({delta:.1f}°)")

    return (False, " | ".join(issues)) if issues else (True, "Good posture")

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
