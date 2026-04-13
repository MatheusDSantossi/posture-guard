import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import numpy as np
import time
import threading
import sys
import platform

# ── Optional deps ──────────────────────────────────────────────────────────────
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False

# ── MediaPipe setup ────────────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

# ── Thresholds (all deltas relative to calibrated baseline) ───────────────────
SHOULDER_DROP_THRESHOLD    = 0.04   # shoulders falling down in frame (Y normalized)
SHOULDER_WIDTH_THRESHOLD   = 0.06   # shoulders getting narrower (slouch forward)
SHOULDER_TILT_THRESHOLD    = 8      # degrees — one shoulder higher than other
HEAD_RISE_THRESHOLD        = 0.05   # head-to-shoulder Y gap increasing (head jutting up/shoulders dropping)
FORWARD_LEAN_THRESHOLD     = 0.04   # lateral drift of nose vs shoulder midpoint
HEAD_TILT_THRESHOLD        = 10     # degrees — ear-to-ear tilt
VISIBILITY_MIN             = 0.5
BAD_POSTURE_DURATION       = 2.0
ALERT_COOLDOWN_SECONDS     = 30
CALIBRATION_FRAMES         = 90     # ~3 seconds

# ── Landmark indices ───────────────────────────────────────────────────────────
NOSE           = 0
LEFT_EAR       = 7
RIGHT_EAR      = 8
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (7, 8),
]

# ── Shared state ───────────────────────────────────────────────────────────────
state = {"running": True, "paused": False, "recalibrate": False, "tray": None}


# ── Helpers ────────────────────────────────────────────────────────────────────
def visible(lm, *idxs):
    return all(lm[i].visibility >= VISIBILITY_MIN for i in idxs)


def angle_from_horizontal(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    a = abs(np.degrees(np.arctan2(dy, dx)))
    return 180 - a if a > 90 else a


def pt(lm, idx, w, h):
    return lm[idx].x * w, lm[idx].y * h


# ── Calibration ────────────────────────────────────────────────────────────────
def collect_sample(lm, w, h):
    sample = {}
    if visible(lm, LEFT_SHOULDER, RIGHT_SHOULDER):
        ls = pt(lm, LEFT_SHOULDER, w, h)
        rs = pt(lm, RIGHT_SHOULDER, w, h)
        mid_y = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2
        mid_x = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2
        width  = abs(lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x)
        tilt   = angle_from_horizontal(ls, rs)

        sample["shoulder_mid_y"] = mid_y          # normalized Y of shoulder midpoint
        sample["shoulder_width"] = width          # normalized X distance between shoulders
        sample["shoulder_tilt"]  = tilt           # angle
        sample["lean_offset"]    = lm[NOSE].x - mid_x
        sample["neck_y_gap"]     = lm[NOSE].y - lm[LEFT_SHOULDER].y  # nose above/below shoulder

    if visible(lm, LEFT_EAR, RIGHT_EAR):
        le = pt(lm, LEFT_EAR, w, h)
        re = pt(lm, RIGHT_EAR, w, h)
        sample["head_tilt"] = angle_from_horizontal(le, re)

    return sample


def build_baseline(samples):
    keys = {k for s in samples for k in s}
    return {k: float(np.mean([s[k] for s in samples if k in s])) for k in keys}


def run_calibration(cap, landmarker):
    samples = []
    win = "PostureGuard — Calibrating"
    print("[INFO] Calibration started — sit straight and look forward")

    while len(samples) < CALIBRATION_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        if result.pose_landmarks:
            s = collect_sample(result.pose_landmarks[0], w, h)
            if s:
                samples.append(s)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        ratio = len(samples) / CALIBRATION_FRAMES
        bar_w = int((w - 80) * ratio)
        cv2.rectangle(frame, (40, h // 2 + 30), (40 + bar_w, h // 2 + 52), (0, 200, 100), -1)
        cv2.putText(frame, "Sit straight and look forward",
                    (40, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Calibrating... {int(ratio * 100)}%",
                    (40, h // 2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 100), 2)
        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            state["running"] = False
            break

    cv2.destroyWindow(win)
    baseline = build_baseline(samples) if samples else {}
    print(f"[INFO] Baseline: {baseline}")
    return baseline


# ── Posture checks ─────────────────────────────────────────────────────────────
def check_posture(lm, w, h, baseline):
    issues = []

    if visible(lm, LEFT_SHOULDER, RIGHT_SHOULDER):
        ls = pt(lm, LEFT_SHOULDER, w, h)
        rs = pt(lm, RIGHT_SHOULDER, w, h)
        mid_y = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2
        mid_x = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2
        width  = abs(lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x)
        tilt   = angle_from_horizontal(ls, rs)

        # 1. Shoulders dropped (Y increased = lower in frame for most cameras)
        drop = mid_y - baseline.get("shoulder_mid_y", mid_y)
        if drop > SHOULDER_DROP_THRESHOLD:
            issues.append(f"Shoulders dropped ({drop:.3f})")

        # 2. Shoulders narrowed = hunching forward
        width_loss = baseline.get("shoulder_width", width) - width
        if width_loss > SHOULDER_WIDTH_THRESHOLD:
            issues.append(f"Hunching forward — shoulders narrowed ({width_loss:.3f})")

        # 3. Shoulder tilt — one side higher
        tilt_delta = tilt - baseline.get("shoulder_tilt", 0)
        if tilt_delta > SHOULDER_TILT_THRESHOLD:
            issues.append(f"Shoulders uneven (+{tilt_delta:.1f}°)")

        # 4. Head-to-shoulder gap changed (shoulders dropped away from head)
        if visible(lm, NOSE):
            neck_gap   = lm[NOSE].y - lm[LEFT_SHOULDER].y
            gap_delta  = neck_gap - baseline.get("neck_y_gap", neck_gap)
            if gap_delta > HEAD_RISE_THRESHOLD:
                issues.append(f"Shoulders dropped from head ({gap_delta:.3f})")

        # 5. Lateral lean
        if visible(lm, NOSE):
            lean = (lm[NOSE].x - mid_x) - baseline.get("lean_offset", 0)
            if abs(lean) > FORWARD_LEAN_THRESHOLD:
                issues.append(f"Leaning {'right' if lean > 0 else 'left'} ({abs(lean):.3f})")

    # 6. Head tilt
    if visible(lm, LEFT_EAR, RIGHT_EAR):
        le = pt(lm, LEFT_EAR, w, h)
        re = pt(lm, RIGHT_EAR, w, h)
        tilt  = angle_from_horizontal(le, re)
        delta = tilt - baseline.get("head_tilt", 0)
        if delta > HEAD_TILT_THRESHOLD:
            issues.append(f"Head tilted ({delta:.1f}°)")

    return (False, " | ".join(issues)) if issues else (True, "Good posture")


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_landmarks(frame, lm, w, h):
    def p(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    for a, b in CONNECTIONS:
        try:
            if visible(lm, a, b):
                cv2.line(frame, p(a), p(b), (245, 66, 230), 2)
        except IndexError:
            pass
    for i in range(len(lm)):
        try:
            if lm[i].visibility >= VISIBILITY_MIN:
                cv2.circle(frame, p(i), 4, (245, 117, 66), -1)
        except IndexError:
            pass


def draw_debug(frame, lm, w, h, baseline):
    """Show live metric values vs baseline for tuning."""
    if not visible(lm, LEFT_SHOULDER, RIGHT_SHOULDER):
        return

    mid_y  = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2
    width  = abs(lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x)
    neck_y = lm[NOSE].y - lm[LEFT_SHOULDER].y if visible(lm, NOSE) else 0

    lines = [
        f"shld_drop:  {mid_y - baseline.get('shoulder_mid_y', mid_y):+.3f}  (>{SHOULDER_DROP_THRESHOLD})",
        f"shld_width: {baseline.get('shoulder_width', width) - width:+.3f}  (>{SHOULDER_WIDTH_THRESHOLD})",
        f"neck_gap:   {neck_y - baseline.get('neck_y_gap', neck_y):+.3f}  (>{HEAD_RISE_THRESHOLD})",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, h - 20 - i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)


# ── Notifications ──────────────────────────────────────────────────────────────
def _notify_os(title, message):
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            import subprocess
            subprocess.run(["osascript", "-e",
                f'display notification "{message}" with title "{title}"'], check=False)
        elif os_name == "Linux":
            import subprocess
            subprocess.run(["notify-send", title, message], check=False)
        elif os_name == "Windows":
            tray = state.get("tray")
            if tray:
                tray.notify(message, title)
    except Exception:
        pass
    print(f"[ALERT] {title}: {message}")


# ── Tray ───────────────────────────────────────────────────────────────────────
def _make_icon(color=(0, 180, 0)):
    img = Image.new("RGB", (64, 64), (30, 30, 30))
    ImageDraw.Draw(img).ellipse([8, 8, 56, 56], fill=color)
    return img


def _on_pause(icon, item):
    state["paused"] = not state["paused"]
    icon.icon = _make_icon((200, 140, 0) if state["paused"] else (0, 180, 0))


def _on_recalibrate(icon, item):
    state["recalibrate"] = True


def _on_quit(icon, item):
    state["running"] = False
    icon.stop()


def start_tray():
    if not HAS_TRAY:
        return
    icon = pystray.Icon("PostureGuard", _make_icon(), "PostureGuard",
        menu=pystray.Menu(
            pystray.MenuItem("Pause / Resume", _on_pause),
            pystray.MenuItem("Recalibrate",    _on_recalibrate),
            pystray.MenuItem("Quit",           _on_quit),
        ))
    state["tray"] = icon
    threading.Thread(target=icon.run, daemon=True).start()


def tray_color(good):
    t = state.get("tray")
    if t:
        t.icon = _make_icon((0, 180, 0) if good else (220, 0, 0))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Set DEBUG=1 to show the camera window with live metric overlay
    import os
    DEBUG = os.environ.get("DEBUG", "0") == "1"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        sys.exit(1)

    start_tray()
    print("PostureGuard starting. Set DEBUG=1 to show camera window.")

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        baseline        = run_calibration(cap, landmarker)
        last_alert_time = 0
        bad_start       = None

        while state["running"]:
            if state["recalibrate"]:
                state["recalibrate"] = False
                baseline = run_calibration(cap, landmarker)
                bad_start = None
                continue

            if state["paused"]:
                time.sleep(0.2)
                continue

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))
            now    = time.time()

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                is_good, reason = check_posture(lm, w, h, baseline)
                tray_color(is_good)

                if not is_good:
                    if bad_start is None:
                        bad_start = now
                    elif (now - bad_start) >= BAD_POSTURE_DURATION:
                        if (now - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                            _notify_os("PostureGuard ⚠️", reason)
                            last_alert_time = now
                else:
                    bad_start = None

                if DEBUG:
                    draw_landmarks(frame, lm, w, h)
                    draw_debug(frame, lm, w, h, baseline)
                    color = (0, 200, 0) if is_good else (0, 0, 220)
                    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
                    cv2.putText(frame, f"{'OK' if is_good else 'FIX'}: {reason}",
                                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
            else:
                bad_start = None

            if DEBUG:
                cv2.imshow("PostureGuard [DEBUG]", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.067)

    cap.release()
    if DEBUG:
        cv2.destroyAllWindows()
    print("PostureGuard stopped.")


if __name__ == "__main__":
    main()