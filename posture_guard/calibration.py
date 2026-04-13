"""
Calibration window — shown on startup and on recalibrate requests.
Captures ~3 seconds of "good posture" frames and returns a baseline dict.
"""
import cv2
import mediapipe as mp
import time

from . import config as cfg
from .detector import collect_sample, build_baseline

def run_calibration(cap, landmarker, state: dict) -> dict:
    samples = []
    win = "PostureGuard — Calibrating"
    print("[INFO] Calibration started — sit straight and look forward")

    while len(samples) < cfg.CALIBRATION_FRAMES:
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
        ratio = len(samples) / cfg.CALIBRATION_FRAMES
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

def _draw_calibration(frame, w: int, h: int, collected: int):
    overlay = frame.copty()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    
    ratio = collected / cfg.CALIBRATION_FRAMES
    bar_w = int((w - 80) * ratio)
    cv2.rectangle(frame, (40, h // 2 + 30), (40 + bar_w, h // 2 + 52), (0, 200, 100), -1)
    cv2.putText(frame, "Sit straight and look forawrd", (40, h // 2 + 30), (40 + bar_w, h // 2 + 52), (0, 200, 100), -1)
    cv2.putText(frame, "Sit straight and look forward", (40, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
