import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import numpy as np
import time

base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

# ── Thresholds ────────────────────────────────────────────────────────────────
SHOULDER_TILT_THRESHOLD   = 8      # degrees above baseline
FORWARD_LEAN_THRESHOLD    = 0.04   # normalized X drift above baseline
NECK_FORWARD_THRESHOLD    = 0.08   # normalized Y: nose too far below shoulders
HEAD_TILT_THRESHOLD       = 10     # degrees: ear-to-ear line vs horizontal
VISIBILITY_MIN            = 0.6    # skip landmark if confidence below this
BAD_POSTURE_DURATION      = 1.5    # seconds of sustained bad posture before alert
ALERT_COOLDOWN_SECONDS    = 8
CALIBRATION_FRAMES        = 60     # ~2 seconds at 30fps

# ── Landmark indices (MediaPipe Pose) ─────────────────────────────────────────
NOSE          = 0
LEFT_EAR      = 7
RIGHT_EAR     = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER= 12

CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (7, 8),   # ears
]


def visible(lm, idx):
    return lm[idx].visibility >= VISIBILITY_MIN


def angle_from_horizontal(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    return 180 - angle if angle > 90 else angle


def collect_baseline(lm, w, h):
    """Return a snapshot of neutral-posture metrics."""
    def pt(idx):
        p = lm[idx]
        return p.x * w, p.y * h

    baseline = {}

    if visible(lm, LEFT_SHOULDER) and visible(lm, RIGHT_SHOULDER):
        baseline["shoulder_tilt"] = angle_from_horizontal(pt(LEFT_SHOULDER), pt(RIGHT_SHOULDER))
        mid_x = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2
        baseline["lean_offset"] = lm[NOSE].x - mid_x
        baseline["neck_y_offset"] = lm[NOSE].y - lm[LEFT_SHOULDER].y  # nose relative to shoulder height

    if visible(lm, LEFT_EAR) and visible(lm, RIGHT_EAR):
        baseline["head_tilt"] = angle_from_horizontal(pt(LEFT_EAR), pt(RIGHT_EAR))

    return baseline


def check_posture(lm, w, h, baseline):
    """Returns list of (is_bad, reason) for each check."""
    issues = []

    def pt(idx):
        p = lm[idx]
        return p.x * w, p.y * h

    # 1. Shoulder tilt
    if visible(lm, LEFT_SHOULDER) and visible(lm, RIGHT_SHOULDER):
        tilt = angle_from_horizontal(pt(LEFT_SHOULDER), pt(RIGHT_SHOULDER))
        delta = tilt - baseline.get("shoulder_tilt", 0)
        if delta > SHOULDER_TILT_THRESHOLD:
            issues.append(f"Shoulders uneven (+{delta:.1f}°)")

    # 2. Lateral lean (nose vs shoulder midpoint on X axis)
    if visible(lm, NOSE) and visible(lm, LEFT_SHOULDER) and visible(lm, RIGHT_SHOULDER):
        mid_x = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2
        lean = (lm[NOSE].x - mid_x) - baseline.get("lean_offset", 0)
        if abs(lean) > FORWARD_LEAN_THRESHOLD:
            direction = "right" if lean > 0 else "left"
            issues.append(f"Leaning {direction} ({abs(lean):.3f})")

    # 3. Neck forward nose dropping too far below shoulder level
    if visible(lm, NOSE) and visible(lm, LEFT_SHOULDER):
        neck_y = lm[NOSE].y - lm[LEFT_SHOULDER].y
        delta_y = neck_y - baseline.get("neck_y_offset", 0)
        if delta_y > NECK_FORWARD_THRESHOLD:
            issues.append(f"Head jutting forward ({delta_y:.3f})")

    # 4. Head tilt (ear-to-ear)
    if visible(lm, LEFT_EAR) and visible(lm, RIGHT_EAR):
        tilt = angle_from_horizontal(pt(LEFT_EAR), pt(RIGHT_EAR))
        delta = tilt - baseline.get("head_tilt", 0)
        if delta > HEAD_TILT_THRESHOLD:
            issues.append(f"Head tilted ({delta:.1f}°)")

    if issues:
        return False, " | ".join(issues)
    return True, "Good posture"


def draw_landmarks(frame, lm, w, h):
    def pt(idx):
        p = lm[idx]
        return int(p.x * w), int(p.y * h)

    for a, b in CONNECTIONS:
        try:
            if visible(lm, a) and visible(lm, b):
                cv2.line(frame, pt(a), pt(b), (245, 66, 230), 2)
        except IndexError:
            pass

    for i in range(len(lm)):
        try:
            if lm[i].visibility >= VISIBILITY_MIN:
                cv2.circle(frame, pt(i), 4, (245, 117, 66), -1)
        except IndexError:
            pass


def draw_status(frame, is_good, reason, bad_since, now):
    color = (0, 200, 0) if is_good else (0, 0, 220)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    label = f"{'✓' if is_good else '✗'} {reason}"
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Progress bar showing how long bad posture has been sustained
    if not is_good and bad_since is not None:
        elapsed = min(now - bad_since, BAD_POSTURE_DURATION)
        ratio = elapsed / BAD_POSTURE_DURATION
        bar_w = int(frame.shape[1] * ratio)
        cv2.rectangle(frame, (0, 44), (bar_w, 50), (0, 80, 220), -1)


def draw_calibration_overlay(frame, collected, total):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    ratio = collected / total
    bar_w = int(frame.shape[1] * ratio)
    cv2.rectangle(frame, (0, frame.shape[0] - 20), (bar_w, frame.shape[0]), (0, 200, 100), -1)
    msg = "Calibrating sit straight and look forward"
    cv2.putText(frame, msg, (20, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    pct = f"{int(ratio * 100)}%"
    cv2.putText(frame, pct, (20, frame.shape[0] // 2 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 100), 2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    last_alert_time = 0
    bad_posture_start = None
    baseline = {}
    calibration_samples = []

    print("PostureGuard running calibrating, sit straight!")

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp)

            # ── Calibration phase ──────────────────────────────────────────
            if len(calibration_samples) < CALIBRATION_FRAMES:
                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    sample = collect_baseline(lm, w, h)
                    if sample:
                        calibration_samples.append(sample)
                draw_calibration_overlay(frame, len(calibration_samples), CALIBRATION_FRAMES)
                cv2.imshow("PostureGuard", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Build averaged baseline once
            if not baseline and calibration_samples:
                keys = calibration_samples[0].keys()
                baseline = {k: np.mean([s[k] for s in calibration_samples if k in s]) for k in keys}
                print(f"[INFO] Baseline calibrated: {baseline}")

            # ── Detection phase ────────────────────────────────────────────
            now = time.time()

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                draw_landmarks(frame, lm, w, h)
                is_good, reason = check_posture(lm, w, h, baseline)
                draw_status(frame, is_good, reason, bad_posture_start, now)

                if not is_good:
                    if bad_posture_start is None:
                        bad_posture_start = now
                    elif (now - bad_posture_start) >= BAD_POSTURE_DURATION:
                        if (now - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                            print(f"[ALERT] {reason}")
                            last_alert_time = now
                else:
                    bad_posture_start = None
            else:
                cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
                cv2.putText(frame, "No person detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 0), 2)
                bad_posture_start = None

            cv2.imshow("PostureGuard", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("PostureGuard stopped.")


if __name__ == "__main__":
    main()