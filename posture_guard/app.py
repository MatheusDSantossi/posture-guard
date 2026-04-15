"""
PostureGuard — main entry point.
Runs the detection loop; UI is headless unless DEBUG=1.
"""
import sys
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

from posture_guard import __version__
from posture_guard import config as cfg
from posture_guard.calibration import run_calibration
from posture_guard.detector import check_posture, draw_debug_overlay
from posture_guard.notifier import notify
from posture_guard.tray import TrayIcon


def _build_landmarker():
    base_options = python.BaseOptions(model_asset_path=cfg.MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        sys.exit(1)
    return cap


def _release_camera(cap: cv2.VideoCapture):
    if cap and cap.isOpened():
        cap.release()


def main():
    print(f"PostureGuard v{__version__} starting...")

    state = {"running": True, "paused": False, "recalibrate": False}
    tray  = TrayIcon(state)
    tray.start()

    cap = _open_camera()

    print("PostureGuard starting. Set DEBUG=1 to show camera window.")

    with _build_landmarker() as landmarker:
        baseline        = run_calibration(cap, landmarker, state)
        
    with _build_landmarker() as landmarker:
        # baseline        = run_calibration(cap, landmarker, state)
        last_alert_time = 0
        bad_start       = None
        start_time = time.monotonic()
        last_timestamp = -1

        print("PostureGuard running in background.")
        if cfg.DEBUG:
            print("[DEBUG] Camera window active — press Q to quit")

        while state["running"]:

            # ── Recalibrate ────────────────────────────────────────────────
            if state["recalibrate"]:
                state["recalibrate"] = False
                if not cap.isOpened():
                    cap = _open_camera()
                tray.set_good()
                # Destroy current landmarker
                landmarker.close()

                # Create a fresh one JUST for calibration
                with _build_landmarker() as calib_landmarker:
                    baseline = run_calibration(cap, calib_landmarker, state)

                # Create a fresh one for main loop again
                landmarker = _build_landmarker()

                # Reset timestamp timeline
                start_time = time.monotonic()
                last_timestamp = -1
                bad_start = None
                continue

            # ── Pause — release camera while idle ──────────────────────────
            if state["paused"]:
                _release_camera(cap)
                while state["paused"] and state["running"]:
                    time.sleep(0.2)
                if state["running"]:
                    cap       = _open_camera()
                    bad_start = None
                continue

            # ── Capture ────────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            current_ts = int((time.monotonic() - start_time) * 1000)
            timestamp_ms = max(current_ts, last_timestamp + 1)
            last_timestamp = timestamp_ms
          
            result = landmarker.detect_for_video(mp_img, timestamp_ms)
            now    = time.time()

            # ── Detect ─────────────────────────────────────────────────────
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                is_good, reason = check_posture(lm, w, h, baseline)

                if not is_good:
                    tray.set_bad()
                    if bad_start is None:
                        bad_start = now
                    elif (now - bad_start) >= cfg.BAD_POSTURE_DURATION:
                        if (now - last_alert_time) >= cfg.ALERT_COOLDOWN_SECONDS:
                            notify("PostureGuard ⚠️", reason, tray=tray._icon)
                            print(f"[ALERT] {reason}")
                            last_alert_time = now
                else:
                    tray.set_good()
                    bad_start = None

                if cfg.DEBUG:
                    draw_debug_overlay(frame, lm, w, h, baseline)
                    color = (0, 200, 0) if is_good else (0, 0, 220)
                    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
                    cv2.putText(frame, f"{'OK' if is_good else 'FIX'}: {reason}",
                                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
            else:
                bad_start = None

            if cfg.DEBUG:
                cv2.imshow("PostureGuard [DEBUG]", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(cfg.LOOP_SLEEP_SECONDS)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    _release_camera(cap)
    if cfg.DEBUG:
        cv2.destroyAllWindows()
    tray.stop()
    print("PostureGuard stopped.")


if __name__ == "__main__":
    main()
    