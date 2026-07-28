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
from posture_guard import settings
from posture_guard.calibration import run_calibration
from posture_guard.detector import check_posture, draw_debug_overlay, reset_history
from posture_guard.notifier import notify
from posture_guard.tray import TrayIcon
from posture_guard.ui import settings_ui
from posture_guard.extreme_alert import trigger as extreme_trigger


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

    user_settings = settings.load()

    def _on_settings_saved(new_settings):
        nonlocal user_settings
        user_settings = new_settings
        print(f"[INFO] Settings reloaded: {new_settings}")

    tray = TrayIcon(state, on_settings=lambda: settings_ui.open_settings(_on_settings_saved))
    tray.start()

    cap = _open_camera()

    print(f"PostureGuard v{__version__} starting. Set DEBUG=1 to show camera window.")

    with _build_landmarker() as landmarker:
        baseline        = run_calibration(cap, landmarker, state)
        last_alert_time = 0
        bad_start       = None
        # calibration already consumed timestamps 0..33*CALIBRATION_FRAMES
        # on the same landmarker, so start the main loop above that range
        timestamp_ms    = cfg.CALIBRATION_FRAMES * 33 + 1000

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
                reset_history()
                with _build_landmarker() as calib_landmarker:
                    baseline = run_calibration(cap, calib_landmarker, state)
                bad_start    = None
                continue

            # ── Pause — release camera while idle ──────────────────────────
            if state["paused"]:
                _release_camera(cap)
                while state["paused"] and state["running"]:  # block here
                    time.sleep(0.2)
                # ← outside the loop: runs once when resumed or quit
                if state["running"]:
                    cap       = _open_camera()
                    bad_start = None
                    reset_history()
                continue

            # ── Capture ────────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms += int(cfg.LOOP_SLEEP_SECONDS * 1000) + 1
            result = landmarker.detect_for_video(mp_img, timestamp_ms)
            now    = time.time()

            # Reload settings each frame so changes apply immediately
            user_settings   = settings.load()
            bad_duration    = user_settings.get("bad_posture_duration", cfg.BAD_POSTURE_DURATION)
            alert_cooldown  = user_settings.get("alert_cooldown",       cfg.ALERT_COOLDOWN_SECONDS)
            extreme_enabled = user_settings.get("extreme_alert",        False)
            extreme_sound   = user_settings.get("extreme_sound",        True)
            extreme_flash   = user_settings.get("extreme_flash",        True)

            # ── Detect ─────────────────────────────────────────────────────
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                is_good, reason = check_posture(lm, w, h, baseline)

                if not is_good:
                    tray.set_bad()
                    if bad_start is None:
                        bad_start = now
                    elif (now - bad_start) >= bad_duration:
                        if (now - last_alert_time) >= alert_cooldown:
                            notify("PostureGuard ⚠️", reason, tray=tray._icon, play_sound=extreme_enabled and extreme_sound)
                            print(f"[ALERT] {reason}")
                            last_alert_time = now
                            if extreme_flash:
                                extreme_trigger(sound=False, flash=True)
                else:
                    tray.set_good()
                    bad_start = None

                if True:
                # if cfg.DEBUG:
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