"""
Central configuration — all tuneable constants live here.
Users can override via environment variables.
"""
import os

def _float(key, default): return float(os.environ.get(key, default))

def _int(key, default): return int(os.environ.get(key, default))

def _bool(key, default): return os.environ.get(key, str(default)).lower() in ("1", "true")

# Model
MODEL_PATH = os.environ.get("PG_MODEL_PATH", "pose_landmarker_full.task")

# Detection thresholds (all deltas relative to calibrated baseline)
SHOULDER_DROP_THRESHOLD   = _float("PG_SHOULDER_DROP",   0.04)
SHOULDER_WIDTH_THRESHOLD  = _float("PG_SHOULDER_WIDTH",  0.06)
SHOULDER_TILT_THRESHOLD   = _float("PG_SHOULDER_TILT",   8.0)
HEAD_RISE_THRESHOLD       = _float("PG_HEAD_RISE",        0.05)
FORWARD_LEAN_THRESHOLD    = _float("PG_FORWARD_LEAN",     0.04)
HEAD_TILT_THRESHOLD       = _float("PG_HEAD_TILT",        10.0)
VISIBILITY_MIN            = _float("PG_VISIBILITY_MIN",   0.5)

# Timing
BAD_POSTURE_DURATION      = _float("PG_BAD_DURATION",    2.0)   # seconds sustained before alert
ALERT_COOLDOWN_SECONDS    = _float("PG_ALERT_COOLDOWN",  30.0)
CALIBRATION_FRAMES        = _int("PG_CALIB_FRAMES",      90)    # ~3s at 30fps
LOOP_SLEEP_SECONDS        = _float("PG_LOOP_SLEEP",      0.067) # ~15fps — saves CPU

# Landmark indices (MediaPipe Pose)
NOSE           = 0
LEFT_EAR       = 7
RIGHT_EAR      = 8
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (7, 8),
]

# Misc
DEBUG = _bool("DEBUG", False)
