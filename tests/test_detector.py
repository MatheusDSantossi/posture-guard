"""
Unit tests for the pure geometry helpers in detector.py.
These don't touch the camera, MediaPipe, or settings I/O — just math.
"""
import math
from posture_guard.detector import _angle_from_horizontal, pt_mid

def test_angle_from_horizontal_level_line_is_zero():
    p1 = (0.0, 100.0)
    p2 = (100.0, 100.0)
    assert _angle_from_horizontal(p1, p2) == 0.0
    
def test_angle_from_horizontal_vertical_line_is_ninety():
    p1 = (50.0, 0.0)
    p2 = (50.0, 100.0)
    assert math.isclose(_angle_from_horizontal(p1, p2), 90.0, abs_tol=1e-6)
    
def test_angle_from_horizontal_is_direction_independent():
    p1 = (0.0, 100.0)
    p2 = (100.0, 105.0)
    assert math.isclose(
        _angle_from_horizontal(p1, p2),
        _angle_from_horizontal(p2, p1),
        abs_tol=1e-6
    )
    
class _FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def test_pt_mid_averages_two_landmarks():
    lm = {0: _FakeLandmark(0.2, 0.4), 1: _FakeLandmark(0.4, 0.6)}
    w, h = 200, 100
    x, y = pt_mid(lm, 0, 1, w, h)
    assert math.isclose(x, 0.3 * w, abs_tol=1e-6)
    assert math.isclose(y, 0.5 * h, abs_tol=1e-6)
