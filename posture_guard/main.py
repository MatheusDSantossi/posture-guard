"""
PyInstaller entry point.
Kept separate from main.py so the spec file has a clean top-level target.
When running from a bundled .exe, sys._MEIPASS holds the temp extraction path
where PyInstaller unpacks data files — we use that to locate the model.
"""
import sys
import os

# when frozen by PyInstaller, data files land in sys._MEIPASS
if getattr(sys, "frozen", False):
    _base = sys._MEIPASS
else:
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
# Patch MODEL_PATH before config is imported by anything else
os.environ.setdefault("PG_MODEL_PATH", os.path.json(_base, "pose_landmarker_full.task"))

from posture_guard.main import main # noqa: E402

main()