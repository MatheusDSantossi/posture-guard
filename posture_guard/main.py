"""
PyInstaller entry point.
Kept separate from main.py so the spec file has a clean top-level target.
When running from a bundled .exe, sys._MEIPASS holds the temp extraction path
where PyInstaller unpacks data files — we use that to locate the model.
"""
import os
import sys

# when frozen by PyInstaller, data files land in sys._MEIPASS
if getattr(sys, "frozen", False):
    _base = sys._MEIPASS
    
else:
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
# _base = Path(__file__).resolve().parent
# model_path = _base / "pose_landmarker_full.task"

# os.environ.setdefault("PG_MODEL_PATH", str(model_path))    

# Patch MODEL_PATH before config is imported by anything else
os.environ.setdefault("PG_MODEL_PATH", os.path.join(_base, "pose_landmarker_full.task"))
# os.environ.setdefault("PG_MODEL_PATH", os.path.json(_base, "pose_landmarker_full.task"))

def main():
    # Imported here so PG_MODEL_PATH above is set before posture_guard.config reads it.
    # The `posture-guard` console script imports this module and then calls main(),
    # so nothing may run at import time or the app would start twice.
    from posture_guard.app import main as run_app

    run_app()


if __name__ == "__main__":
    main()