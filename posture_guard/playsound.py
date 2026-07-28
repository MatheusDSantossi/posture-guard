
"""
Cross-platform sound playback.
Resolves the asset path whether running from source or a PyInstaller bundle.
"""
import os
import sys
import platform

def _get_sound_path() -> str:
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # posture_guard\ui\assets\audio\evolved-into-a-shrimp-cutted.wav
        
    return os.path.join(base, "posture_guard", "ui", "assets", "audio", "evolved-into-a-shrimp-cutted.wav")

def play_sound():
    
    path = _get_sound_path()
    
    if not os.path.exists(path):
        print(f"[WARN] Sound file not found: {path}")
        
    try:
        os_name = platform.system()
        
        if os_name == "Windows":
            import winsound
            # No SND_ASYNC, the daemon thread is already the async mechanism
            # SND_ASYNC on a daemon thread kills the sond mid-play.
            winsound.PlaySound(
                path,
                winsound.SND_FILENAME
            )
            
        elif os_name == "Darwin":
            import subprocess
            subprocess.run(["afplay", path], timeout=10)
            
        elif os_name == "Linux":
            import subprocess
            subprocess.run(["aplay", path], timeout=10, capture_output=True)
        
    except Exception as e:
        print(f"[WARN] Sound playback failed: {e}")
