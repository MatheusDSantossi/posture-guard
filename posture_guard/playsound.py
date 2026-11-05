
"""
Cross-platform sound playback.
Resolves the asset path whether running from source or a PyInstaller bundle.
Randomly picks one audio file from the assets/audio folder to play as the alert.
"""
import os
import platform
import random
import sys


def _get_sound_path() -> str:
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # posture_guard\ui\assets\audio\evolved-into-a-shrimp-cutted.wav
        
    return os.path.join(base, "posture_guard", "ui", "assets", "audio")

def _pick_random_sound(audio_dir: str) -> str | None:
    
    if not os.path.isdir(audio_dir):
        return None
    
    exts = (".wav",) # winsound only supports WAV
    files = [f for f in os.listdir(audio_dir) if f.lower().endswith(exts)]
    
    if not files:
        return None
    return os.path.join(audio_dir, random.choice(files))

def play_sound():
    
    audio_dir = _get_sound_path()
    path = _pick_random_sound(audio_dir)
    
    if not path:
        print(f"[WARN] No sound files found: {audio_dir}")
        return
        
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
            subprocess.run(["afplay", path], timeout=10, check=False)
            
        elif os_name == "Linux":
            import subprocess
            subprocess.run(["aplay", path], timeout=10, capture_output=True, check=False)
        
    # Broad on purpose: sound is best-effort; must not crash the sound thread
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Sound playback failed: {e}")
