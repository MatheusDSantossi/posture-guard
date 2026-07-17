"""
Extreme alert — plays a sound and/or flashes the screen.
Sound uses the built-in winsound on Windows, afplay on macOS, aplay on Linux.
Screen flash uses a temporary fullscreen tkinter window.
"""
import platform
import threading

def trigger(sound: bool = True, flash: bool = True):
    """Fire extreme alert in a background thread so it never blocks detection."""
    if sound:
        threading.Thread(target=_play_sound, daemon=True).start()
    if flash:
        threading.Thread(target=_flash_screen, daemon=True).start()
        
def _play_sound():
    os_name = platform.system()
    try:
        if os_name == "Windows":
            import winsound
            # MB_ICONHAND is the system "critical stop" sound — punchy and hard to ignore
            winsound.MessageBeep(winsound.MB_ICONHAND)
            
        elif os_name == "Darwin":
            import subprocess
            subprocess.run(["afplay", "/System/Library/Sounds/Basso.aiff"], timeout=5)
            
        elif os_name == "Linux":
            import subprocess
            # Most Linux distros have this; falls back silently if not
            subprocess.run(
                ["aplay", "/usr/share/sounds/alsa/Front_Left.wav"],
                timeout=5, capture_output=True,
            )
    except Exception as e:
        print(f"[WARN] Sound alert failed: {e}")

def _flash_screen():
    """Show a bright red fullscreen overlay for ~400ms."""
    try:
        import tkinter as tk
        
        root = tk.Tk()
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", True)
        root.configure(bg="red")
        root.overrideredirect(True)
        
        # Auto-close after 400ms
        root.after(400, root.destroy)
        root.mainloop()
        
    except Exception as e:
        print(f"[WARN] Flash alert failed: {e}")        
