"""
Cross-platform OS notifications.
Falls back to a terminal print if no native method is available.
"""
import platform
import subprocess
import threading

def notify(title: str, message: str, tray=None, play_sound: bool = False):
    """Send a native OS notification."""
    os_name = platform.system()
 
    if os_name == "Darwin":
        _notify_macos(title, message, play_sound)
    elif os_name == "Linux":
        _notify_linux(title, message, play_sound)
    elif os_name == "Windows":
        _notify_windows(title, message, tray, play_sound)
    else:
        print(f"[NOTIFY] {title}: {message}")
 
def _notify_macos(title: str, message: str, play_sound: bool):
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{message}" with title "{title}"'], check=False, timeout=5
        ) 
        
    except Exception as e:
        print(f"[NOTIFY] macOS notification failed: {e}")
        print(f"[NOTIFY] {title}: {message}")

    if play_sound:
        _fire_sound()

def _notify_linux(title: str, message: str, play_sound: bool):
    try:
        subprocess.run(["notify-send", title, message], check=False, timeout=5)
    except FileNotFoundError:
        print(f"[NOTIFY] notify-send not found. Install libnotify-bin.")
        print(f"[NOTIFY] {title}: {message}")
    except Exception as e:
        print(f"[NOTIFY] Linux notification failed: {e}")
        print(f"[NOTIFY] {title}: {message}")
        
    if play_sound:
        _fire_sound()

def _notify_windows(title: str, message: str, tray=None, play_sound: bool = False):
    # Try winotify first (best Windows experience)
    try:
        from winotify import Notification, audio
        
        toast = Notification(
            app_id="PostureGuard",
            title=title,
            msg=message,
        )
        toast.set_audio(audio.Silent, loop=False)
        toast.show()
    except Exception:
        # Fallback: pystray bubble
        if tray is not None:
            try:
                tray.notify(message, title)
            except Exception:
                pass
    
    # Fallback: pystray buble
    if play_sound:
        _fire_sound()
    
def _fire_sound():
    """Play the alert sound in a daemon thread so it never blocks detection"""
    from posture_guard.playsound import play_sound
    threading.Thread(target=play_sound, daemon=True).start()
