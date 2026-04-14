"""
Cross-platform OS notifications.
Falls back to a terminal print if no native method is available.
"""
import platform
import subprocess
from winotify import Notification, audio
 
def notify(title: str, message: str, tray=None):
    """Send a native OS notification."""
    os_name = platform.system()
 
    if os_name == "Darwin":
        _notify_macos(title, message)
    elif os_name == "Linux":
        _notify_linux(title, message)
    elif os_name == "Windows":
        _notify_windows(title, message, tray)
    else:
        toast = Notification(
            app_id="PostureGuard",
            title=title,
            msg=message,
        )
        toast.set_audio(audio.Default, loop=False)
        toast.show()
        print(f"[NOTIFY] {title}: {message}")
 
def _notify_macos(title: str, message: str):
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{message}" with title "{title}"'], check=False, timeout=5
        ) 
        
    except Exception as e:
        print(f"[NOTIFY] macOS notification failed: {e}")
        print(f"[NOTIFY] {title}: {message}")

def _notify_linux(title: str, message: str):
    try:
        subprocess.run(["notify-send", title, message], check=False, timeout=5)
    except FileNotFoundError:
        print(f"[NOTIFY] notify-send not found. Install libnotify-bin.")
        print(f"[NOTIFY] {title}: {message}")
    except Exception as e:
        print(f"[NOTIFY] Linux notification failed: {e}")
        print(f"[NOTIFY] {title}: {message}")

def _notify_windows(title: str, message: str, tray=None):
    # Prefer pystray bubble (no extra dep)
    if tray is not None:
        try:
            tray.notify(message, title)
            return
        except Exception:
            pass
    
    # Fallback: win10toast
    toast = Notification(
        app_id="PostureGuard",
        title=title,
        msg=message,
    )
    toast.set_audio(audio.Default, loop=False)
    toast.show()
    print(f"[NOTIFY] {title}: {message}")
    
