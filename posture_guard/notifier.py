"""
Cross-platform OS notifications.
Falls back to a terminal print if no native method is available.
"""
import platform
import subprocess
import threading

_WIN_APP_ID   = "PostureGuard"
_WIN_REG_KEY  = r"HKCU\SOFTWARE\Classes\AppUserModelId\PostureGuard"
_registered   = False   # only register once per process

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
 
#  Windows
def _ensure_registered():
    """
    Write the AppUserModelId registry key so Windows accepts our toasts.
    Only runs once per process. Uses reg.exe — no extra deps.
    """
    global _registered
    if _registered:
        return
    
    try:
        subprocess.run(
            ["reg", "add", _WIN_REG_KEY,
             "/v", "DisplayName", "/t", "REG_SZ", "/d", "PostureGuard", "/f"], capture_output=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        _registered = True
        print("[NOTIFY] App registered for Windows toasts")
    except Exception as e:
        print(f"[NOTIFY] Registry write failed (toasts may not show): {e}")
        
 
def _notify_windows(title: str, message: str, tray=None, play_sound: bool = False):
    _ensure_registered()
    shown = False
    
    # Try winotify first (best Windows experience)
    try:
        title_safe   = _escape_xml(title)
        message_safe = _escape_xml(message)
        # The closing "@ MUST start at column 0 — no leading whitespace allowed
        ps_script = (
            '[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null\n'
            '[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null\n'
            f'$xml = New-Object Windows.Data.Xml.Dom.XmlDocument\n'
            f'$xml.LoadXml("<toast><visual><binding template=\'ToastGeneric\'>'
            f'<text>{title_safe}</text>'
            f'<text>{message_safe}</text>'
            f'</binding></visual></toast>")\n'
            '$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)\n'
            '$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("PostureGuard")\n'
            '$notifier.Show($toast)\n'
        )
        
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script], timeout=8, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        
        if result.returncode == 0:
            shown = True
            print("[NOTIFY] PowerShell toast sent")
        else:
            print(f"[NOTIFY] PowerShell toast failed (rc={result.returncode}): {result.stderr.strip()}")
            
    except Exception as e:
        print(f"[NOTIFY] PowerShell toast failed: {e}")
        
    # Fallback: winotify
    if not shown:
        try:
            from winotify import Notification, audio
            
            toast = Notification(
                app_id="PostureGuard",
                title=title,
                msg=message,
                duration="short"
            )
            toast.set_audio(audio.Silent, loop=False)
            toast.show()
            shown = True
        except Exception as e:
            print(f"[NOTIFY] winotify failed: {e}")
            
    # Fallback: pystray bubble
    if not shown and tray is not None:
        try:
            tray.notify(message, title)
            shown = True
        except Exception as e:
            print(f"[NOTIFY] pystray fallback failed: {e}")
    
    if not shown:
        print(f"[NOTIFY] {title}: {message}")
    
    # Fallback: pystray buble
    if play_sound:
        _fire_sound()
 
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
        print("[NOTIFY] notify-send not found. Install libnotify-bin.")
        print(f"[NOTIFY] {title}: {message}")
    except Exception as e:
        print(f"[NOTIFY] Linux notification failed: {e}")
        print(f"[NOTIFY] {title}: {message}")
        
    if play_sound:
        _fire_sound()

def _escape_xml(text: str) -> str:
    """Escape characters that would break the PowerShell XML tempalte."""
    return (
          text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
    )        

def _fire_sound():
    """Play the alert sound in a daemon thread so it never blocks detection"""
    from posture_guard.playsound import play_sound
    threading.Thread(target=play_sound, daemon=True).start()
