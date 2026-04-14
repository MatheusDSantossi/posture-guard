"""
System tray icon — green (good), red (bad), yellow (paused).
Requires pystray + Pillow. Gracefully skipped if not installed.
"""
import threading

try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False

def _make_icon(color=(0, 180, 0)):
    img = Image.new("RGB", (64, 64), (30, 30, 30))
    ImageDraw.Draw(img).ellipse([8, 8, 56, 56], fill=color)
    return img

class TrayIcon:
    """Wrapper around pystray.Icon with PostureGuard-specific controls."""
    
    def __init__(self, state: dict):
        self._state = state
        self._icon = None
        
        if not HAS_TRAY:
            print("[WARN] pystray/Pillow not installed — tray icon disabled.")
            return

        self._icon = pystray.Icon(
            "PostureGuard",
            _make_icon(),
            "PostureGuard",
            menu=pystray.Menu(
                pystray.MenuItem("Pause / Resume", self._on_pause),
                pystray.MenuItem("Recalibrate",    self._on_recalibrate),
                pystray.MenuItem("Quit",           self._on_quit),
            ),
        )
        
    def start(self):
        if self._icon:
            threading.Thread(target=self._icon.run, daemon=True).start()
            
    def stop(self):
        if self._icon:
            self._icon.stop()
            
    def set_good(self):
        self._set_color((0, 180, 0))
        
    def set_bad(self):
        self._set_color((200, 0, 0))
    
    def set_paused(self):
        self._set_color((200, 140, 0))
        
    def notify(self, title: str, message: str):
        if self._icon:
            try:
                self._icon.notify(message, title)
            except Exception:
                pass
    def _set_color(self, color: tuple):
        if self._icon:
            self._icon.icon = _make_icon(color)
            
    # Menu callbacks
    def _on_pause(self, icon, item):
        self._state["paused"] = not self._state["paused"]
        if self._state["paused"]:
            self.set_paused()
            print("[INFO] Paused")

    def _on_recalibrate(self, icon, item):
        self._state["recalibrate"] = True
        print("[INFO] Recalibration requested")

    def _on_quit(self, icon, item):
        self._state["running"] = False
        self.stop()

