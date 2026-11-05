"""
Persistent user settings — saved to a JSON file in the per-user config directory.
Provides defaults and a clean API to read/write individual keys.
"""
import json
import os
import platform


def _settings_dir() -> str:
    """
    Per-user config directory. Not the bundle directory: in the one-file .exe that
    is sys._MEIPASS, a temp folder deleted on exit, so settings would never persist.
    """
    system = platform.system()
    if system == "Windows":
        return os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "PostureGuard")
    if system == "Darwin":
        return os.path.expanduser("~/Library/Application Support/PostureGuard")
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "posture-guard")

_SETTINGS_FILE = os.path.join(_settings_dir(), "posture_guard_settings.json")
 
DEFAULTS = {
    "alert_cooldown":       30,     # seconds between standard notifications
    "bad_posture_duration": 2.0,    # seconds of sustained bad posture before alert
    "extreme_alert":        False,  # play sound + flash screen
    "extreme_sound":        True,   # play audio on extreme alert
    "extreme_flash":        True,   # flash screen on extreme alert
    "sensitivity":          "medium",  # low | medium | high
}

# Sensitivity presets — map to threshold multipliers applied on top of config.py defaults
SENSITIVITY_MULTIPLIERS = {
    "low":    1.6,   # harder to trigger — forgives minor slouching
    "medium": 1.0,   # default
    "high":   0.6,   # easier to trigger — catches subtle posture changes
}

_load_warned = False

def load() -> dict:
    global _load_warned
    if os.path.exists(_SETTINGS_FILE):
        try:
            with open(_SETTINGS_FILE, "r") as f:
                data = json.load(f)
            # Fill in any missing keys from defaults (handles upgrades)
            return {**DEFAULTS, **data}
        # Broad on purpose: any unreadable/corrupt file (OSError, JSONDecodeError,
        # TypeError from a non-dict payload) must fall back to defaults
        except Exception as e:  # noqa: BLE001
            # load() runs every frame, so warn once instead of flooding the log
            if not _load_warned:
                print(f"[WARN] Could not read settings, using defaults: {e}")
                _load_warned = True

    return dict(DEFAULTS)

def save(settings: dict): 
    try:
        os.makedirs(os.path.dirname(_SETTINGS_FILE), exist_ok=True)
        with open(_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    # Broad on purpose: a failed save must not crash the settings window
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Could not save settings: {e}")

def get(key: str):
    return load().get(key, DEFAULTS.get(key))

def set(key: str, value):
    s = load()
    s[key] = value
    save(s)
