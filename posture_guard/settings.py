"""
Persistent user settings — saved to a JSON file next to the .exe / in the project root.
Provides defaults and a clean API to read/write individual keys.
"""
import json
import os
from posture_guard.utils import get_resource_path

_SETTINGS_FILE = os.path.join(os.path.dirname(get_resource_path(".")), "posture_guard_settings.json")
 
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

def load() -> dict:
    if os.path.exists(_SETTINGS_FILE):
        try:
            with open(_SETTINGS_FILE, "r") as f:
                data = json.load(f)
            # Fill in any missing keys from defaults (handles upgrades)
            return {**DEFAULTS, **data}
        except Exception:
            pass
        
    return dict(DEFAULTS)

def save(settings: dict): 
    try:
        with open(_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save settings: {e}")

def get(key: str):
    return load().get(key, DEFAULTS.get(key))

def set(key: str, value):
    s = load()
    s[key] = value
    save(s)
