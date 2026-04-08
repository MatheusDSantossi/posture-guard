# PostureGuard 🪑

Real-time posture monitoring using your webcam and MediaPipe. No images or video are saved — frames are processed in memory and discarded immediately.

## What it checks

- **Shoulder tilt** — detects if shoulders are uneven
- **Forward/sideward lean** — detects if your head drifts too far from your shoulders

Alerts are printed to the terminal with a configurable cooldown so you're not spammed.

## Requirements

- Python 3.9+
- A webcam

## Setup

```bash
pip install -r requirements.txt
python posture_guard.py
```

Press **Q** in the camera window to quit.

## Configuration

At the top of `posture_guard.py` you can adjust:

| Variable | Default | Description |
|---|---|---|
| `SHOULDER_TILT_THRESHOLD` | `15` deg | Max allowed shoulder angle |
| `FORWARD_LEAN_THRESHOLD` | `0.07` | Max normalized lateral head offset |
| `ALERT_COOLDOWN_SECONDS` | `10` | Seconds between repeated alerts |

## Privacy

No data leaves your machine. No frames are written to disk. The camera feed is processed live by MediaPipe and the raw pixels are never stored.

## License

MIT
