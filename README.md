# PostureGuard 🪑

Real-time posture monitoring that runs quietly in the background and alerts you when you start slouching.
No accounts. No recordings. No data leaves your machine.

![demo](gif-or-screenshot)

## 🚀 Download (No setup required)

👉 **[Download for Windows](https://github.com/MatheusDSantossi/posture-guard/releases/tag/v0.2.1)**

- No Python required
- Runs in the background
- Takes ~5 seconds to start

> First launch: sit straight and look forward for ~3 seconds during calibration

## Quick start

> ⚠️ First-time setup requires downloading a small model file (~20MB)

```bash
# 1. Download the model file
curl -L -o pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

# 2. Install
pip install -e .

# 3. Run
posture-guard

# And for direct module execution:
python -m posture_guard
```

On first launch a calibration window appears for ~3 seconds — **sit straight and look forward** — then it closes and runs silently in the background.

## Usage

| Command | Description |
|---|---|
| `posture-guard` | Run in background (headless) |
| `DEBUG=1 posture-guard` | Show camera window with metric overlay |
| `make run` | Same as `posture-guard` |
| `make debug` | Same as `DEBUG=1 posture-guard` |

## Tray icon

Right-click the tray icon (🟢 good / 🔴 bad / 🟡 paused):

| Option | Action |
|---|---|
| Pause / Resume | Temporarily stop monitoring |
| Recalibrate | Re-run calibration (e.g. after moving your chair) |
| Quit | Stop the application |

## Build a standalone app

No Python required for end users:

```bash
make dev        # install PyInstaller
make build-app  # produces dist/PostureGuard(.app on macOS)
```

- **macOS** — move `dist/PostureGuard.app` to `/Applications`, then double-click
- **Windows** — run `dist/PostureGuard.exe`
- **Linux** — run `dist/PostureGuard`

## 🧠 What it detects

PostureGuard compares your current posture to your calibrated “good posture” baseline:

- **Slouching** → shoulders drop
- **Hunching** → shoulders pull inward
- **Leaning** → head shifts left/right
- **Tilting** → uneven shoulders or head angle
- **Neck strain** → head stays forward while shoulders move

All checks are personalized to you, not based on fixed body proportions.

## Configuration

All thresholds can be tuned via environment variables, no code changes needed:

| Variable | Default | Description |
|---|---|---|
| `PG_SHOULDER_DROP` | `0.04` | Shoulder Y drop threshold (normalized) |
| `PG_SHOULDER_WIDTH` | `0.06` | Shoulder narrowing threshold (normalized) |
| `PG_SHOULDER_TILT` | `8.0` | Shoulder tilt delta (degrees) |
| `PG_HEAD_RISE` | `0.05` | Neck gap threshold (normalized) |
| `PG_FORWARD_LEAN` | `0.04` | Lateral lean threshold (normalized) |
| `PG_HEAD_TILT` | `10.0` | Head tilt delta (degrees) |
| `PG_BAD_DURATION` | `2.0` | Seconds of sustained bad posture before alert |
| `PG_ALERT_COOLDOWN` | `30.0` | Minimum seconds between alerts |
| `PG_MODEL_PATH` | `pose_landmarker_full.task` | Path to MediaPipe model |
| `DEBUG` | `0` | Set to `1` to show camera + metric overlay |

Example:

```bash
PG_BAD_DURATION=3 PG_ALERT_COOLDOWN=60 posture-guard
```

## Project structure

```
posture-guard/
├── posture_guard/
│   ├── __init__.py       # version
│   ├── __main__.py       # python -m posture_guard
│   ├── main.py           # entry point & main loop
│   ├── config.py         # all constants (env-overridable)
│   ├── detector.py       # posture logic & baseline
│   ├── calibration.py    # calibration window
│   ├── notifier.py       # OS notifications
│   ├── tray.py           # system tray icon
│   └── utils.py          # helper functions
├── pyproject.toml
├── Makefile
├── README.md
├── PostureGuard.spec
├── requirements.txt
└── LICENSE
```

## 💡 Why I built this

Most posture apps are either intrusive, inaccurate, or send your data to the cloud.

PostureGuard is:
- local-first
- privacy-focused
- actually usable in the background

## Versioning

The version lives in `posture_guard/__init__.py` and `pyproject.toml`. To release a new version:

1. Bump `__version__` in `__init__.py`
2. Bump `version` in `pyproject.toml`
3. Tag the commit: `git tag v0.2.1 && git push --tags`

## 🔒 Safety

- 100% local, no internet connection required after setup
- No data collection
- Open source (inspect everything in this repo)

## Privacy

No data leaves your machine. No frames are written to disk.

## License

MIT
