dev:
	pip install -e ".[dev]"

# Run
run:
	posture-guard

debug:
	DEBUG=1 posture-guard

# Build standalone app (requires: make dev, and pose_landmarker_full.task in the project root)
# Uses the tracked spec, same as the release workflow, so the audio assets are bundled
build-app:
	pyinstaller PostureGuard.spec
	@echo ""
	@echo "✅ App built → dist/PostureGuard"

# Housekeeping (PostureGuard.spec is tracked, so it is not removed)
clean:
	rm -rf build dist __pycache__ posture_guard/__pycache__
