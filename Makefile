dev:
	pip install -e ".[dev]"

# Run
run:
	posture-guard

debug:
	DEBUG=1 posture-guard

# Build standalone app (requires: make dev)
build-app:
	pyinstaller \
		--name "PostureGuard" \
		--onefile \
		--windowed \
		--add-data "pose_landmarker_full.task:." \
		posture_guard/main.py
	@echo ""
	@echo "✅ App built → dist/PostureGuard"
	@echo "   macOS: move dist/PostureGuard.app to /Applications"
	@echo "   Windows/Linux: run dist/PostureGuard directly"

# Housekeeping
clean:
	rm -rf build dist *.spec __pycache__ posture_guard/__pycache__
