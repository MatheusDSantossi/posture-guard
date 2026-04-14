# -*- mode: python ; coding: utf-8 -*-
# Run from project root with: pyinstaller PostureGuard.spec

from PyInstaller.utils.hooks import collect_all, collect_submodules

mp_datas, mp_binaries, mp_hiddenimports = collect_all('mediapipe')
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all('cv2')

a = Analysis(
    ['posture_guard/app.py'],          # relative to project root, not inside posture_guard/
    pathex=['.'],                      # project root on the search path
    binaries=mp_binaries + cv2_binaries,
    datas=[
        ('pose_landmarker_full.task', '.'),
        *mp_datas,
        *cv2_datas,
    ],
    hiddenimports=[
        *mp_hiddenimports,
        *cv2_hiddenimports,
        'mediapipe.tasks',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.vision',
        'mediapipe.tasks.python.core',
        'mediapipe.tasks.c',
        'mediapipe.tasks.c.vision',
        'mediapipe.python',
        'pystray',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
    ],
    excludes=[
        # exclude heavy mediapipe LLM modules — not needed for pose detection
        'mediapipe.tasks.python.genai',
        'jax',
        'torch',
        'sentencepiece',
    ],
    hookspath=[],
    runtime_hooks=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PostureGuard',
    debug=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,    # no terminal window
    onefile=True,
)
