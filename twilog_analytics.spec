# -*- mode: python ; coding: utf-8 -*-
import subprocess
import sys
import shutil
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH)
subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "generate_licenses.py")],
    check=True,
)

datas = [
    ("app/templates", "app/templates"),
    ("app/static", "app/static"),
]
datas += collect_data_files("sudachidict_full")
datas += collect_data_files("sudachipy")

hiddenimports = [
    "sudachidict_full",
]

a = Analysis(
    ["run_exe.py"],
    pathex=[".", "src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="twilog-analytics",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="twilog-analytics",
)

# Keep runtime files under _internal, but place sample CSV next to the exe for convenience.
sample_name = "TweeTeaFOX223-260301_only.csv"
sample_src = ROOT / "sample_csv" / sample_name
sample_dst_dir = Path(coll.name) / "sample_csv"
sample_dst_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(sample_src, sample_dst_dir / sample_name)
