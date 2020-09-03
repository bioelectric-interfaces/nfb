# -*- mode: python ; coding: utf-8 -*-
# NOTE: pyinstaller 4.0 does not work. Works with 3.6
# NOTE: to build this matplotlib needs to be downgraded. Works with 3.2.2
import sys
from distutils.sysconfig import get_python_lib

sys.setrecursionlimit(5000)
block_cipher = None


a = Analysis(
    ['pynfb\\main.py'],
    pathex=['D:\\Files\\Develop\\nfb'],
    binaries=[],
    datas=[
        (get_python_lib()+"/pylsl/lib", "pylsl/lib"),
        (get_python_lib()+"/mne/channels/data", "mne/channels/data"),
        ("pynfb/static/imag", "pynfb/static/imag"),
    ],
    hiddenimports=[
        "scipy.special.cython_special",
        "sklearn.neighbors._typedefs",
        "sklearn.neighbors._quad_tree",
        "sklearn.utils._cython_blas",
        "sklearn.tree._utils"
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='pynfb',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True
)
