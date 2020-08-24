"""Setup script for pynfb package.
Warning: pynfb requires you install some outdated versions of packages. Consider installing pynfb in a virtual
enviroment, using tools such as venv or conda.
"""
from setuptools import setup

install_requires = [
    "h5py",
    "pyqt5",
    "pyopengl",
    "pyqtgraph",
    "seaborn",
    "scikit-learn",
    "sympy",
    "mne==0.12",
    "pylsl",
    "matplotlib==3.2.2",
]

extras_require = {
    "freeze":  ["pyinstaller==3.6"]
}

setup(
    name="pynfb",
    version="0.1",
    description="Conduct experiments in real-time EEG/MEG paradigm",
    install_requires=install_requires,
    extras_require=extras_require
)
