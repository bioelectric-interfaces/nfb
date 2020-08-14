"""Setup script for nfb package.
Warning: nfb requires you install some outdated versions of packages. Consider installing nfb in a virtual enviroment,
using tools such as venv or conda.
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
    "pylsl"
]

setup(
    name="nfb",
    version="0.1",
    description="Conduct experiments in real-time EEG/MEG paradigm",
    install_requires=install_requires
)
