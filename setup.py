"""Setup script for pynfb package.
Warning: pynfb requires you install some outdated versions of packages. Consider installing pynfb in a virtual
enviroment, using tools such as venv or conda.
"""
from setuptools import setup, find_packages

install_requires = [
    "h5py",
    "pyqt5",
    "pyopengl",
    "pyqtgraph",
    "seaborn",
    "scikit-learn",
    "sympy",
    "mne==0.14.1",
    "pylsl",
    "matplotlib==3.2.2",
]

extras_require = {
    "freeze":  ["pyinstaller==3.6"]
}

entry_points = {
    "console_scripts": ["pynfb=pynfb.main:main"]
}

setup(
    name="pynfb",
    version="0.1",
    description="Conduct experiments in real-time EEG/MEG paradigm",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    packages=find_packages(),
)
