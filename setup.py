"""Setup script for pynfb package.
Warning: pynfb requires you install some outdated versions of packages. Consider installing pynfb in a virtual
enviroment, using tools such as venv or conda.
"""
from setuptools import setup, find_packages
from pathlib import Path

def setuptools_glob_workaround(package_name, glob):
    # https://stackoverflow.com/q/27664504/9118363
    package_path = Path(f'./{package_name}').resolve()
    return [str(path.relative_to(package_path)) for path in package_path.glob(glob)]


install_requires = [
    "h5py",
    "pyqt5",
    "pyopengl",
    "pyqtgraph",
    "seaborn",
    "scikit-learn",
    "sympy",
    "mne",
    "pylsl",
    "matplotlib",
    "gtts",
    "googletrans"
]

extras_require = {
    "freeze":  [
        "pyinstaller",
    ]
}

package_data = {
    "pynfb": setuptools_glob_workaround("pynfb", "static/**/*")
}

entry_points = {
    "gui_scripts": ["pynfb = pynfb.main:main"],
    "console_scripts": ["pynfb-d = pynfb.main:main"],
}

setup(
    name="pynfb",
    version="0.2.0",
    description="Conduct experiments in real-time EEG/MEG paradigm",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    packages=find_packages(),
    package_data=package_data,
)
