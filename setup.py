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
    "mne",
    "pylsl",
    "matplotlib",
]

extras_require = {
    "freeze":  [
        "pyinstaller-hooks-contrib @ https://github.com/pyinstaller/pyinstaller-hooks-contrib/archive/465a2caccb5913ebfc64561e8055e81d73188736.zip",
        "pyinstaller",
    ]
}

package_data = {
    "pynfb": ["static/imag/*"]
}

entry_points = {
    "gui_scripts": ["pynfb=pynfb.main:main"]
}

setup(
    name="pynfb",
    version="0.1",
    description="Conduct experiments in real-time EEG/MEG paradigm",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    packages=find_packages(),
    package_data=package_data,
)
