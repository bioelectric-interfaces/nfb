## NFB Lab 
*NFB Lab* is a software that allows you to configure the design and conduct an experiment in real-time EEG/MEG paradigm
### Setup python interpreter
1. Download and install [miniconda](https://conda.io/miniconda.html). Optionally create new environment.
2. Install conda-packages by command:
```
conda install h5py pyqt pyopengl pyqtgraph seaborn scikit-learn sympy
```
3. Install pip-packages by command:
```
pip install mne pylsl
```
### Running NFB Lab *experiment design module*
Run `main.py` by command 
```
python pynfb\main.py`
```
Where `python` interpreter is from installed miniconda folder