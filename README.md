## NFB Lab 
*NFB Lab* is a software that allows you to configure the design and conduct an experiment in real-time EEG/MEG paradigm
### Installation
1. Download and install [miniconda](https://conda.io/miniconda.html)
2. Open `<path to conda>\Scripts` folder in terminal
2. Install conda-packages from `requirements_conda.txt` by command: 

```
conda install --yes --file <path to requirements_conda.txt>
```
3. Install pip-packages from `requirements_pip.txt` by command: 
```
pip install -r <path to requirements_pip.txt>
```
### Open NFB Lab *experiment design module*
Run `main.py` by command 
```
python pynfb\main.py`
```