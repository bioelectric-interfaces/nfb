## NFB Lab 
*NFB Lab* is a software that allows you to configure the design and conduct an experiment in real-time EEG/MEG paradigm
### Setup python interpreter
1. Download and install [miniconda](https://conda.io/miniconda.html)
2. Open `Scripts` folder from installed miniconda folder
3. Install conda-packages from `requirements_conda.txt` by command: 
```
conda install --yes --file <path to requirements_conda.txt>
```
4. Install pip-packages from `requirements_pip.txt` by command: 
```
pip install -r <path to requirements_pip.txt>
```
### Running NFB Lab *experiment design module*
Run `main.py` by command 
```
python pynfb\main.py`
```
Where `python` interpreter is from installed miniconda folder