# NFB Lab 
*NFB Lab* allows you to configure the design and conduct an experiment in real-time EEG/MEG paradigm.

## Installation
Prerequisites: [python](https://www.python.org/), [git](https://git-scm.com/), optionally [conda](https://docs.conda.io/en/latest/miniconda.html).

Clone this repository and install the package in editable mode by running:
```
git clone https://github.com/nikolaims/nfb
cd nfb
pip install -e .
```
**Warning**: nfb requires you install some outdated versions of packages. Consider installing nfb in a virtual enviroment, using tools such as venv or conda.

## Running the experiment designer
If you are using `conda` as your virtual environment provider, don't forget to create and activate a virtual environment:
```
conda create -n nfb python pip
conda activate nfb
```
Run the experiment designer:
```
python pynfb/main.py
```

## Acknowledgements
This work was supported by the [Center for Bioelectric Interfaces](https://bioelectric.hse.ru/en/) of the Institute for Cognitive Neuroscience of the National Research University Higher School of Economics, RF Government grant, ag. No. 14.641.31.0003.
