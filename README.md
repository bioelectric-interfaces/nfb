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

### Commandline arguments
NFB supports commandline arguments to streamline opening and running experiments:
```
usage: main.py [-h] [-x] [file]

positional arguments:
  file           open an xml experiment file when launched (optional)

optional arguments:
  -h, --help     show this help message and exit
  -x, --execute  run the experiment without configuring (requires file to be specified)
```
For example, to open an experiment file from command line, specify the path to it like so:
```
python pynfb/main.py experiment-file.xml
```
To run the experiment without configuring, use the `-x` or `--execute` option:
```
python pynfb/main.py -x experiment-file.xml
```

## Acknowledgements
This work was supported by the [Center for Bioelectric Interfaces](https://bioelectric.hse.ru/en/) of the Institute for Cognitive Neuroscience of the National Research University Higher School of Economics, RF Government grant, ag. No. 14.641.31.0003.
