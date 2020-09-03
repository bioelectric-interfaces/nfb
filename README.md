# NFB Lab 
*NFB Lab* allows you to configure the design and conduct an experiment in real-time EEG/MEG paradigm.

## Installation
Prerequisites: [python](https://www.python.org/), [git](https://git-scm.com/), optionally [conda](https://docs.conda.io/en/latest/miniconda.html).

**Warning:** NFB Studio requires you install some outdated versions of packages. Consider installing it in a virtual enviroment, using tools such as venv or conda. For example, if using conda, create and activate a new environment by running these commands first:
```
conda create -n nfb python pip
conda activate nfb
```

Regardless of whether or not you are using a virtual environment, clone this repository and install the package in editable mode by running:
```
git clone https://github.com/nikolaims/nfb
cd nfb
pip install -e .
```

## Running the experiment designer
After installation, NFB Lab can be run from anywhere by using this command (when using a virtual environment, it has to be active):
```
pynfb
```
Or from the folder you installed it in:
```
python pynfb/main.py
```
If you need to use NFB Lab without a console, or run it from anywhere and distribute it, the best option is to freeze it into an executable.

### Freezing
NFB supports building as an executable, using the `pyinstaller` module. To use it, first install nfb with `freeze` addon:
```
pip install -e .[freeze]
```
Then build the executable from the included spec file (this might take some time):
```
pyinstaller freeze.spec
```
The executable can then be found in the `dist` folder.

### Commandline arguments
NFB supports commandline arguments to streamline opening and running experiments:
```
usage: pynfb [-h] [-x] [file]

positional arguments:
  file           open an xml experiment file when launched (optional)

optional arguments:
  -h, --help     show this help message and exit
  -x, --execute  run the experiment without configuring (requires file to be specified)
```
For example, to open an XML file with an experiment that you designed from command line, specify the path to it like so:
```
python pynfb/main.py your-experiment-file.xml
```
To run the experiment without configuring, use the `-x` or `--execute` option:
```
python pynfb/main.py -x your-experiment-file.xml
```
**Note:** The same command-line options are available for the `pynfb` executable, but since the console output is hidden, if you make a mistake the application will just not start.

## Acknowledgements
This work was supported by the [Center for Bioelectric Interfaces](https://bioelectric.hse.ru/en/) of the Institute for Cognitive Neuroscience of the National Research University Higher School of Economics, RF Government grant, ag. No. 14.641.31.0003.
