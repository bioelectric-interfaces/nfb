import os
from importlib import import_module
STATIC_PATH = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/static')

# pylsl has a dll order problem with kiwisolver (which is used by matplotlib). If kiwisolver is first imported before
# pylsl, pylsl will not find any streams. Import this package (pynfb) in your entry point to work around the problem.
import_module("pylsl")
import_module("kiwisolver")
