import sys, os, argparse, importlib
from PyQt5 import QtGui, QtWidgets

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nfblab')
parser.add_argument('--design')
args = parser.parse_args()

# validate arguments
if args.design is None or args.nfblab is None:
    raise SyntaxError('\nPlease provide all arguments: '
                      '\n\t --nfblab Path to nfb directory'
                      '\n\t --design Path to design file')

if not os.path.isfile(args.design):
    raise FileNotFoundError('No such design file {}. Please, correct --design value.'.format(args.design))

if not os.path.isdir(args.nfblab):
    raise FileNotFoundError("Path to nfblab doesn't exist. Please, correct --nfblab value.")

sys.path.insert(0, args.nfblab)
if importlib.util.find_spec('pynfb') is None:
    raise ImportError('Cannot import nfblab package. Please, correct --nfblab value')

# import NFBLab
from pynfb.experiment import Experiment
from pynfb.serializers.xml_ import xml_file_to_params

# run experiment
app = QtWidgets.QApplication(sys.argv)
experiment = Experiment(app, xml_file_to_params(args.design))
sys.exit(app.exec_())
