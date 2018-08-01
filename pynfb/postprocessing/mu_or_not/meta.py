from scipy.io import loadmat
import pandas as pd
from scipy.signal import hilbert, welch
import pylab as plt
from pynfb.postprocessing.utils import fft_filter
from pynfb.signal_processing.helpers import get_outliers_mask
import numpy as np
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog

from pynfb.inlets.montage import Montage

fs = 1000
band = (8, 14)
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T3', 'C3', 'Cz',
            'C4', 'T4', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']
data_dir = r'C:\Users\Nikolai\Desktop\Liza_diplom_data\Liza_diplom_data'
ref = 'Pz'
