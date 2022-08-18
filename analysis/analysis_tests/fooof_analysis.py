# Import required code for visualizing example models
import mne
from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectrum
from fooof.plts.annotate import plot_annotated_model

# load the data
fname_raw = "/Users/2354158T/Documents/GitHub/nfb/analysis/PO5_nfb3.vhdr"
raw = mne.io.read_raw_brainvision(fname_raw, preload=True)


print("")