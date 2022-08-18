"""
Check some pilot data to see if alpha is modulated specifically, or if the 1/f changes
i.e.
    - compare the baseline alpha power before NFB to the baseline alpha power after NFB
    - can also compare alpha power DURING the nfb task (ones that definitely get modulated) to the cue before the task
"""
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