"""
Check some pilot data to see if alpha is modulated specifically, or if the 1/f changes
i.e.
    - compare the baseline alpha power before NFB to the baseline alpha power after NFB
    - can also compare alpha power DURING the nfb task (ones that definitely get modulated) to the cue before the task
"""
# LOOK HERE: https://fooof-tools.github.io/fooof/auto_examples/analyses/plot_mne_example.html#sphx-glr-auto-examples-analyses-plot-mne-example-py


# General imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar

# Import MNE, as well as the MNE sample dataset
import mne
from mne import io
from mne.datasets import sample
from mne.viz import plot_topomap
from mne.time_frequency import psd_welch

# FOOOF imports
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum

# load the data

fname_nfb = "/Users/2354158T/Documents/GitHub/nfb/analysis/PO5_nfb3.vhdr"
# fname_ec1 = "/Users/2354158T/Documents/GitHub/nfb/analysis/PO5_nfb3.vhdr"
# fname_ec2 = "/Users/2354158T/Documents/GitHub/nfb/analysis/PO5_nfb3.vhdr"
raw_nfb = mne.io.read_raw_brainvision(fname_nfb, preload=True)
# raw_ec1 = mne.io.read_raw_brainvision(fname_ec1, preload=True)
# raw_ec2 = mne.io.read_raw_brainvision(fname_ec2, preload=True)

raw_nfb.info['bads'].extend([x for x in raw_nfb.ch_names if x not in ['PO7', 'PO8', 'ECG', 'EOG']])
raw = raw_nfb.pick_types(meg=False, eeg=True, eog=False, exclude='bads')


# Calculate power spectra across the the continuous data
spectra, freqs = psd_welch(raw, fmin=1, fmax=40, tmin=0, tmax=250,
                           n_overlap=150, n_fft=300)
# Initialize a FOOOFGroup object, with desired settings
fg = FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=0.15,
                peak_threshold=2., max_n_peaks=6, verbose=False)

# Define the frequency range to fit
freq_range = [1, 30]

###################################################################################################

# Fit the power spectrum model across all channels
fg.fit(freqs, spectra, freq_range)

###################################################################################################

# Check the overall results of the group fits
fg.plot()

# Define frequency bands of interest
bands = Bands({'theta': [3, 7],
               'alpha': [7, 14],
               'beta': [15, 30]})

###################################################################################################

# Extract alpha peaks
alphas = get_band_peak_fg(fg, bands.alpha)

# Extract the power values from the detected peaks
alpha_pw = alphas[:, 1]

###################################################################################################

# Plot the topography of alpha power
plot_topomap(alpha_pw, raw.info, cmap=cm.viridis, contours=0)

# Compare the power spectra between low and high exponent channels
fig, ax = plt.subplots(figsize=(8, 6))
plot_spectrum(fg.freqs, fg.get_fooof(np.argmin(exps)).power_spectrum,
              ax=ax, label='Low Exponent')
plot_spectrum(fg.freqs, fg.get_fooof(np.argmax(exps)).power_spectrum,
              ax=ax, label='High Exponent')


print("")