"""
Script to look at artefacts from tACS stim
start with stim box ON
OFF at ~10sec
ON at ~25s
TRIG WAIT at ~1:20
OFF at ~1:40
"""
import mne
from utils.load_results import load_data
import numpy as np
import pandas as pd
import plotly_express as px

fname_raw = "/Users/christopherturner/Downloads/tavs_trig_test_2.vhdr"
raw = mne.io.read_raw_brainvision(fname_raw, preload=True)

alpha = raw.copy().filter(l_freq=8, h_freq=12)

print("doing stuff")