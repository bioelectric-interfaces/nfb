"""
This script calculates the mean and max values for the cvsa/posner neurofeedback protocol.
It does this by fitting a normal distribution to the mean AAI values for a number of test runs
AAI values are those recorded online
the Mean of this normal distribution is 0 lateralisation (i.e. baseline threshold)
2 standard deviations of the distribution correspond to the minimum and maximum AAI values

based on:
@article{Schneider2020,
   abstract = {Visual attention can be spatially oriented, even in the absence of saccadic eye-movements, to facilitate the processing of incoming visual information. One behavioral proxy for this so-called covert visuospatial attention (CVSA) is the validity effect (VE): the reduction in reaction time (RT) to visual stimuli at attended locations and the increase in RT to stimuli at unattended locations. At the electrophysiological level, one correlate of CVSA is the lateralization in the occipital α-band oscillations, resulting from α-power increases ipsilateral and decreases contralateral to the attended hemifield. While this α-band lateralization has been considerably studied using electroencephalography (EEG) or magnetoencephalography (MEG), little is known about whether it can be trained to improve CVSA behaviorally. In this cross-over sham-controlled study we used continuous real-time feedback of the occipital α-lateralization to modulate behavioral and electrophysiological markers of covert attention. Fourteen subjects performed a cued CVSA task, involving fast responses to covertly attended stimuli. During real-time feedback runs, trials extended in time if subjects reached states of high α-lateralization. Crucially, the ongoing α-lateralization was fed back to the subject by changing the color of the attended stimulus. We hypothesized that this ability to self-monitor lapses in CVSA and thus being able to refocus attention accordingly would lead to improved CVSA performance during subsequent testing. We probed the effect of the intervention by evaluating the pre-post changes in the VE and the α-lateralization. Behaviorally, results showed a significant interaction between feedback (experimental–sham) and time (pre-post) for the validity effect, with an increase in performance only for the experimental condition. We did not find corresponding pre-post changes in the α-lateralization. Our findings suggest that EEG-based real-time feedback is a promising tool to enhance the level of covert visuospatial attention, especially with respect to behavioral changes. This opens up the exploration of applications of the proposed training method for the cognitive rehabilitation of attentional disorders.},
   author = {Christoph Schneider and Michael Pereira and Luca Tonin and José del R. Millán},
   doi = {10.1007/s10548-019-00725-9},
   issn = {15736792},
   issue = {1},
   journal = {Brain Topography},
   keywords = {Alpha band lateralization,Brain-computer interface,Closed-loop,Covert visuospatial attention,EEG,Hemispatial neglect},
   pages = {48-59},
   pmid = {31317285},
   publisher = {Springer US},
   title = {Real-time EEG Feedback on Alpha Power Lateralization Leads to Behavioral Improvements in a Covert Attention Task},
   volume = {33},
   url = {https://doi.org/10.1007/s10548-019-00725-9},
   year = {2020},
}
"""
import os
import platform
from utils.load_results import load_data
import numpy as np
from scipy.stats import norm
import plotly.graph_objs as go
import matplotlib.pyplot as plt

import analysis.analysis_functions as af

if platform.system() == "Windows":
    userdir = "2354158T"
else:
    userdir = "christopherturner"


def cvsa_threshold(h5file, plot=False, alpha_band=(8, 12)):
    df1, fs, channels, p_names = load_data(h5file)
    df1['sample'] = df1.index

    df1[df1.cue > 0].groupby("block_number", as_index=True).first()

    # Drop non eeg data
    drop_cols = [x for x in df1.columns if x not in channels]
    drop_cols.extend(['MKIDX', 'EOG', 'ECG', 'signal_AAI'])
    eeg_data = df1.drop(columns=drop_cols)

    # Rescale the data (units are microvolts - i.e. x10^-6
    eeg_data = eeg_data * 1e-6

    # Drop everthing except the AAI
    df1_aai = df1[['signal_AAI', 'signal_left', 'signal_right', 'block_name', 'block_number', 'sample', 'chunk_n']]

    # Extract all of the AAI blocks
    df1_aai = df1_aai[df1_aai['block_name'].str.contains("nfb")]

    # Calculate mean of all AAI blocks
    # block_means = df1.groupby('block_number', as_index=False)['signal_AAI'].mean()

    # Fit normal distribution
    data = df1_aai['signal_AAI'].dropna().to_numpy()# norm.rvs(10.0, 2.5, size=500)
    # Fit a normal distribution to the data:
    mu_online, std_online = norm.fit(data)
    # Plot the histogram.
    plt.hist(data, bins=20, density=True, alpha=0.6, color='g')
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu_online, std_online)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu_online = %.2f,  std _online= %.2f" % (mu_online, std_online)
    plt.title(title)
    if plot:
        plt.show()
    print(f"ONLINE AAI MEAN: {mu_online}, STD: {std_online}")

    # Do the same, but for the raw AAI
    # Get AAI by calculating raw signals from hdf5 (i.e. no smoothing)------------------------------------

    aai_duration_samps = df1.shape[0]#10000
    # alpha_band = (7.75, 11.75)

    chunksize = df1[df1.chunk_n > 0]['chunk_n'].median()

    mean_raw_l, std1_raw_l, pwr_raw_l = af.get_nfblab_power_stats_pandas(eeg_data[0:aai_duration_samps], fband=alpha_band, fs=fs,
                                                                 channel_labels=eeg_data.columns, chs=["PO7=1"],
                                                                 fft_samps=fs, chunksize=chunksize)

    mean_raw_r, std1_raw_r, pwr_raw_r = af.get_nfblab_power_stats_pandas(eeg_data[0:aai_duration_samps], fband=alpha_band, fs=fs,
                                                                 channel_labels=eeg_data.columns, chs=["PO8=1"],
                                                                 fft_samps=fs, chunksize=chunksize)
    aai_raw_left = (pwr_raw_l - pwr_raw_r) / (pwr_raw_l + pwr_raw_r)


    df1['raw_aai'] = aai_raw_left

    # Replicate the moving average smoother
    df1['raw_smoothed'] = df1['raw_aai'].rolling(window=int(fs/10)).mean()

    # Extract all of the AAI blocks
    df1 = df1[df1['block_name'].str.contains("nfb")]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df1.index, y=df1['raw_aai'],
                        mode='lines',
                        name='AAI_calc'))
    fig1.add_trace(go.Scatter(x=df1.index, y=df1['signal_AAI'],
                        mode='lines',
                        name='AAI_online'))
    fig1.add_trace(go.Scatter(x=df1.index, y=df1['raw_smoothed'],
                        mode='lines',
                        name='raw_smoothed'))
    fig1.add_trace(go.Scatter(x=df1.index, y=df1['signal_left'],
                        mode='lines',
                        name='signal_left'))
    fig1.add_trace(go.Scatter(x=df1.index, y=df1['signal_right'],
                        mode='lines',
                        name='signal_right'))
    fig1.show()

    # Calculate mean of all AAI blocks
    # Fit normal distribution
    data = df1['raw_aai'].to_numpy()# norm.rvs(10.0, 2.5, size=500)
    # Fit a normal distribution to the data:
    mu_raw, std_raw = norm.fit(data)
    # Plot the histogram.
    plt.hist(data, bins=20, density=True, alpha=0.6, color='g')
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu_raw, std_raw)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu_raw = %.2f,  std_raw = %.2f" % (mu_raw, std_raw)
    plt.title(title)
    if plot:
        plt.show()

    print(f"RAW AAI MEAN: {mu_raw}, STD: {std_raw}")

    return mu_raw, std_raw

if __name__ == "__main__":
    # Read in the raw data of the test
    task_data = {}
    # h5file = f"/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-test_task_cvsa_test_04-16_17-00-25/experiment_data.h5"
    h5file = f"/Users/christopherturner/Documents/EEG_Data/cvsa_pilot_testing/lab_test_20220428/0-test_task_ct_test_04-28_16-56-03/experiment_data.h5"
    mu, std = cvsa_threshold(h5file)

