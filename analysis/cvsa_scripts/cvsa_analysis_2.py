"""
Scropt to analyse covert attention protocols
"""

import os
import mne
import numpy as np
import ast
import matplotlib.pyplot as plt
import platform
from scipy.stats import norm

from pynfb.signal_processing.filters import ExponentialSmoother, FFTBandEnvelopeDetector
from utils.load_results import load_data
import pandas as pd
import plotly_express as px
import plotly.graph_objs as go
import analysis.analysis_functions as af
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw, apply_inverse, apply_inverse_epochs

from pynfb.helpers import roi_spatial_filter as rsf
from philistine.mne import savgol_iaf


task_data = {}
# h5file = f"/Users/{userdir}/Documents/EEG_Data/cvsa_test1/0-nfb_task_cvsa_test_04-06_17-35-45/experiment_data.h5"
# h5file = f"/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-test_task_cvsa_test_04-16_17-00-25/experiment_data.h5"
# h5file = f"/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-nfb_task_cvsa_test_04-22_16-09-15/experiment_data.h5"
# h5file = f"/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-test_task_ct_noise_test_04-26_10-00-42/experiment_data.h5"
# h5file = f"/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-test_task_ct_test_04-29_17-42-07/experiment_data.h5"

#--- lab test 28/04/22
# h5file = "/Users/christopherturner/Documents/EEG_Data/cvsa_pilot_testing/lab_test_20220428/0-nfb_task_ct_test_04-28_17-39-22/experiment_data.h5"

# --- pilot PO1 04/04/22
# h5file = "../../pynfb/results/0-nfb_task_PO1_05-04_10-31-34/experiment_data.h5"

def get_cue_dir(df1, channels):
    df1['sample'] = df1.index
    channels.append("signal_AAI")

    # Get the cued NFB side (preceding cue direction)
    cue_dir_df = df1[(df1['cue'].isin([1,2,3])) & (df1['block_name'] == 'cue')]
    cue_dir_key = {}
    for index, row in cue_dir_df.iterrows():
        cue_dir_key[row['block_number']+1] = row['cue']
    df1["cue_dir"] = df1['block_number'].map(cue_dir_key)
    return df1

def get_posner_time(df1):
    # Get the reaction times for valid and invalid trials
    # Valid cue => Posner_stim == cue
    # Invalid cue => posner_stim != cue
    valid_cue_key = {}
    for index, row in df1[df1['posner_stim'].isin([1,2])].iterrows():
        if row['posner_stim'] == row['cue_dir']:
            valid_cue_key[row['block_number']] = True
        elif row['cue_dir'] == 3:
            valid_cue_key[row['block_number']] = 'NA'
        else:
            valid_cue_key[row['block_number']] = False
    df1["valid_cue"] = df1['block_number'].map(valid_cue_key)

    # RT = time between 'posner_stim' and end of block
    # Posner stim onset time = first time that isn't 0 in the block
    grouped_df = df1[df1['posner_time'] >0].groupby("block_number", as_index=False)
    stim_times = grouped_df.min()
    reaction_time_key = {}
    for index, row in stim_times.iterrows():
        block = df1[df1['block_number'] == row['block_number']]
        if not bool(block[block['response_data']>0].empty):
            response_time = block[block['response_data']>0].iloc[0]['response_data']
            reaction_time_key[row['block_number']] = response_time - row['posner_time']
    df1["reaction_time"] = df1['block_number'].map(reaction_time_key)
    return df1

def read_log_file(logfile):
    string1 = 'BLOCK SCORE'

    # opening a text file
    file1 = open(logfile, "r")

    # setting flag and index to 0
    flag = 0
    index = 0
    score_line = ""
    score = {}

    # Loop through the file line by line
    for line in file1:
        index += 1

        # checking string is present in line or not
        if string1 in line:
            flag = 1
            score_line = line
            score = ast.literal_eval(score_line.split("BLOCK SCORE: ")[1])

        # checking condition for string found or not
    if flag == 0:
        print('String', string1, 'Not Found')
    else:
        print('String', string1, 'Found In Line', index)

    # closing text file
    file1.close()
    return score

def cvsa_analysis(df1, fs, channels, p_names, block_idx=0, participant="", score={}, df1_bl=None):
    # TODO - refactor all this nicely (and use/ add to/ refactor the helper functions)

    #=============================================
    # BASELINE STUFF -----------------------------
    df1_bl['sample'] = df1_bl.index
    # aai_df_bl  = df1_bl [['PO8', 'PO7', 'signal_AAI', 'block_name', 'block_number', 'sample', 'chunk_n']]
    drop_cols = [x for x in df1_bl.columns if x not in channels]
    drop_cols.extend(['MKIDX', 'EOG', 'ECG', 'signal_AAI'])
    eeg_data_bl = df1_bl.drop(columns=drop_cols)

    # Rescale the data (units are microvolts - i.e. x10^-6
    eeg_data_bl = eeg_data_bl * 1e-6
    aai_duration_samps_bl = df1_bl.shape[0]#10000
    alpha_band = (7.25, 11.25)
    chunksize = df1_bl[df1_bl.chunk_n > 0]['chunk_n'].median()
    mean_raw_l, std1_raw_l, pwr_raw_l = af.get_nfblab_power_stats_pandas(eeg_data_bl[0:aai_duration_samps_bl], fband=alpha_band, fs=fs,
                                                                 channel_labels=eeg_data_bl.columns, chs=["PO7=1"],
                                                                 fft_samps=fs, chunksize=chunksize)

    mean_raw_r, std1_raw_r, pwr_raw_r = af.get_nfblab_power_stats_pandas(eeg_data_bl[0:aai_duration_samps_bl], fband=alpha_band, fs=fs,
                                                                 channel_labels=eeg_data_bl.columns, chs=["PO8=1"],
                                                                 fft_samps=fs, chunksize=chunksize)
    aai_raw_left_bl = (pwr_raw_l - pwr_raw_r) / (pwr_raw_l + pwr_raw_r)

    # Plot the median of the left and right alphas from the online AAI signal
    aai_df_raw_bl = df1_bl[['PO8', 'PO7', 'block_name', 'block_number', 'sample', 'signal_AAI']]
    aai_df_raw_bl['raw_aai'] = aai_raw_left_bl
    aai_df_raw_bl['raw_aai'] = aai_df_raw_bl['raw_aai'].rolling(window=int(fs / 10)).mean()
    aai_df_raw_bl['raw_smoothed'] = aai_df_raw_bl['raw_aai'].rolling(window=int(fs / 10)).mean()
    aai_df_raw_bl['P08_pwr'] = pwr_raw_r
    aai_df_raw_bl['P07_pwr'] = pwr_raw_l
    aai_df_raw_bl = aai_df_raw_bl[(aai_df_raw_bl['block_name'] == "baseline_eo") | (aai_df_raw_bl['block_name'] == "baseline_ec")]

    fig = px.violin(aai_df_raw_bl, x="block_name",  y="raw_aai", box=True, range_y=[-1, 1], title=f"block:{block_idx}_{participant} (calc AAI)")
    fig.show()

    fig1 = go.Figure()
    fig1.add_trace(go.Box(x=aai_df_raw_bl['block_name'], y=aai_df_raw_bl['P07_pwr'],
                          line=dict(color='blue'),
                          name='PO7_pwr'))
    fig1.add_trace(go.Box(x=aai_df_raw_bl['block_name'], y=aai_df_raw_bl['P08_pwr'],
                          line=dict(color='red'),
                          name='PO8_pwr')).show()


    #=============================================
    #---NFB STUFF---------------------------------
    # Get the AAI for the left, right, and centre trials from signal_AAI----------------------------
    cue_dirs = [1, 2, 3]
    cue_recode = ["left", "right", "centre"]

    df1['cue_dir'] = df1['cue_dir'].replace(cue_dirs, cue_recode)
    aai_df = df1[['PO8', 'PO7', 'signal_AAI', 'block_name', 'block_number', 'sample', 'cue_dir', 'chunk_n']]
    aai_df = aai_df[aai_df['block_name'].str.contains("nfb")]
    fig = px.violin(aai_df, x="block_name",  y="signal_AAI", color='cue_dir', box=True, range_y=[-1, 1], title=f"block:{block_idx}_{participant} (online AAI)")
    fig.show()

    grouped_aai_df = aai_df.groupby("block_number", as_index=False)
    median_aais = grouped_aai_df.median()
    median_aais['cue_dir'] = grouped_aai_df.first()['cue_dir']
    fig = px.violin(median_aais, x="cue_dir", y="signal_AAI", box=True, points='all', range_y=[-0.3, 0.2], title=f"block:{block_idx}_{participant} (median online AAIs)")
    fig.show()

    fig2 = go.Figure()
    print('starting...')
    for i, row in median_aais.iterrows():
        print(i)
        colour = "red"
        if row['cue_dir'] == 'left':
            colour = 'blue'
        if row['cue_dir'] == 'centre':
            colour = 'green'
        plt_df = aai_df[aai_df['block_number'] == row['block_number']]
        fig2.add_trace(go.Box(x=plt_df['block_number'], y=plt_df['signal_AAI'],
                              line=dict(color=colour),
                              name=f"{row['block_number']}-{row['cue_dir']}"))
    fig2.show()

    # Plot time course of median AAIs
    figma = px.scatter(median_aais, x='block_number', y='signal_AAI', color='cue_dir')
    figma.show()



    # Get AAI by calculating raw signals from hdf5 (i.e. no smoothing)------------------------------------
    # Drop non eeg data
    # eeg_df = df1[df1['block_name'].str.contains("nfb")]
    drop_cols = [x for x in df1.columns if x not in channels]
    drop_cols.extend(['MKIDX', 'EOG', 'ECG', 'signal_AAI'])
    eeg_data = df1.drop(columns=drop_cols)

    # Rescale the data (units are microvolts - i.e. x10^-6
    eeg_data = eeg_data * 1e-6
    aai_duration_samps = df1.shape[0]#10000
    alpha_band = (7.25, 11.25)
    chunksize = aai_df[aai_df.chunk_n > 0]['chunk_n'].median()
    mean_raw_l, std1_raw_l, pwr_raw_l = af.get_nfblab_power_stats_pandas(eeg_data[0:aai_duration_samps], fband=alpha_band, fs=fs,
                                                                 channel_labels=eeg_data.columns, chs=["PO7=1"],
                                                                 fft_samps=fs, chunksize=chunksize)

    mean_raw_r, std1_raw_r, pwr_raw_r = af.get_nfblab_power_stats_pandas(eeg_data[0:aai_duration_samps], fband=alpha_band, fs=fs,
                                                                 channel_labels=eeg_data.columns, chs=["PO8=1"],
                                                                 fft_samps=fs, chunksize=chunksize)
    aai_raw_left = (pwr_raw_l - pwr_raw_r) / (pwr_raw_l + pwr_raw_r)

    # Plot the median of the left and right alphas from the online AAI signal
    aai_df_raw = df1[['PO8', 'PO7', 'block_name', 'block_number', 'sample', 'cue_dir', 'signal_AAI']]
    aai_df_raw['raw_aai'] = aai_raw_left
    aai_df_raw['raw_aai'] = aai_df_raw['raw_aai'].rolling(window=int(fs / 10)).mean()
    aai_df_raw['raw_smoothed'] = aai_df_raw['raw_aai'].rolling(window=int(fs / 10)).mean()
    aai_df_raw['P08_pwr'] = pwr_raw_r
    aai_df_raw['P07_pwr'] = pwr_raw_l
    aai_df_raw = aai_df_raw[aai_df_raw['block_name'].str.contains("nfb")]
    grouped_aai_df_raw = aai_df_raw.groupby("block_number", as_index=False)
    median_aais_raw = grouped_aai_df_raw.median()
    median_aais_raw['cue_dir'] = grouped_aai_df_raw.first()['cue_dir']
    median_aais['raw_aai'] = median_aais_raw['raw_aai']
    fig = px.violin(median_aais_raw, x="cue_dir", y="raw_aai", box=True, points='all', range_y=[-0.3, 0.2], title=f"block:{block_idx}_{participant} (raw_ref)")
    fig.show()

    # Plot time course of median AAIs
    m_a_r = median_aais_raw.copy()
    cue_dirs = ["left", "right", "centre"]
    cue_recode = ["left_raw", "right_raw", "centre_raw"]

    m_a_r['cue_dir'] = m_a_r['cue_dir'].replace(cue_dirs, cue_recode)
    figma.add_traces(
        list(px.scatter(m_a_r, x='block_number', y='raw_aai', color='cue_dir', color_discrete_sequence=px.colors.qualitative.Pastel1).select_traces())
    )
    figma.show()

    # look at scores
    if score:
        median_aais_raw['score'] = list(score.values())

    calc_score = []
    for b_no in median_aais_raw['block_number']:
        calc_score.append(calculate_score(aai_df[aai_df['block_number'] == b_no], side=median_aais_raw[median_aais_raw['block_number'] == b_no]['cue_dir'].iloc[0], threshold=0.0))
    median_aais_raw['calc_score'] = calc_score

    print(f"mean: {median_aais_raw['calc_score'].mean()}")
    print(f"median: {median_aais_raw['calc_score'].median()}")
    print(f"std: {median_aais_raw['calc_score'].std()}")
    print(f"rng: {median_aais_raw['calc_score'].min()} - {median_aais_raw['calc_score'].max()}")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=aai_df_raw.index, y=aai_df_raw['raw_aai'],
                        mode='lines',
                        name='AAI_calc'))
    fig1.add_trace(go.Scatter(x=aai_df_raw.index, y=aai_df_raw['signal_AAI'],
                        mode='lines',
                        name='AAI_online'))
    fig1.add_trace(go.Scatter(x=aai_df_raw.index, y=aai_df_raw['raw_smoothed'],
                        mode='lines',
                        name='AAI_calc_smoothed'))
    fig1.show()


    # Plot the aai for eachblock
    # fig = px.line(aai_df_raw, x='sample', y='raw_aai', color='cue_dir').show()

    # Plot the median of the left and right alphas from the online AAI signal
    side_data_raw = pd.melt(median_aais_raw, id_vars=['cue_dir'], value_vars=['P07_pwr', 'P08_pwr'], var_name='side', value_name='data')
    figx = px.box(side_data_raw, x="cue_dir", y="data", color='side', points='all', title=f"block:{block_idx}_{participant} (raw_ref)")
    # figx.show()


    figb = go.Figure()
    figb.add_trace(go.Box(x=median_aais_raw['cue_dir'], y=median_aais_raw['raw_aai'],
                          line=dict(color='blue'),
                          name='nfb'))

    # TODO - fix this code so the names don't copy from above
    aai_df_raw = df1[['PO8', 'PO7', 'block_name', 'block_number', 'sample', 'cue_dir']]
    aai_df_raw['raw_aai'] = aai_raw_left
    aai_df_raw['P08_pwr_fc'] = pwr_raw_r
    aai_df_raw['P07_pwr_fc'] = pwr_raw_l
    aai_df_raw = aai_df_raw[aai_df_raw['block_name'].str.contains("fc")]
    grouped_aai_df_raw = aai_df_raw.groupby("block_number", as_index=False)
    median_aais_raw = grouped_aai_df_raw.median()
    median_aais_raw['cue_dir'] = grouped_aai_df_raw.first()['cue_dir']
    figb.add_trace(go.Box(y=median_aais_raw['raw_aai'],
                          line=dict(color='green'),
                          name='fc'))

    side_data_raw = pd.melt(median_aais_raw, id_vars=['cue_dir'], value_vars=['P07_pwr_fc', 'P08_pwr_fc'], var_name='side', value_name='data')
    side_data_raw['cue_dir'] = 'fc'
    figx.add_traces(
        list(px.box(side_data_raw, x='cue_dir', y="data", color='side', points='all', title=f"block:{block_idx}_{participant} (raw_ref)", color_discrete_sequence=px.colors.qualitative.Dark2).select_traces())
    )

    # TODO - fix this code so the names don't copy from above
    aai_df_raw = df1[['PO8', 'PO7', 'block_name', 'block_number', 'sample', 'cue_dir']]
    aai_df_raw['raw_aai'] = aai_raw_left
    aai_df_raw['P08_pwr_cue'] = pwr_raw_r
    aai_df_raw['P07_pwr_cue'] = pwr_raw_l
    aai_df_raw = aai_df_raw[aai_df_raw['block_name'].str.contains("cue")]
    grouped_aai_df_raw = aai_df_raw.groupby("block_number", as_index=False)
    median_aais_raw = grouped_aai_df_raw.median()
    median_aais_raw['cue_dir'] = grouped_aai_df_raw.first()['cue_dir']
    figb.add_trace(go.Box(y=median_aais_raw['raw_aai'],
                          line=dict(color='red'),
                          name='cue'))
    figb.update_layout(
        title=f"block:{block_idx}_{participant} section AAI",
        hovermode="x",
        yaxis_range=[-0.25, 0.25]
    )
    figb.show()

    # Plot left and right power for each block
    side_data_raw = pd.melt(median_aais_raw, id_vars=['cue_dir'], value_vars=['P07_pwr_cue', 'P08_pwr_cue'], var_name='side', value_name='data')
    side_data_raw['cue_dir'] = 'cue'
    figx.add_traces(
        list(px.box(side_data_raw, x='cue_dir', y="data", color='side', points='all', title=f"block:{block_idx}_{participant} (raw_ref)", color_discrete_sequence=px.colors.qualitative.Bold).select_traces())
    )
    figx.show()

    # Look at time course of AAI for best and worst blocks - how does the score correspond

    # Would magnitude of deviation above threshold make score better?

    return eeg_data, alpha_band


def epoch_analysis(eeg_data, alpha_band, block_idx, participant):
    # Get AAI by calculating signals from MNE-------------------------------
    # (try with avg and no referencing (no ref should be the same))

    # create an MNE info
    m_info = mne.create_info(ch_names=list(eeg_data.columns), sfreq=fs, ch_types=['eeg' for ch in list(eeg_data.columns)])

    # Set the montage (THIS IS FROM roi_spatial_filter.py)
    standard_montage = mne.channels.make_standard_montage(kind='standard_1020')
    standard_montage_names = [name.upper() for name in standard_montage.ch_names]
    for j, channel in enumerate(eeg_data.columns):
        try:
            # make montage names uppercase to match own data
            standard_montage.ch_names[standard_montage_names.index(channel.upper())] = channel.upper()
        except ValueError as e:
            print(f"ERROR ENCOUNTERED: {e}")
    m_info.set_montage(standard_montage, on_missing='ignore')

    # Create the mne raw object with eeg data
    m_raw = mne.io.RawArray(eeg_data.T, m_info, first_samp=0, copy='auto', verbose=None)

    # set the reference to average
    # m_raw = m_raw.set_eeg_reference(projection=False) # Be careful about using a projection vs actual data referencing in the analysis

    aai_duration_samps = eeg_data.shape[0]#10000
    raw_ref_avg_df = pd.DataFrame(m_raw.get_data(stop=aai_duration_samps).T, columns=m_raw.info.ch_names)
    mean_avg_l, std_avg_l, pwr_avg_l = af.get_nfblab_power_stats_pandas(raw_ref_avg_df, fband=alpha_band, fs=fs,
                                                                 channel_labels=m_raw.info.ch_names, chs=["PO7=1"],
                                                                 fft_samps=1000)
    mean_avg_r, std_avg_r, pwr_avg_r = af.get_nfblab_power_stats_pandas(raw_ref_avg_df, fband=alpha_band, fs=fs,
                                                                 channel_labels=m_raw.info.ch_names, chs=["PO8=1"],
                                                                 fft_samps=1000)
    aai_avg_left = (pwr_avg_l - pwr_avg_r) / (pwr_avg_l + pwr_avg_r)
    # fig1.add_trace(go.Scatter(x=eeg_data.index, y=aai_avg_left,
    #                     mode='lines',
    #                     name='AAI_avg'))
    # fig1.show()

    # Plot the median of the left and right alphas from the online AAI signal
    df1['raw_aai'] = aai_avg_left
    aai_df_avg = df1[['PO8', 'PO7', 'block_name', 'block_number', 'sample', 'cue_dir', 'raw_aai']]
    aai_df_avg = aai_df_avg[aai_df_avg['block_name'].str.contains("nfb")]
    grouped_aai_df_avg = aai_df_avg.groupby("block_number", as_index=False)
    median_aais_avg = grouped_aai_df_avg.median()
    median_aais_avg['cue_dir'] = grouped_aai_df_avg.first()['cue_dir']
    fig = px.violin(median_aais_avg, x="cue_dir", y="raw_aai", box=True, points='all', range_y=[-0.3, 0.2], title=f"block:{block_idx}_{participant} (avg_ref)")
    fig.show()
    # Plot the median of the left and right alphas from the online AAI signal
    side_data_avg = pd.melt(median_aais_avg, id_vars=['cue_dir'], value_vars=['PO7', 'PO8'], var_name='side', value_name='data')
    fig = px.box(side_data_avg, x="cue_dir", y="data", color='side', points='all', title=f"block:{block_idx}_{participant} (avg_ref)")
    fig.show()


    # TODO:
    #   Just use the already calculated AAI power for the whole experiment, Add a 'type' column and have '<x>_nfb', cue, fc'
    conditions = [
        (df1['cue_dir'] == 1) & (df1['block_name'] == "nfb"),
        (df1['cue_dir'] == 2) & (df1['block_name'] == "nfb"),
        (df1['cue_dir'] == 3) & (df1['block_name'] == "nfb"),
        (df1['block_name'] == "fc"),
        (df1['block_name'] == "cue")
        ]

    # create a list of the values we want to assign for each condition
    values = ['left_nfb', 'right_nfb', 'centre_nfb', 'fc', 'cue']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df1['block_type'] = np.select(conditions, values)

    # Plot the different conditions
    block_type_df = df1[['PO7', 'PO8', 'block_name', 'block_number', 'sample', 'cue_dir', 'raw_aai', 'block_type']]
    grouped_block_type_df = block_type_df.groupby("block_number", as_index=False)
    median_block_type = grouped_block_type_df.median(numeric_only=False)
    median_block_type['block_type'] = grouped_block_type_df.first()['block_type']
    fig = px.violin(median_block_type, x="block_type", y="raw_aai", box=True, points='all', range_y=[-0.3, 0.2], title=f"block:{block_idx}_{participant} (avg_ref)")
    fig.show()
    fig = px.box(median_block_type, x="block_type", y="raw_aai", points='all', range_y=[-0.3, 0.2], title=f"block:{block_idx}_{participant} (avg_ref)")
    fig.show()
    # Plot the median of the left and right alphas from the online AAI signal
    block_type_side_data = pd.melt(median_block_type, id_vars=['block_type'], value_vars=['PO7', 'PO8'], var_name='side', value_name='data')
    fig = px.box(block_type_side_data, x="block_type", y="data", color='side', points='all', title=f"block:{block_idx}_{participant} (avg_ref)")
    fig.show()



    # Do the epoching and Add baseline and cue AAIs to the plot-----------------
    # TODO:
    #   2. look at epochs for all sections
    #   3. plot against cues and fixation crosses

    # Create the stim channel
    # --- Get the probe events and MNE raw objects
    # Get start of blocks as different types of epochs (1=start, 2=right, 3=left, 4=centre)
    df1['protocol_change'] = df1['block_number'].diff()
    df1['choice_events'] = df1.apply(lambda row: row.protocol_change if row.block_name == "start" else
                                     row.protocol_change * 2 if row.cue_dir == "left" else
                                     row.protocol_change * 3 if row.cue_dir == "right" else
                                     row.protocol_change * 4 if row.cue_dir == "centre" else 0, axis=1)

    # Create the events list for the protocol transitions
    probe_events = df1[['choice_events']].to_numpy()
    left_probe= 2
    right_probe = 3
    centre_probe = 4
    event_dict = {'right_probe': right_probe, 'left_probe': left_probe, 'centre_probe': centre_probe}
    info = mne.create_info(['STI'], m_raw.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(probe_events.T, info)
    m_raw.add_channels([stim_raw], force_update_info=True)

    # Get the epoch object
    m_filt = m_raw.copy()
    m_filt.filter(8, 14, n_jobs=1,  # use more jobs to speed up.
                   l_trans_bandwidth=1,  # make sure filter params are the same
                   h_trans_bandwidth=1)  # in each band and skip "auto" option.

    events = mne.find_events(m_raw, stim_channel='STI')
    reject_criteria = dict(eeg=100e-6)

    left_chs = ['PO7=1']
    right_chs = ['PO8=1']

    epochs = mne.Epochs(m_filt, events, event_id=event_dict, tmin=-2, tmax=10, baseline=None,
                        preload=True, detrend=1)#, reject=reject_criteria)
    # epochs.drop([19,22,27,32]) # Drop bads for K's 1st dataset

    # fig = epochs.plot(events=events)

    probe_left = epochs['left_probe'].average()
    probe_right = epochs['right_probe'].average()

    dataframes = []
    # ----Look at the power for the epochs in the left and right channels for left and right probes
    e_mean1, e_std1, epoch_pwr1 = af.get_nfb_epoch_power_stats(epochs['left_probe'], fband=alpha_band, fs=fs, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
    e_mean2, e_std2, epoch_pwr2 = af.get_nfb_epoch_power_stats(epochs['left_probe'], fband=alpha_band, fs=fs, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])

    df_i = pd.DataFrame(dict(left=e_mean1, right=e_mean2))
    df_i['section'] = f"left_probe"
    dataframes.append(df_i)

    dataframes_aai_cue = []
    dataframes_aai = []
    aai_nfb_left = (epoch_pwr1 - epoch_pwr2) / (epoch_pwr1 + epoch_pwr2)
    df_ix = pd.DataFrame(dict(aai=aai_nfb_left.mean(axis=0)[0]))
    df_ix['probe'] = f"left"
    dataframes_aai.append(df_ix[2000:9000])
    dataframes_aai_cue.append(df_ix[0:2000])
    fig2 = go.Figure()
    af.plot_nfb_epoch_stats(fig2, aai_nfb_left.mean(axis=0)[0], aai_nfb_left.std(axis=0)[0], name=f"left probe aai",
                         title=f"left probe aai",
                         color=(230, 20, 20, 1), y_range=[-1, 1])


    e_mean1, e_std1, epoch_pwr1 = af.get_nfb_epoch_power_stats(epochs['right_probe'], fband=alpha_band, fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
    e_mean2, e_std2, epoch_pwr2 = af.get_nfb_epoch_power_stats(epochs['right_probe'], fband=alpha_band, fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])

    df_i = pd.DataFrame(dict(left=e_mean1, right=e_mean2))
    df_i['section'] = f"right_probe"
    dataframes.append(df_i)

    aai_nfb_right = (epoch_pwr1 - epoch_pwr2) / (epoch_pwr1 + epoch_pwr2)
    df_ix = pd.DataFrame(dict(aai=aai_nfb_right.mean(axis=0)[0]))
    df_ix['probe'] = f"right"
    dataframes_aai.append(df_ix[2000:9000])
    dataframes_aai_cue.append(df_ix[0:2000])
    af.plot_nfb_epoch_stats(fig2, aai_nfb_right.mean(axis=0)[0], aai_nfb_right.std(axis=0)[0], name=f"right probe aai",
                         title=f"mean aai time course",
                         color=(20, 20, 230, 1), y_range=[-1, 1])


    e_mean1, e_std1, epoch_pwr1 = af.get_nfb_epoch_power_stats(epochs['centre_probe'], fband=alpha_band, fs=1000, channel_labels=epochs.info.ch_names, chs=["PO7=1"])
    e_mean2, e_std2, epoch_pwr2 = af.get_nfb_epoch_power_stats(epochs['centre_probe'], fband=alpha_band, fs=1000, channel_labels=epochs.info.ch_names,  chs=["PO8=1"])

    df_i = pd.DataFrame(dict(left=e_mean1, right=e_mean2))
    df_i['section'] = f"centre_probe"
    dataframes.append(df_i)

    aai_nfb_centre = (epoch_pwr1 - epoch_pwr2) / (epoch_pwr1 + epoch_pwr2)
    df_ix = pd.DataFrame(dict(aai=aai_nfb_centre.mean(axis=0)[0]))
    df_ix['probe'] = f"centre"
    dataframes_aai.append(df_ix[2000:9000])
    dataframes_aai_cue.append(df_ix[0:2000])
    af.plot_nfb_epoch_stats(fig2, aai_nfb_centre.mean(axis=0)[0], aai_nfb_centre.std(axis=0)[0], name=f"centre probe aai",
                         title=f"mean aai time course",
                         color=(20, 230, 20, 1), y_range=[-1, 1])
    fig2.add_vline(x=2000, annotation_text="cvsa start")
    fig2.show()


def plot_best_vs_worst_nfb_aai(df1, worst=0, best=0):
    """
    Plot the best vs the worst nfb aais
    """
    aai_worst = df1[(df1['block_number'] == worst) | (df1['block_number'] == worst - 1)]['signal_AAI'].reset_index(
        drop=True)
    aai_best = df1[(df1['block_number'] == best) | (df1['block_number'] == best - 1)]['signal_AAI'].reset_index(
        drop=True)
    # Best is a right condition so flip the plot
    aai_best = aai_best * -1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=aai_worst.index, y=aai_worst,
                             mode='lines',
                             name=f'worst (trial_{worst})'))
    fig.add_trace(go.Scatter(x=aai_best.index, y=aai_best,
                             mode='lines',
                             name=f'best (trial_{best})'))
    fig.add_hline(np.mean(aai_best), annotation_text="best_mean")
    fig.add_hline(np.mean(aai_worst), annotation_text="worst_mean")
    fig.add_vline(np.mean(df1[(df1['block_number'] == worst - 1)].shape[0]), annotation_text="worst_start")
    fig.add_vline(np.mean(df1[(df1['block_number'] == best - 1)].shape[0]), annotation_text="best_start")
    fig.show()

def psychopy_rt(csvfile):
    """
    plot rt data from psychopy posner task
    """
    df_rt = pd.read_csv(csvfile)
    df_rt = df_rt[['cue', 'stim_side', 'key_resp.rt']]

    conditions = [
        df_rt['stim_side'].eq(10) & df_rt['cue'].isin([1,2]),
        df_rt['stim_side'].eq(11) & df_rt['cue'].isin([1,2]),
        df_rt['cue'].isin([3]),
    ]

    choices = [True, False, 'N/A']

    df_rt['valid_cue'] = np.select(conditions, choices, default=0)

    df_rt = df_rt.iloc[1:, :] # remove the first row
    df_rt = df_rt.reset_index(drop=True)
    df_rt['key_resp.rt'] = df_rt['key_resp.rt'].apply(ast.literal_eval) # Convert to list vals
    df_rt['key_resp.rt'] = df_rt["key_resp.rt"].apply(lambda x: x[0])
    px.violin(df_rt, x="valid_cue", y="key_resp.rt", box=True, points='all', title=f"block:").show()#, range_y=[200, 800]).show()
    print('done')

def calculate_score(block_data, side, fs=1000, threshold=0.0):
    nfb_duration = block_data.shape[0]
    threshold_extra = 0.2
    # rate_of_increase = 0.25
    # max_reward = round(nfb_duration / fs / rate_of_increase)
    # fb_score =
    # self.percent_score = round((self.fb_score / max_reward) * 100)
    reward_factor = 1
    if side == "right":
        reward_factor = -1
    pos_points = block_data[(block_data['signal_AAI']) * reward_factor > (threshold + threshold_extra)].shape[0]
    return int((pos_points/nfb_duration)*100)

if __name__ == "__main__":
    # TODO:
    # get all NFB and sham data files in participant directory
    if platform.system() == "Windows":
        userdir = "2354158T"
    else:
        userdir = "christopherturner"
    participant_id = 'PO2'
    participant_dir = os.path.join(os.sep, "Users", userdir,"Documents","EEG_Data","pilot2_COPY",participant_id)
    nfb_data_dirs = [x for x in os.listdir(participant_dir) if '0-nfb' in x]
    sham_data_dirs = [x for x in os.listdir(participant_dir) if '1-nfb' in x]
    # Look at RTs and AAI stuff for each section in NFB and SHAM
    nfb_data_dfs = []
    for idx, data_dir in enumerate(nfb_data_dirs):
        h5file = os.path.join(participant_dir, data_dir, "experiment_data.h5")
        # h5file = "/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-test_task_test3_05-11_15-31-24/experiment_data.h5" # Test case with negative RT
        # h5file = "/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-test_task_test3_05-11_16-19-51/experiment_data.h5" # Test case with negative RT

        #-----------------------
        # MAC - no online posner
        # h5file = "/Users/christopherturner/Documents/EEG_Data/mac_testing_20220527/0-nfb_task_posner_test_mac_off_05-27_17-15-46/experiment_data.h5" # Correct dir
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/mac_testing_20220527/0-nfb_task_posner_test_mac_off_05-27_17-15-46/05-27_17-15-46.log") # TODO: automatically get this file

        # h5file = "/Users/christopherturner/Documents/EEG_Data/mac_testing_20220527/0-nfb_task_posner_test_mac_off_05-27_17-27-36/experiment_data.h5" # Opposite dir
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/mac_testing_20220527/0-nfb_task_posner_test_mac_off_05-27_17-27-36/05-27_17-27-36.log")

        # h5file = "/Users/christopherturner/Documents/EEG_Data/mac_testing_20220527/0-nfb_task_posner_test_mac_off_05-27_17-55-50/experiment_data.h5" # Doing nothing
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/mac_testing_20220527/0-nfb_task_posner_test_mac_off_05-27_17-55-50/05-27_17-55-50.log")

        # h5file = "/Users/christopherturner/Documents/EEG_Data/mac_testing_20220527/0-posner_task_posner_test_mac_off_05-27_16-53-51/experiment_data.h5" # posner

        #-----------------------
        # LINUX - with online posner
        # h5file = "/Users/christopherturner/Documents/EEG_Data/posner_testing_20220526/0-nfb_task_posner_test_on_05-26_16-57-32/experiment_data.h5" #correct dir
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/posner_testing_20220526/0-nfb_task_posner_test_on_05-26_16-57-32/05-26_16-57-32.log")

        # h5file = "/Users/christopherturner/Documents/EEG_Data/posner_testing_20220526/0-nfb_task_posner_test_on_05-26_17-12-29/experiment_data.h5" # opposite dir
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/posner_testing_20220526/0-nfb_task_posner_test_on_05-26_17-12-29/05-26_17-12-29.log")

        # h5file = "/Users/christopherturner/Documents/EEG_Data/posner_testing_20220526/0-nfb_task_posner_test_on_05-26_17-26-10/experiment_data.h5" # Do nothing
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/posner_testing_20220526/0-nfb_task_posner_test_on_05-26_17-26-10/05-26_17-26-10.log")

        # h5file = "/Users/christopherturner/Documents/EEG_Data/posner_testing_20220526/0-nfb_task_posner_test_on_05-26_17-46-51/experiment_data.h5" #correct dir
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/posner_testing_20220526/0-nfb_task_posner_test_on_05-26_17-46-51/05-26_17-46-51.log")

        # h5file = "/Users/christopherturner/Documents/EEG_Data/posner_testing_20220526/0-posner_task_posner_test_on_05-26_16-26-58/experiment_data.h5"

        #-----------------------
        # RT Testing
        # h5file = "/Users/christopherturner/Documents/EEG_Data/rt_testing_20220530/0-posner_task_posner_test_mac_off_05-30_12-25-51_mac_extkb_100hz_full/experiment_data.h5"
        # h5file = "/Users/christopherturner/Documents/EEG_Data/rt_testing_20220530/0-posner_task_posner_test_mac_off_05-30_11-11-45_mac_intkb_1000hz/experiment_data.h5"

        #-----------------------
        # AAI Testing
        h5file = "/Users/christopherturner/Documents/EEG_Data/aai_testing_20220601/0-nfb_task_PO0_1_06-01_17-27-12/experiment_data.h5" # correct dir
        score = read_log_file("/Users/christopherturner/Documents/EEG_Data/aai_testing_20220601/0-nfb_task_PO0_1_06-01_17-27-12/06-01_17-27-12.log") # correct dir

        # h5file = "/Users/christopherturner/Documents/EEG_Data/aai_testing_20220601/0-nfb_task_PO0_1_06-01_18-15-17/experiment_data.h5" # pattern (LLRLRRRLRL)
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/aai_testing_20220601/0-nfb_task_PO0_1_06-01_18-15-17/06-01_18-15-17.log") # pattern

        # h5file = "/Users/christopherturner/Documents/EEG_Data/aai_testing_20220601/0-nfb_task_PO0_1_06-01_17-51-16/experiment_data.h5" # do nothing
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/aai_testing_20220601/0-nfb_task_PO0_1_06-01_17-51-16/06-01_17-51-16.log") # do nothing

        # h5file = "/Users/christopherturner/Documents/EEG_Data/aai_testing_20220601/0-nfb_task_PO0_1_06-01_18-23-46/experiment_data.h5" #wrong dir
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/aai_testing_20220601/0-nfb_task_PO0_1_06-01_18-23-46/06-01_18-23-46.log") # wrong dir


        # h5file = "/Users/christopherturner/Documents/EEG_Data/aai_testing_20220601/0-posner_task_PO0_06-01_17-02-59/experiment_data.h5"
        # score = {}

        #-----------------------
        # EPRIME PC TESTING
        psychopy_csv = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/posner/1_posner_2022-06-14_16h54.47.522.csv"

        psychopy_csv = "/Users/2354158T/Downloads/2_posner_2022-07-08_16h30.30.493.csv"
        psychopy_rt(psychopy_csv)

        baseline_h5file = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-baseline_test_psychopy_06-14_16-35-47/experiment_data.h5"

        # h5file = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-13-14/experiment_data.h5" # Correct direction
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-13-14/06-14_17-13-14.log")

        # h5file = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-41-40/experiment_data.h5" # wrong direction - left is normally lower here (looks like working)
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-41-40/06-14_17-41-40.log")

        # h5file = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-27-31/experiment_data.h5" # pattern
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-27-31/06-14_17-27-31.log")
        #
        # h5file = "/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-53-44/experiment_data.h5" # nothing
        # score = read_log_file("/Users/christopherturner/Documents/EEG_Data/testing_20220614/0-nfb_task_test_psychopy_06-14_17-53-44/06-14_17-53-44.log")


        #-----------------------
        # DECENT LOOKING PILOT DATA
        h5file = "/Users/christopherturner/Documents/EEG_Data/pilot2_COPY/PO5/0-nfb_task_PO2_05-10_11-18-09/experiment_data.h5"
        score = None

        #-----------------------
        # h1 = df1.groupby("block_number").first()
        # h1 = h1[h1.block_name == 'nfb']
        # px.scatter(h1, x='sample', y='reaction_time').show()

        df1, fs, channels, p_names = load_data(h5file)
        df1 = get_cue_dir(df1, channels=channels)
        df1 = get_posner_time(df1)
        # TODO - resampling to make everything faster?
        # plot_best_vs_worst_nfb_aai(df1, worst=58, best=43)
        df1["block_number"] = df1["block_number"] + idx * 100

        # Get the baseline dataframe
        df1_bl, fs_bl, channels_bl, p_names_bl = load_data(baseline_h5file)

        eeg_data, alpha_band = cvsa_analysis(df1, fs, channels, p_names, block_idx=f"NFB_{idx}", participant=participant_id, score=score, df1_bl=df1_bl)
        # epoch_analysis(eeg_data, alpha_band, f"NFB_{idx}", participant_id)
        nfb_data_dfs.append(df1)

        # dd = df1.groupby('block_number').first()
        # dd[dd.block_name == 'nfb']

        # block_lengths = {}
        # for block_no in df1[df1.posner_time > 0]['block_number'].to_list():
        #     block = df1[df1.block_number == block_no]
        #     block_lengths[block_no] = block.shape[0]

    sham_data_dfs = []
    for idx, data_dir in enumerate(sham_data_dirs):
        h5file = os.path.join(participant_dir, data_dir, "experiment_data.h5")
        df1, fs, channels, p_names = load_data(h5file)
        df1 = get_cue_dir(df1)
        df1 = get_posner_time(df1)
        df1["block_number"] = df1["block_number"] + idx * 100
        cvsa_analysis(df1, fs, channels, p_names,  f"SHAM_{idx}")
        sham_data_dfs.append(df1)

    # Add all NFB together and all SHAm and look at RTs and AAIs in total
    if nfb_data_dirs:
        nfb_data_all = pd.concat(nfb_data_dfs)
        nfb_data_all = get_cue_dir(nfb_data_all)
        nfb_data_all = get_posner_time(nfb_data_all)
        cvsa_analysis(nfb_data_all, fs, channels, p_names, "NFB_ALL")

    if sham_data_dirs:
        sham_data_all = pd.concat(sham_data_dfs)
        sham_data_all = get_cue_dir(sham_data_all)
        sham_data_all = get_posner_time(sham_data_all)
        cvsa_analysis(sham_data_all, fs, channels, p_names, "SHAM_ALL")

    # Compare between NFB and SHAM
    # Compare between 1st and 2nd session

    # TODO: Add the baseline to the AAI plots
    # TODO: do the combination / comparison also for all participants combined (loop over all participant dirs)