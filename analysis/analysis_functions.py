"""
Analysis functions for the free viewing task
"""
import os

import mne
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

from pynfb.serializers import read_spatial_filter


# ------Define analysis functions
from philistine.mne import write_raw_brainvision

from pynfb.signal_processing.filters import ExponentialSmoother, FFTBandEnvelopeDetector
from utils.load_results import load_data


def get_protocol_data(task_data, channels, p_names, eog_filt=True, out="dict"):
    """
    return a dictionary containing data for each protocol
    """

    # Extract the individual protocols
    protocol_data = {}
    block_numbers = task_data['block_number'].unique()
    protocol_names = [f"{a_}{b_}" for a_, b_ in zip(p_names, block_numbers)]
    channels_signal = channels.copy()
    channels_signal.append("signal_AAI")
    if eog_filt:
        channels_signal.append("EOG_FILTERED")
        channels_signal.append("ECG_FILTERED")
    df2 = pd.melt(task_data, id_vars=['block_name', 'block_number', 'sample', 'choice', 'probe', 'answer', 'reward'],
                  value_vars=channels_signal, var_name="channel", value_name='data')
    if out == "df":
        return df2

    for protocol_n in block_numbers:
        protocol_data[protocol_names[protocol_n - 1]] = df2.loc[df2['block_number'] == protocol_n]


    return protocol_data


def get_task_fixation_bias(protocol_data):
    """
    Calculate fixation bias for entire task
    This is the median of ratio of right biases (left biases is 1 - right bias) for each trial
    """
    # TODO: fix the plotting so that can plot all image fixations on the one plot
    trail_offset = 0
    trial_fb = {}
    for protocol, data in protocol_data.items():
        if "fix_cross" in protocol.lower():
            # Recalculate the centre point
            # TODO: MAKE SURE LEFT AND RIGHT ARE CORRECT
            eog_right = data.loc[data['channel'] == "EOG_FILTERED"]['data'].reset_index(drop=True)
            eog_left = data.loc[data['channel'] == "ECG_FILTERED"]['data'].reset_index(drop=True)
            eog_signal = pd.DataFrame({"EOG_SIGNAL": eog_left - eog_right})
            # just use the last 1 second to remove the initial saccade
            eog_signal = eog_signal.iloc[1000:-1, :]
            trail_offset = eog_signal.median()[0]
            # fig = px.line(eog_signal, y=eog_signal["EOG_SIGNAL"])
            # fig.show()
            # (if needed) recalculate the scale to fit the screen (this would be for normalising the fb between 0 and 1)
            pass
        elif "image" in protocol.lower():
            # subtract previous offset (centre point) from data
            eog_right = data.loc[data['channel'] == "EOG_FILTERED"]['data'].reset_index(drop=True)
            eog_left = data.loc[data['channel'] == "ECG_FILTERED"]['data'].reset_index(drop=True)
            eog_signal = pd.DataFrame({"EOG_SIGNAL": eog_left - eog_right}) - trail_offset
            # fig = px.line(eog_signal, y=eog_signal["EOG_SIGNAL"])
            # fig.show()

            # Calculate fixation bias - this is the ratio of number of leftward vs rightward eye locations (average is biased by looking to the edges of the screen)
            fb_data = {}
            right_fx_number = eog_signal.agg(lambda x: sum(x > 0)).sum()  # Right is greater than 0
            left_fx_number = eog_signal.agg(lambda x: sum(x < 0)).sum()
            fb_data['ratio'] = right_fx_number / 5000  # TODO: figure out if this is ok
            fb_data['median'] = eog_signal.median()[0]
            trial_fb[protocol] = fb_data
            pass

    # Get median FB over all trials
    trail_fb = pd.DataFrame(trial_fb).T
    return trail_fb['ratio'].median(), trail_fb['median'].median()


def get_eog_calibration(protocol_data):
    # get filtered eog data
    eog_data = protocol_data['EyeCalib2'].loc[
        protocol_data['EyeCalib2']['channel'].isin(["ECG_FILTERED", "EOG_FILTERED"])]

    # - CALIBRATION -
    # get samples for onset of each calibration stage (probe: left=10, right=11, top=12, bottom=13, cross=14)
    eog_data['probe_change'] = eog_data['probe'].diff()
    calibration_samples = eog_data[eog_data['probe_change'] != 0]
    calibration_samples = calibration_samples.loc[protocol_data['EyeCalib2']['channel'].isin(["EOG_FILTERED"])][
        ['sample', 'probe']]
    calibration_samples['probe'] = calibration_samples['probe'].replace(
        {14: 'cross', 10: 'left', 11: 'right', 12: 'top', 13: 'bottom'})

    # fig = px.line(eog_data, x="sample", y="data", color='channel')
    # for index, row in calibration_samples.iterrows():
    #     fig.add_vline(x=row['sample'], line_dash="dot",
    #                   annotation_text=row['probe'],
    #                   annotation_position="bottom right")
    # fig.show()

    # Get the offsets for each calibration point
    calibration_delay = 400  # 500ms to react to probe # TODO: find a way to automate this
    previous_offset = 0
    calibration_offsets_ecg = {}
    calibration_offsets_eog = {}
    for idx in range(len(calibration_samples)):
        offset = calibration_samples.iloc[idx].loc['sample']
        type = calibration_samples.iloc[idx].loc['probe']
        if idx == 0:
            previous_offset = offset
            previous_type = type
        else:
            second_EOG = eog_data[
                eog_data['sample'].between(previous_offset + calibration_delay, offset + calibration_delay,
                                           inclusive="neither")]
            calibration_offsets_ecg[previous_type] = second_EOG.loc[second_EOG['channel'] == "EOG_FILTERED"][
                'data'].median()
            calibration_offsets_eog[previous_type] = second_EOG.loc[second_EOG['channel'] == "ECG_FILTERED"][
                'data'].median()
            previous_offset = offset
            previous_type = type

        if idx == len(calibration_samples) - 1:
            type = 'cross2'
            second_EOG = eog_data[eog_data['sample'].between(offset + calibration_delay, eog_data['sample'].iloc[-1],
                                                             inclusive="neither")]
            calibration_offsets_ecg[type] = second_EOG.loc[second_EOG['channel'] == "EOG_FILTERED"]['data'].median()
            calibration_offsets_eog[type] = second_EOG.loc[second_EOG['channel'] == "ECG_FILTERED"]['data'].median()

    # TODO: do the above but with MNE (to try get automatic eog events)

    # Plot the filtered calibration signal and the medians
    fig = px.line(eog_data, x="sample", y=eog_data["data"], color='channel')
    for index, row in calibration_samples.iterrows():
        fig.add_vline(x=row['sample'] + calibration_delay, line_dash="dot",
                      annotation_text=row['probe'],
                      annotation_position="bottom right")
    for type, value in calibration_offsets_eog.items():
        fig.add_hline(y=value, line_color='red',
                      annotation_text=f"{type}: MEDIAN",
                      annotation_position="bottom right")
    # for type, value in calibration_offsets_ecg.items():
    #     fig.add_hline(y=value, line_color='blue',
    #                       annotation_text=f"{type}: MEDIAN",
    #                       annotation_position="bottom right")
    fig.show()

    # Actual signal is the difference between the two electrodes
    eog_right = eog_data.loc[eog_data['channel'] == "EOG_FILTERED"]['data'].reset_index(drop=True)
    eog_left = eog_data.loc[eog_data['channel'] == "ECG_FILTERED"]['data'].reset_index(drop=True)
    eog_signal = pd.DataFrame({"EOG_SIGNAL": eog_left - eog_right})

    fig = px.line(eog_signal, y=eog_signal["EOG_SIGNAL"])
    fig.show()
    pass

def free_view_analysis():
    pass
    # do t-test for all delta_fb for scalp compared to sham
    # do a t-test for all delta_fb for source compared to sham

    # Get average delta_fb for scalp, source, and sham for all participants (OR T-STATISTIC?)
    # do permutation test for difference in above averages (or t-stats?) for scalp->sham and source->sham and scalp->source


def convert_hdf5_to_bv(h5file, output_file="output-bv.vhdr"):

    # Put data in pandas data frame
    df1, fs, channels, p_names = load_data(h5file)
    df1['sample'] = df1.index

    # Drop non eeg data
    drop_cols = [x for x in df1.columns if x not in channels]
    drop_cols.extend(['MKIDX', 'EOG', 'ECG'])
    eeg_data = df1.drop(columns=drop_cols)

    # Rescale the data (units are microvolts - i.e. x10^-6
    eeg_data = eeg_data * 1e-6

    # create an MNE info
    m_info = mne.create_info(ch_names=list(eeg_data.columns), sfreq=fs,
                             ch_types=['eeg' for ch in list(eeg_data.columns)])
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
    brainvision_file = os.path.join(os.getcwd(), output_file)
    write_raw_brainvision(m_raw, brainvision_file, events=True)


def get_nfb_derived_sig(eeg_data, pick_chs, fs, channel_labels, signal_estimator):
    spatial_matrix = read_spatial_filter(pick_chs, fs, channel_labels=channel_labels)
    chunksize = 20
    filtered_data = np.empty(0)
    for k, chunk in eeg_data.groupby(np.arange(len(eeg_data)) // chunksize):
        filtered_chunk = np.dot(chunk, spatial_matrix)
        current_chunk = signal_estimator.apply(filtered_chunk)
        filtered_data = np.append(filtered_data, current_chunk)
    return filtered_data


def get_nfb_derived_sig_epoch(epochs, pick_chs, fs, channel_labels, signal_estimator):
    """
    Assume that the data is already filtered with appropriate channels?
    """
    spatial_matrix = read_spatial_filter(pick_chs, fs, channel_labels=channel_labels)
    chunksize = 20
    filtered_data = np.empty(0)
    for chunk in np.array_split(epochs,round(len(epochs[1])/chunksize),axis=1):
        filtered_chunk = np.dot(chunk.T, spatial_matrix)
        current_chunk = signal_estimator.apply(filtered_chunk)
        filtered_data = np.append(filtered_data, current_chunk)
    return filtered_data


def get_nfb_epoch_power_stats(epochs, fband=(8, 14), fs=1000,channel_labels=None, chs=None):
    """
    TODO: refactor (seems i'm returning epoch_pwr more than once with mean and std
    """
    smoothing_factor = 0.7
    smoother = ExponentialSmoother(smoothing_factor)
    n_samples = 1000
    signal_estimator = FFTBandEnvelopeDetector(fband, fs, smoother, n_samples)
    epoch_pwr = np.ndarray((epochs.get_data().shape[0], len(chs), epochs.get_data().shape[2]))
    pick_chs_string = ";".join(chs)
    for idx, epoch in enumerate(epochs.get_data()):
        epoch_pwr[idx] = get_nfb_derived_sig_epoch(epoch, pick_chs_string, fs, channel_labels, signal_estimator)
    epoch_pwr_mean = epoch_pwr.mean(axis=0)[0]
    epoch_pwr_std = epoch_pwr.std(axis=0)[0]
    return epoch_pwr_mean, epoch_pwr_std, epoch_pwr


# TODO: add capability to plot epoch start time etc (verticle lines)
def plot_nfb_epoch_stats(fig, e_mean1, e_std1, name="case1", title="epoch power", color=(255,0,0,1), y_range=None):
    fig.add_trace(
        go.Scatter(
            name=name,
            y=e_mean1,
            mode='lines',
            line=dict(color=f'rgb({color[0]}, {color[1]}, {color[2]})'),
        ))
    fig.add_trace(go.Scatter(
            name='Upper Bound',
            y=e_mean1 + e_std1,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ))
    color = list(color)
    color[3] = 0.3
    fig.add_trace(go.Scatter(
            name='Lower Bound',
            y=e_mean1 - e_std1,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor=f'rgba({color[0]}, {color[1]}, {color[2]}, {color[3]})',
            fill='tonexty',
            showlegend=False
        ))
    fig.update_layout(
        yaxis_title='epoch mean power',
        title=title,
        hovermode="x",
        yaxis_range=y_range
    )
    return fig


def hdf5_to_mne(h5file):
    # Put data in pandas data frame
    df1, fs, channels, p_names = load_data(h5file)
    df1['sample'] = df1.index

    # Drop non eeg data
    drop_cols = [x for x in df1.columns if x not in channels]
    drop_cols.extend(['MKIDX'])
    eeg_data = df1.drop(columns=drop_cols)

    # Rescale the data (units are microvolts - i.e. x10^-6
    eeg_data = eeg_data * 1e-6

    # create an MNE info - set types appropriately
    m_info = mne.create_info(ch_names=list(eeg_data.columns), sfreq=fs,
                             ch_types=['eeg' if ch not in ['ECG', 'EOG'] else 'eog' for ch in list(eeg_data.columns)])

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
    m_raw.set_eeg_reference(projection=True)
    return df1, m_raw

def get_nfb_protocol_change_events(df1, m_raw):
    # ----------------------------------------
    # Get nfb trials as epochs from hdf5 data
    # ----------------------------------------
    df1['protocol_change'] = df1['block_number'].diff()
    df1['p_change_events'] = df1.apply(lambda row: row.protocol_change if row.block_name == "NFB" else
    row.protocol_change * 2 if row.block_name == "fc_w" else
    row.protocol_change * 3 if row.block_name == "fc_b" else
    row.protocol_change * 4 if row.block_name == "delay" else
    row.protocol_change * 5 if row.block_name == "Input" else 0, axis=1)

    # Create the events list for the protocol transitions
    probe_events = df1[['p_change_events']].to_numpy()
    event_dict = {'nfb': 1, 'fc_w': 2, 'fc_b': 3, 'delay': 4, 'Input': 5}

    # Create the stim channel
    info = mne.create_info(['STI'], m_raw.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(probe_events.T, info)
    m_raw.add_channels([stim_raw], force_update_info=True)

if __name__ == "__main__":

    h5file = "/Users/christopherturner/Documents/EEG_Data/pilot_202201/kk/scalp/0-pre_task_kk_01-27_18-27-31/experiment_data.h5"
    convert_hdf5_to_bv(h5file, "cap60.vhdr")