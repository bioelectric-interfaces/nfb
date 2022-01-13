"""
Analysis functions for the free viewing task
"""
import pandas as pd
import numpy as np
import plotly.express as px



# ------Define analysis functions

def get_protocol_data(task_data, channels, p_names):
    """
    return a dictionary containing data for each protocol
    """

    # Extract the individual protocols
    protocol_data = {}
    block_numbers = task_data['block_number'].unique()
    protocol_names = [f"{a_}{b_}" for a_, b_ in zip(p_names, block_numbers)]
    channels_signal = channels.copy()
    channels_signal.append("signal_AAI")
    channels_signal.append("EOG_FILTERED")
    channels_signal.append("ECG_FILTERED")
    df2 = pd.melt(task_data, id_vars=['block_name', 'block_number', 'sample', 'choice', 'probe', 'answer'],
                  value_vars=channels_signal, var_name="channel", value_name='data')

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