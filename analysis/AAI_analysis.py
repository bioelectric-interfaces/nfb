"""
Script to analyse the alpha asymmetry index
"""
import platform

import matplotlib.pyplot as plt
import numpy as np
from pynfb.serializers.hdf5 import load_h5py_all_samples, load_h5py_protocol_signals, load_h5py_protocols_raw, load_h5py
from utils.load_results import load_data
import os
import glob
import pandas as pd
import plotly.express as px
from scipy.signal import butter, lfilter, freqz
import mne

import analysis_functions as af

if platform.system() == "Windows":
    userdir = "2354158T"
else:
    userdir = "christopherturner"
# ------ Get data files
data_directory = f"/Users/{userdir}/Documents/EEG_Data/pilot_202201" # This is the directory where all participants are in

# get participants
participants = next(os.walk(data_directory))[1]

# Get scalp, sham, source data for each participant
experiment_dirs = {}
for p in participants:
# TODO: fix data file structure (when have a solid test setup) - maybe include 'sc', 'so', 'sh' in the data directory names
#       and allocate this way - this way don't have to sort into separate folders. - if do 2 tasks per session, then also don't have to copy
    experiment_dirs[p] = {}
    experiment_dirs[p]["scalp"] = next(os.walk(os.path.join(data_directory, p, "scalp")))[1]
    experiment_dirs[p]["source"] = next(os.walk(os.path.join(data_directory, p, "source")))[1]
    experiment_dirs[p]["sham"] = next(os.walk(os.path.join(data_directory, p, "sham")))[1]

experiment_data = []
for participant, participant_dirs in experiment_dirs.items():
    participant_data = {"participant_id": participant, "session_data": []}
    print(f"processing...")
    print(f"participant: {participant}")
    if participant:# == 'sh':
        for session, session_dirs in participant_dirs.items():
            session_data = {}
            print(f"session: {session}")
            for task_dir in session_dirs:
                if "nfb" in task_dir:
                    print(f"task: {task_dir}")
                    task_data = {}
                    h5file = os.path.join(data_directory, participant, session, task_dir, "experiment_data.h5") #"/Users/christopherturner/Documents/EEG_Data/ChrisPilot20220110/0-pre_task_ct01_01-10_16-07-00/experiment_data.h5"
                    # h5file = "/Users/christopherturner/Documents/EEG_Data/ChrisPilot20220110/0-post_task_ct01_01-10_16-55-15/experiment_data.h5"

                    # Put data in pandas data frame
                    df1, fs, channels, p_names = load_data(h5file)
                    df1['sample'] = df1.index

                    # Plot AAI of whole experiment
                    # fig = px.line(df1, x="sample", y="signal_AAI")
                    # fig.show()

                    df2 = af.get_protocol_data(df1, channels=channels, p_names=p_names, eog_filt=False, out="df")
                    aai_data = df2.loc[df2['channel'] == "signal_AAI"]
                    nfb_data = aai_data.loc[aai_data['block_name'] =='NFB'].reset_index(drop=True)
                    nfb_data_1 = nfb_data[: int(len(nfb_data)/4)].reset_index(drop=True)
                    nfb_data_2 = nfb_data[int(len(nfb_data)/4): 2*int(len(nfb_data)/4)].reset_index(drop=True)
                    nfb_data_3 = nfb_data[2*int(len(nfb_data)/4): 3*int(len(nfb_data)/4)].reset_index(drop=True)
                    nfb_data_4 = nfb_data[3*int(len(nfb_data)/4): -1].reset_index(drop=True)
                    baseline_data = aai_data.loc[aai_data['block_name'] =='baseline'].reset_index(drop=True)
                    fig = px.line(baseline_data, x=baseline_data.index, y="data", title=f"{participant}>{session}>{task_dir}: baseline aai")
                    fig.show()
                    fig = px.line(nfb_data, x=nfb_data.index, y="data", title=f"{participant}>{session}>{task_dir}: nfb aai")
                    fig.show()
                    fig = px.line(nfb_data_1, x=nfb_data_1.index, y="data", title=f"{participant}>{session}>{task_dir}: nfb aai")
                    fig.add_scatter(y=nfb_data_2['data'], name="2nd")
                    fig.add_scatter(y=nfb_data_3['data'], name="3rd")
                    fig.add_scatter(y=nfb_data_4['data'], name="4th")
                    fig.show()

                    fig = px.box(aai_data, x="block_name", y="data", notched=True,  title=f"{participant}>{session}>{task_dir}: aai data")
                    fig.show()

                    # get the protocol data and average the AAI for each protocol
                    protocol_data = af.get_protocol_data(df1, channels=channels, p_names=p_names, eog_filt=False)
                    nfb_aai_medians = []
                    all_aai_medians = []
                    previous_score = 0
                    score = []
                    for protocol, data in protocol_data.items():
                        if "nfb" in protocol.lower():
                            median_aai = data.loc[data['channel'] == "signal_AAI"]['data'].median()
                            nfb_aai_medians.append(median_aai)
                            score.append(data['reward'].iloc[-1] - previous_score)
                            previous_score = data['reward'].iloc[-1]
                        all_median_aai = data.loc[data['channel'] == "signal_AAI"]['data'].median()
                        all_aai_medians.append(all_median_aai)

                    # ----- Plot the nfb aai medians-----
                    fig = px.line(nfb_aai_medians, title=f"{participant}>{session}>{task_dir}")
                    fig.show()

                    nfb_fcb_medians = []
                    for protocol, data in protocol_data.items():
                        if "fc_b" in protocol.lower():
                            median_fcb = data.loc[data['channel'] == "signal_AAI"]['data'].median()
                            nfb_fcb_medians.append(median_fcb)

                    # ----- Plot the nfb aai medians-----
                    fig.add_scatter(y=nfb_fcb_medians, name="fcb_medians")
                    fig.show()

                    # ----- Plot the aai medians-----
                    fig = px.line(all_aai_medians, title=f"{participant}>{session}>{task_dir}")
                    fig.show()

                    # ----- Get the choice and answer results
                    choice_data = df1.loc[df1['choice'] != 0]
                    choice_data['choice_results'] = choice_data.apply(lambda row: 1 if row["choice"] == row["answer"] else 0, axis = 1)
                    fig = px.scatter(choice_data['choice_results'], title=f"{participant}>{session}>{task_dir}: choice time course")
                    fig.show()
                    dfg = choice_data.groupby(["choice_results"]).count().reset_index()
                    fig = px.bar(dfg,
                                 y=choice_data.groupby(["choice_results"]).size(),
                                 x="choice_results",
                                 color='choice_results',
                                 title='choices')
                    fig.show()
                    percent_correct = len(choice_data.loc[choice_data['choice_results'] == 1]) / len(choice_data)

                    # ----- Plot the score
                    px.bar(score, title=f"{participant}>{session}>{task_dir}: percent score")


                    # split AAI into first/second/third blocks for the participant

                    task_data["aai_1"] = nfb_aai_medians[0: int(len(nfb_aai_medians)/4)]
                    task_data["aai_2"] = nfb_aai_medians[int(len(nfb_aai_medians)/4): int(len(nfb_aai_medians)/4) * 2]
                    task_data["aai_3"] = nfb_aai_medians[(int(len(nfb_aai_medians)/4) * 2): int(len(nfb_aai_medians)/4) * 3]
                    task_data["aai_4"] = nfb_aai_medians[(int(len(nfb_aai_medians)/4) * 3): -1]
                    session_data[task_dir] = {'aai_medians': pd.DataFrame(task_data), 'score': score, "percent_correct": percent_correct}
                    pass
            participant_data["session_data"].append(session_data)
        experiment_data.append(participant_data)

# TODO: save experiment data off here - so can run in group data scripts after
for experiment in experiment_data:
    print(f'Participant: {experiment["participant_id"]}')
    for session_data in experiment['session_data']:
        for s_no, s_data in session_data.items():
            for s_name, section in s_data.items():
                print(s_name)
                if s_name == "aai_medians":
                    section_data = pd.melt(section, value_vars=['aai_1', 'aai_2', 'aai_3', 'aai_4'], var_name='section_number')
                    title = f'{experiment["participant_id"]}: {s_name}'
                    fig = px.box(section_data, x="section_number", y="value", title=title)
                    fig.show()
                    pass


# TODO: group analysis (permutation testing?)

# TODO: calculate brain AAI and do same analysis as above i.e. source power for lateralised source signals over filtered data for whole runs

pass