"""
File to get individual alpha from each test run
"""
import sys
import os

sys.path.append(f"{os.getcwd()}")


from philistine.mne import savgol_iaf
from utils.load_results import load_data
import mne
import pandas as pd

# ------ Get data files
data_directory = "/Users/christopherturner/Documents/EEG_Data/pilot_202201" # This is the directory where all participants are in

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

iaf_data = []
for participant, participant_dirs in experiment_dirs.items():
    participant_data = {"participant_id": participant, "session_data": []}
    if participant:# == "kk":
        for session, session_dirs in participant_dirs.items():
            session_data = {}
            for task_dir in session_dirs:
                if "nfb" in task_dir:# and session == 'source':
                    task_data = {}
                    h5file = os.path.join(data_directory, participant, session, task_dir, "experiment_data.h5")

                    # Put data in pandas data frame
                    df1, fs, channels, p_names = load_data(h5file)
                    df1['sample'] = df1.index

                    # get baseline blocks and block numbers
                    df1_all_bl = df1.loc[df1['block_name'].str.contains("baseline", case=False)]
                    df1_all_bl = df1_all_bl.loc[~df1_all_bl['block_name'].str.contains("start", case=False)]
                    baseline_blocks = df1_all_bl['block_number'].unique()

                    for block in baseline_blocks:

                    # # First rename baseline to baseline_eo
                    # if 'baseline' in df1['block_name'].values:
                    #     df1.loc[df1['block_name'] == "baseline", "block_name"] = "baseline_eo"
                    # df1_bl = df1.loc[df1['block_name'] == 'baseline_eo']
                    # # Get first eyes open
                    # baseline_first = df1_bl['block_number'].unique()[0]
                    # df1_bl = df1_bl.loc[df1_bl["block_number"] == baseline_first]

                        df1_bl = df1_all_bl.loc[df1_all_bl["block_number"] == block]
                        baseline_name = df1_bl["block_name"].unique()[0]

                        # Drop non eeg data
                        eeg_data = df1_bl.drop(
                            columns=['signal_Alpha_Left', 'signal_Alpha_Right', 'signal_AAI', 'events', 'reward', 'choice', 'answer', 'probe', 'block_name',
                                     'block_number', 'sample', 'MKIDX'])


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

                        # ----------get individual peak alpha
                        iaf_raw = savgol_iaf(m_raw, pink_max_r2=1, fmax=14).PeakAlphaFrequency
                        # _____------____----____--_-_-_-____-

                        print(f"participant, session, baseline_name, individual_alpha")
                        print(f"{participant}, {session}, {baseline_name}, {iaf_raw}")
                        iaf_data.append(pd.DataFrame(dict(participant=[participant], session=[session], baseline_name=[baseline_name], block_number=[block], iaf=[iaf_raw])))
iaf_data_all = pd.concat(iaf_data)
pass
